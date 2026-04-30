"""
CoexpressDeconvolve main module.

Multi-slice Visium deconvolution to in-silico single cells.
Pipeline: load -> density -> HVG -> manifold -> K-sweep -> LDA -> sampling -> placement -> export.

Multi-slice handling:
  visium_path can be a string (single slice) or a dict {name: path} / list (multi).
  Per-slice params (min_umi, anchor_mean_factor, low_slice_quality) accept either a
  scalar (broadcast to all slices) or a dict keyed by slice name.

LDA is fit per-slice and topics are aligned across slices by Hungarian matching on
cosine similarity of betas; the consensus beta is the mean of aligned betas.
Per-slice theta is then refit against the frozen consensus beta via a hand-rolled
variational E-step. The gene-coexpression manifold is joint over the intersected
gene set; per-slice beta projection extends consensus topics to each slice's full
cleaned gene list.
"""

import os
import re
import json
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

try:
    import numpy as np
    import pandas as pd
    import scipy.io
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.notebook import tqdm
    from sklearn.decomposition import FastICA, LatentDirichletAllocation
    from sklearn.preprocessing import StandardScaler, normalize
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from scipy.special import digamma
    import h5py
    import umap
except ImportError as e:
    missing_pkg = str(e).split()[-1]
    raise ImportError(
        f"Missing package: {missing_pkg}. "
        f"Please install required dependencies via:\n"
        f"pip install numpy pandas scipy matplotlib seaborn scikit-learn h5py tqdm umap-learn"
    )


# Module-level RNG state. Set via codeconv.set_seed(int) at the top of the notebook.
_SEED = 42
_RNG = np.random.default_rng(_SEED)


def set_seed(seed: int):
    """Set the module-wide random seed. Call once at the top of the notebook."""
    global _SEED, _RNG
    _SEED = int(seed)
    _RNG = np.random.default_rng(_SEED)
    np.random.seed(_SEED)
    print(f"codeconv: random seed set to {_SEED}")


# Container types

@dataclass
class SliceData:
    """Per-slice data accumulator. Steps progressively fill in fields."""
    name: str
    spatial_path: str
    counts: sp.csr_matrix
    gene_names: List[str]
    barcodes: List[str]
    coords: np.ndarray
    total_umi: np.ndarray
    scale_factors: dict
    # filled by step2
    n_cells: Optional[np.ndarray] = None
    engine_params: Optional[dict] = None
    # filled by step3 per-slice (post noise filter, slice-specific)
    counts_clean: Optional[sp.csr_matrix] = None
    genes_clean: Optional[List[str]] = None
    # filled by step6 per-slice
    theta: Optional[np.ndarray] = None
    beta_final: Optional[np.ndarray] = None


@dataclass
class HvgPack:
    """Output of step3. Joint HVG selection on the intersected gene set."""
    intersected_genes: List[str]
    hvg_names: List[str]
    hvg_per_slice: Dict[str, sp.csr_matrix]
    hvg_concat: sp.csr_matrix
    species: str


@dataclass
class Manifold:
    """Output of step4. Joint manifold on intersected genes."""
    embedding: np.ndarray
    intersected_genes: List[str]
    hvg_indices_in_intersected: List[int]
    species: str


@dataclass
class Model:
    """Output of step6. Consensus beta + per-slice theta."""
    n_topics: int
    hvg_names: List[str]
    beta_consensus: np.ndarray
    qc_df: pd.DataFrame
    per_slice_betas: Dict[str, np.ndarray]


# Helpers

def _load_config(config_path: str, species: str) -> dict:
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if species not in cfg["species_profiles"]:
        raise ValueError(
            f"Species '{species}' not in config. Available: {list(cfg['species_profiles'].keys())}"
        )
    profile = cfg["species_profiles"][species]
    profile["min_topic_percentage"] = cfg.get("min_topic_percentage", 0.05)
    profile["species"] = species
    return profile


def _normalize_paths(visium_path) -> Dict[str, str]:
    """Coerce a string / list / dict input into a {name: path} dict.

    Single string -> {basename: path}.
    List of strings -> {basename(p): p for p in list}; duplicate basenames are an error.
    Dict -> passthrough.
    """
    if isinstance(visium_path, str):
        name = os.path.basename(visium_path.rstrip('/')) or visium_path
        return {name: visium_path}
    if isinstance(visium_path, dict):
        return dict(visium_path)
    if isinstance(visium_path, (list, tuple)):
        names = [os.path.basename(p.rstrip('/')) or p for p in visium_path]
        if len(set(names)) != len(names):
            raise ValueError(
                "Duplicate slice basenames detected in visium_path list; "
                "use the dict form to disambiguate: {name: path, ...}."
            )
        return dict(zip(names, visium_path))
    raise TypeError(f"visium_path must be str, list, or dict, got {type(visium_path)}")


def _broadcast(param, slice_names: List[str], default=None):
    """Expand a scalar/dict param into a {name: value} dict aligned with slice_names."""
    if isinstance(param, dict):
        for k in slice_names:
            if k not in param:
                if default is not None:
                    param[k] = default
                else:
                    raise KeyError(f"Per-slice param missing entry for slice '{k}'.")
        return param
    if param is None:
        return {n: default for n in slice_names}
    return {n: param for n in slice_names}


def _variational_e_step(
    X: sp.csr_matrix,
    beta: np.ndarray,
    alpha: float,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> np.ndarray:
    """Estimate theta given fixed beta via mean-field variational LDA inference.

    Standard textbook update: gamma_d = alpha + sum_n count_n * phi_{n,k}
    with phi_{n,k} proportional to beta[k, w_n] * exp(digamma(gamma_d[k])).

    X: (n_docs, n_words) sparse counts.
    beta: (K, n_words), rows sum to 1.
    alpha: scalar Dirichlet prior on theta.
    """
    X_csr = X.tocsr()
    n_docs, n_words = X_csr.shape
    K = beta.shape[0]

    doc_lens = np.array(X_csr.sum(axis=1)).flatten()
    gamma = np.full((n_docs, K), float(alpha)) + (doc_lens[:, None] / K)

    for it in range(max_iter):
        gamma_old = gamma.copy()
        Elogtheta = digamma(gamma) - digamma(gamma.sum(axis=1, keepdims=True))
        expElogtheta = np.exp(Elogtheta)

        new_gamma = np.full((n_docs, K), float(alpha))
        for d in range(n_docs):
            start, end = X_csr.indptr[d], X_csr.indptr[d + 1]
            if start == end:
                continue
            cols = X_csr.indices[start:end]
            vals = X_csr.data[start:end].astype(float)

            phinorm = expElogtheta[d] @ beta[:, cols]
            phinorm = np.maximum(phinorm, 1e-100)
            ratio = vals / phinorm
            contrib = expElogtheta[d] * (beta[:, cols] @ ratio)
            new_gamma[d] = alpha + contrib

        gamma = new_gamma
        if np.mean(np.abs(gamma - gamma_old)) < tol:
            break

    theta = gamma / gamma.sum(axis=1, keepdims=True)
    return theta


def _align_topics(betas: Dict[str, np.ndarray], anchor: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Reorder per-slice topics so that index k means the same biological program in every slice.

    Hungarian matching on cosine similarity of beta rows against an anchor slice.
    Returns a dict of betas with rows reordered to anchor's topic order.
    """
    names = list(betas.keys())
    if anchor is None:
        anchor = names[0]
    anchor_beta = betas[anchor]

    def _row_normalize(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return M / norms

    anchor_norm = _row_normalize(anchor_beta)
    aligned = {anchor: anchor_beta}
    for s in names:
        if s == anchor:
            continue
        b = betas[s]
        b_norm = _row_normalize(b)
        sim = anchor_norm @ b_norm.T
        # Hungarian minimizes; we want to maximize similarity, so negate.
        row_ind, col_ind = linear_sum_assignment(-sim)
        new_b = np.zeros_like(b)
        new_b[row_ind] = b[col_ind]
        aligned[s] = new_b
    return aligned


def _safe_multinomial(rng, n, p):
    """multinomial sample that tolerates float roundoff.

    Tries the vanilla call first so well-formed inputs draw bit-for-bit identically
    to a direct rng.multinomial(n, p) call. Falls back to a clip + normalize + shave
    pass only when vanilla raises ValueError (numpy strictly checks
    pvals[:-1].sum() > 1.0 and rejects ULP-level overflow).
    """
    p = np.asarray(p, dtype=np.float64)
    try:
        return rng.multinomial(int(n), p)
    except ValueError:
        pass
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        out = np.zeros(len(p), dtype=int)
        out[0] = int(n)
        return out
    p = p / s
    overflow = p[:-1].sum() - 1.0
    if overflow > 0:
        idx = int(np.argmax(p[:-1]))
        p[idx] = max(0.0, p[idx] - overflow)
    return rng.multinomial(int(n), p)


# STEP 1: load

def step1_acquisition_and_anchoring(visium_path) -> Dict[str, SliceData]:
    """Load 10X Visium data per slice. Accepts str, list, or dict of paths.

    Returns a dict {slice_name: SliceData}.
    """
    start_time = time.perf_counter()
    paths = _normalize_paths(visium_path)
    out: Dict[str, SliceData] = {}

    for name, data_path in paths.items():
        print(f"\nStep 1 [{name}]: loading from {data_path}")
        spatial_dir = os.path.join(data_path, "spatial")
        h5_file = os.path.join(data_path, "filtered_feature_bc_matrix.h5")
        matrix_dir = os.path.join(data_path, "filtered_feature_bc_matrix")

        if os.path.exists(h5_file):
            print(f"   > h5 file: {h5_file}")
            with h5py.File(h5_file, 'r') as f:
                mat_group = f['matrix'] if 'matrix' in f else f
                data = mat_group['data'][:]
                indices = mat_group['indices'][:]
                indptr = mat_group['indptr'][:]
                shape = mat_group['shape'][:]
                counts = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
                if 'features' in mat_group:
                    feat_group = mat_group['features']
                    gene_names = [x.decode('utf-8') for x in feat_group['name'][:]]
                else:
                    gene_names = [x.decode('utf-8') for x in mat_group['genes'][:]]
                raw_barcodes = [x.decode('utf-8') for x in mat_group['barcodes'][:]]
        elif os.path.exists(matrix_dir):
            print(f"   > matrix dir: {matrix_dir}")
            counts = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz")).T.tocsr()
            features = pd.read_csv(os.path.join(matrix_dir, "features.tsv.gz"), header=None, sep='\t')
            gene_names = features[1].values.tolist()
            raw_barcodes = pd.read_csv(
                os.path.join(matrix_dir, "barcodes.tsv.gz"), header=None, sep='\t'
            )[0].values.tolist()
        else:
            raise FileNotFoundError(
                f"[{name}] No filtered_feature_bc_matrix.h5 or filtered_feature_bc_matrix/ in {data_path}"
            )

        # Spatial manifest
        pos_path = os.path.join(spatial_dir, "tissue_positions.csv")
        if not os.path.exists(pos_path):
            pos_path = os.path.join(spatial_dir, "tissue_positions_list.csv")
        has_header = 0
        if "list" in pos_path:
            has_header = None
        else:
            with open(pos_path, 'r') as f:
                first_line = f.readline()
                if "in_tissue" not in first_line and "barcode" not in first_line:
                    has_header = None

        spatial_df = pd.read_csv(pos_path, header=has_header)
        if len(spatial_df.columns) == 6:
            spatial_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
        else:
            spatial_df = spatial_df.rename(columns={
                'pxl_row_in_fullres': 'pxl_row',
                'pxl_col_in_fullres': 'pxl_col',
            })
        spatial_df = spatial_df.set_index('barcode')

        with open(os.path.join(spatial_dir, "scalefactors_json.json"), 'r') as f:
            scale_factors = json.load(f)

        # Barcode reconciliation
        if not any(b in spatial_df.index for b in raw_barcodes[:10]):
            print("   ! barcode mismatch; checking suffix conventions")
            if raw_barcodes[0].endswith("-1") and not spatial_df.index[0].endswith("-1"):
                raw_barcodes = [b.split('-')[0] for b in raw_barcodes]
            elif not raw_barcodes[0].endswith("-1") and spatial_df.index[0].endswith("-1"):
                raw_barcodes = [b + "-1" for b in raw_barcodes]

        valid_barcodes = [b for b in raw_barcodes if b in spatial_df.index]
        valid_indices = [raw_barcodes.index(b) for b in valid_barcodes]
        if len(valid_barcodes) == 0:
            raise ValueError(f"[{name}] no common barcodes between matrix and spatial data")

        counts = counts[valid_indices, :]
        coords = spatial_df.loc[valid_barcodes, ['pxl_row', 'pxl_col', 'array_row', 'array_col']].values
        total_counts_orig = np.array(counts.sum(axis=1)).flatten()

        # QC plot per slice
        mean_umi = np.mean(total_counts_orig)
        median_umi = np.median(total_counts_orig)
        p1 = np.percentile(total_counts_orig, 1)
        print(f"   median UMI/spot: {median_umi:.0f}   mean: {mean_umi:.0f}   1st pct: {p1:.0f}")

        plt.figure(figsize=(10, 6), dpi=100)
        bins = np.logspace(np.log10(max(1, total_counts_orig.min())), np.log10(total_counts_orig.max()), 50)
        plt.hist(total_counts_orig, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
        plt.axvline(median_umi, color='green', linestyle='--', label=f'Median: {int(median_umi)}')
        plt.axvline(p1, color='purple', linestyle=':', label=f'1st pct: {int(p1)}')
        plt.xscale('log')
        plt.title(f"[{name}] UMI per spot   N={len(valid_barcodes)} spots")
        plt.xlabel("Total UMI (log)")
        plt.ylabel("Spots")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.show()

        out[name] = SliceData(
            name=name,
            spatial_path=spatial_dir,
            counts=counts,
            gene_names=gene_names,
            barcodes=valid_barcodes,
            coords=coords,
            total_umi=total_counts_orig,
            scale_factors=scale_factors,
        )

    duration = time.perf_counter() - start_time
    print(f"\nStep 1 done in {duration:.2f}s   {len(out)} slice(s) loaded.")
    return out


# STEP 2: density

def step2_estimate_cell_density(
    slices: Dict[str, SliceData],
    config_path: str,
    species: str,
    min_umi=300,
    anchor_mean_factor=1.0,
    low_slice_quality=False,
    colormap='hsv',
) -> Dict[str, SliceData]:
    """Estimate per-spot cell counts from HK gene calibration.

    Per-slice min_umi and anchor_mean_factor accepted as scalar (broadcast) or dict.
    low_slice_quality=True (per-slice) enforces a floor of >=1 cell on every spot
    that passes the UMI gate.
    """
    start_time = time.perf_counter()
    profile = _load_config(config_path, species)
    hk_reference = profile['hk_profiles']
    engine_params = profile['engine_parameters']

    if not hk_reference:
        raise ValueError(
            f"Species '{species}' has no hk_profiles in config. "
            "Provide HK reference values for non-standard organisms."
        )

    names = list(slices.keys())
    min_umi_d = _broadcast(min_umi, names)
    anchor_d = _broadcast(anchor_mean_factor, names)
    low_q_d = _broadcast(low_slice_quality, names, default=False)

    for name in names:
        sd = slices[name]
        slice_min_umi = min_umi_d[name]
        slice_anchor = anchor_d[name]
        slice_low_q = low_q_d[name]

        print(f"\nStep 2 [{name}]: HK calibration (min_umi={slice_min_umi}, factor={slice_anchor}, low_quality={slice_low_q})")

        common_hk = [g for g in hk_reference.keys() if g in sd.gene_names]
        if not common_hk:
            raise ValueError(f"[{name}] No HK overlap between data and config for species='{species}'")
        print(f"   {len(common_hk)} HK genes used")

        hk_indices = [sd.gene_names.index(g) for g in common_hk]
        ref_values_log = np.array([hk_reference[g] for g in common_hk])

        safe_total = sd.total_umi.copy()
        safe_total[safe_total == 0] = 1
        hk_counts_raw = sd.counts[:, hk_indices].toarray()
        normalized_hk_log = np.log1p((hk_counts_raw / safe_total[:, np.newaxis]) * 10000)

        spot_hk_log_means = np.mean(normalized_hk_log, axis=1)
        standard_anchor_log_mean = np.mean(ref_values_log)
        adjusted_standard_log_mean = standard_anchor_log_mean + np.log(slice_anchor)

        spot_signal_linear = np.expm1(spot_hk_log_means)
        standard_signal_linear = np.expm1(adjusted_standard_log_mean)
        if standard_signal_linear < 0.001:
            standard_signal_linear = 0.001

        raw_n_cells = spot_signal_linear / standard_signal_linear
        n_cells = np.round(raw_n_cells).astype(int)

        is_low_quality = sd.total_umi < slice_min_umi
        n_cells[is_low_quality] = 0
        # Low-slice-quality floor: every spot above UMI gate gets at least 1 cell
        if slice_low_q:
            floor_mask = (~is_low_quality) & (n_cells == 0)
            n_cells[floor_mask] = 1

        # Calibration plot
        plt.figure(figsize=(8, 5), dpi=100)
        valid_signals = spot_signal_linear[sd.total_umi > slice_min_umi]
        max_x = max(np.percentile(valid_signals, 99) if len(valid_signals) > 0 else 10, standard_signal_linear * 2)
        bins = np.linspace(0, max_x, 50)
        plt.hist(valid_signals, bins=bins, color='purple', alpha=0.6, label='Spot HK signal')
        plt.axvline(standard_signal_linear, color='red', linewidth=2, linestyle='--',
                    label=f'Standard ({standard_signal_linear:.2f})')
        plt.title(f"[{name}] Calibration check (factor={slice_anchor})")
        plt.xlabel("Geometric mean of HK normalized expression")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        n_zeros = int(np.sum(n_cells == 0))
        n_low = int(sum(is_low_quality))
        print(f"   empty spots: {n_zeros}/{len(n_cells)} ({n_zeros/len(n_cells)*100:.1f}%)   filtered by UMI: {n_low}")

        # Density histogram
        plt.figure(figsize=(8, 5), dpi=100)
        max_val = int(np.max(n_cells)) if len(n_cells) > 0 else 0
        if max_val > 0:
            cell_bins = np.arange(1, max_val + 2) - 0.5
            plt.hist(n_cells[n_cells > 0], bins=cell_bins,
                     color='#27ae60', edgecolor='white', alpha=0.9, label='Tissue')
        plt.bar(0, n_zeros, color='#95a5a6', edgecolor='white', width=0.8, label='Background')
        plt.title(f"[{name}] Cell density")
        plt.xlabel("Cells per spot")
        plt.xticks(np.arange(0, max(1, max_val) + 1, 1))
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.show()

        # Spatial map
        y = sd.coords[:, 0]
        x = sd.coords[:, 1]
        plt.figure(figsize=(8, 8), dpi=100)
        bg_mask = n_cells == 0
        plt.scatter(x[bg_mask], y[bg_mask], c='grey', s=10, alpha=0.3)
        fg_mask = n_cells > 0
        if np.any(fg_mask):
            sc = plt.scatter(x[fg_mask], y[fg_mask], c=n_cells[fg_mask],
                             cmap=colormap, s=15, linewidth=0)
            plt.colorbar(sc, label='Cells per spot', fraction=0.046, pad=0.04)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.title(f"[{name}] spatial density")
        plt.show()

        print(f"   total cells on slide: {int(sum(n_cells))}")

        sd.n_cells = n_cells
        sd.engine_params = engine_params

    duration = time.perf_counter() - start_time
    print(f"\nStep 2 done in {duration:.2f}s")
    return slices


# STEP 3: HVG selection (joint, on intersected gene set)

def step3_feature_selection(
    slices: Dict[str, SliceData],
    config_path: str,
    species: str,
    n_hvg: int = 2000,
) -> HvgPack:
    """Per-slice noise filter, intersect across slices, then HVG select on concatenated counts.

    Each slice's counts_clean and genes_clean are filled in. HvgPack returns the joint HVG
    set on the intersected gene index space.
    """
    start_time = time.perf_counter()
    profile = _load_config(config_path, species)
    noise_pattern = profile.get('noise_regex', '')

    print(f"Step 3: noise filter (species={species})")
    if noise_pattern:
        print(f"   regex: {noise_pattern}")
        bio_re = re.compile(noise_pattern, re.IGNORECASE)
    else:
        bio_re = None
        print("   (no noise filter for this species)")

    # Per-slice noise filter
    for name, sd in slices.items():
        if bio_re is None:
            keep_indices = list(range(len(sd.gene_names)))
            genes_kept = list(sd.gene_names)
        else:
            keep_indices = []
            genes_kept = []
            for i, g in enumerate(sd.gene_names):
                if not bio_re.match(g):
                    keep_indices.append(i)
                    genes_kept.append(g)
        sd.counts_clean = sd.counts[:, keep_indices]
        sd.genes_clean = genes_kept
        n_dropped = len(sd.gene_names) - len(genes_kept)
        print(f"   [{name}] removed {n_dropped} noise genes; kept {len(genes_kept)}")

    # Intersect gene sets
    if len(slices) == 1:
        only_name = next(iter(slices.keys()))
        intersected = list(slices[only_name].genes_clean)
    else:
        sets = [set(sd.genes_clean) for sd in slices.values()]
        intersected = sorted(set.intersection(*sets)) if len(sets) > 0 else []
    print(f"\nStep 3: gene set size = {len(intersected)}")
    if not intersected:
        raise ValueError("Empty intersection of slice gene sets!")

    # Per-slice expression matrix on the intersected order
    intersected_set = set(intersected)
    inter_counts_per_slice: Dict[str, sp.csr_matrix] = {}
    for name, sd in slices.items():
        gene_to_idx = {g: i for i, g in enumerate(sd.genes_clean)}
        ord_idx = [gene_to_idx[g] for g in intersected]
        inter_counts_per_slice[name] = sd.counts_clean[:, ord_idx]

    # Concatenate (rows = spots) for joint HVG calculation
    inter_concat = sp.vstack([inter_counts_per_slice[n] for n in slices.keys()]).tocsr()

    # HVG selection on the joint matrix using Seurat-style log-variance dispersion
    print(f"Step 3: selecting top {n_hvg} HVGs from joint intersected matrix...")
    row_sums = np.array(inter_concat.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    norm_counts = inter_concat.copy().astype(float)
    norm_counts.data /= np.repeat(row_sums, np.diff(norm_counts.indptr))
    norm_counts.data *= 10000.0
    norm_counts.data = np.log1p(norm_counts.data)

    mean_expr = np.array(norm_counts.mean(axis=0)).flatten()
    sq_counts = norm_counts.copy()
    sq_counts.data **= 2
    mean_sq = np.array(sq_counts.mean(axis=0)).flatten()
    var_expr = mean_sq - (mean_expr ** 2)

    # Bin-normalized dispersion
    n_bins = 20
    bins = np.linspace(np.min(mean_expr), np.max(mean_expr), n_bins + 1)
    dispersion_norm = np.zeros_like(var_expr)
    for i in range(n_bins):
        idx = np.where((mean_expr >= bins[i]) & (mean_expr < bins[i + 1]))[0]
        if len(idx) > 0:
            bin_var = var_expr[idx]
            bin_mean_var = np.mean(bin_var)
            bin_std_var = np.std(bin_var)
            if bin_std_var > 0:
                dispersion_norm[idx] = (bin_var - bin_mean_var) / bin_std_var

    n_hvg_eff = min(n_hvg, len(intersected))
    hvg_indices_local = np.argsort(dispersion_norm)[-n_hvg_eff:][::-1]
    hvg_names = [intersected[i] for i in hvg_indices_local]

    # Per-slice HVG count matrices
    hvg_per_slice: Dict[str, sp.csr_matrix] = {
        name: inter_counts_per_slice[name][:, hvg_indices_local] for name in slices.keys()
    }
    hvg_concat = inter_concat[:, hvg_indices_local]

    # Diagnostic plot
    plt.figure(figsize=(9, 7))
    plt.scatter(mean_expr, dispersion_norm, s=1, color='grey', alpha=0.5, label='Non-HVG')
    plt.scatter(mean_expr[hvg_indices_local], dispersion_norm[hvg_indices_local], s=1, color='red', label='HVG')
    plt.xlabel('Mean expression (log)')
    plt.ylabel('Normalized dispersion')
    plt.title(f'Step 3: joint HVG selection ({n_hvg_eff} of {len(intersected)} intersected genes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    duration = time.perf_counter() - start_time
    print(f"Step 3 done in {duration:.2f}s   joint HVG matrix: {hvg_concat.shape}")
    return HvgPack(
        intersected_genes=intersected,
        hvg_names=hvg_names,
        hvg_per_slice=hvg_per_slice,
        hvg_concat=hvg_concat,
        species=species,
    )


# STEP 4: joint manifold (over intersected genes)

def step4_gene_manifold(
    slices: Dict[str, SliceData],
    hvg_pack: HvgPack,
    config_path: str,
    n_components: int = 30,
) -> Manifold:
    """ICA + UMAP on the gene matrix to produce a gene manifold.

    Single slice: uses the slice's counts_clean directly in its native gene order
    (matches legacy codeconv step4 behavior bit-for-bit modulo upstream tie-breaking).
    Multi-slice: uses the alphabetical-intersected stacked matrix so all slices share
    a common gene index space.
    """
    start_time = time.perf_counter()
    species = hvg_pack.species
    profile = _load_config(config_path, species)
    qc_markers = profile.get('qc_markers', [])

    n_slices = len(slices)
    if n_slices == 1:
        only_name = next(iter(slices.keys()))
        sd = slices[only_name]
        inter_concat = sd.counts_clean
        manifold_genes = list(sd.genes_clean)
        print(f"Step 4: building gene manifold over {inter_concat.shape[1]} genes (single-slice native order)...")
    else:
        inter_concat_per_slice = []
        for name, sd in slices.items():
            gene_to_idx = {g: i for i, g in enumerate(sd.genes_clean)}
            ord_idx = [gene_to_idx[g] for g in hvg_pack.intersected_genes]
            inter_concat_per_slice.append(sd.counts_clean[:, ord_idx])
        inter_concat = sp.vstack(inter_concat_per_slice).tocsr()
        manifold_genes = list(hvg_pack.intersected_genes)
        print(f"Step 4: building gene manifold over {inter_concat.shape[1]} intersected genes...")

    # Transpose: now rows are genes, columns are spots-across-slices
    X_genes = inter_concat.T
    X_dense = np.log1p(X_genes.toarray())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    print(f"   FastICA n_components={n_components}")
    ica = FastICA(n_components=n_components, random_state=_SEED, max_iter=1000, tol=0.005)
    X_ica = ica.fit_transform(X_scaled)

    print("   UMAP cosine projection")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=_SEED,
    )
    embedding = reducer.fit_transform(X_ica)

    # HVG positions within the manifold's gene index space
    inter_index = {g: i for i, g in enumerate(manifold_genes)}
    hvg_indices_in_intersected = [inter_index[g] for g in hvg_pack.hvg_names]

    # Visualization
    plt.figure(figsize=(8, 7), dpi=100)
    plt.scatter(embedding[:, 0], embedding[:, 1], s=2, c='lightgrey', alpha=0.4, label='Genes')

    if qc_markers:
        colors = plt.cm.hsv(np.linspace(0, 1, len(qc_markers)))
        found = False
        for idx, gene in enumerate(qc_markers):
            if gene in inter_index:
                gi = inter_index[gene]
                plt.scatter(embedding[gi, 0], embedding[gi, 1],
                            s=100, color=colors[idx], edgecolors='black', label=gene, zorder=10)
                plt.text(embedding[gi, 0] + 0.1, embedding[gi, 1] + 0.1, gene, fontsize=9)
                found = True
        if found:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        print("   (no QC markers configured for this species; manifold shown un-annotated)")

    plt.title("Step 4: joint gene co-expression manifold (ICA+UMAP)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()

    duration = time.perf_counter() - start_time
    print(f"Step 4 done in {duration:.2f}s")
    return Manifold(
        embedding=embedding,
        intersected_genes=manifold_genes,
        hvg_indices_in_intersected=hvg_indices_in_intersected,
        species=species,
    )


# STEP 5: K sweep (per-slice; joint and heatmap only when multi-slice)

def step5_ksweep(
    hvg_pack: HvgPack,
    min_k: int = 3,
    max_k: int = 20,
    step: int = 1,
    subsample_frac: float = 1.0,
    doc_topic_prior: float = 0.1,
    topic_word_prior: float = 0.01,
) -> dict:
    """Run LDA perplexity sweep per slice. With multiple slices, also runs a joint sweep
    on the concatenated HVG matrix and shows a relative-perplexity heatmap for cross-slice
    comparison. Single-slice runs skip the joint sweep and the heatmap (both redundant).
    """
    start_time = time.perf_counter()
    ks = list(range(min_k, max_k + 1, step))
    sweep: Dict[str, List[float]] = {}

    n_slices = len(hvg_pack.hvg_per_slice)
    targets = dict(hvg_pack.hvg_per_slice)
    if n_slices > 1:
        targets['_joint'] = hvg_pack.hvg_concat

    for label, X in targets.items():
        n_spots = X.shape[0]
        if subsample_frac < 1.0 and n_spots > 1000:
            n_sub = int(n_spots * subsample_frac)
            rng = np.random.default_rng(_SEED)
            idx = rng.choice(n_spots, n_sub, replace=False)
            X_use = X[idx, :]
            print(f"Step 5 [{label}]: {n_sub}/{n_spots} subsample")
        else:
            X_use = X
            print(f"Step 5 [{label}]: full {n_spots} spots")

        perps = []
        for k in tqdm(ks, desc=f"LDA sweep [{label}]"):
            lda = LatentDirichletAllocation(
                n_components=k,
                learning_method='online',
                learning_offset=50.,
                max_iter=5,
                random_state=_SEED,
                n_jobs=-1,
                doc_topic_prior=doc_topic_prior,
                topic_word_prior=topic_word_prior,
            )
            lda.fit(X_use)
            perps.append(lda.perplexity(X_use))
        sweep[label] = perps

    # Plot 1: line plot of perplexity vs K
    plt.figure(figsize=(10, 5))
    for label, perps in sweep.items():
        lw = 2.5 if label == '_joint' else 1.5
        plt.plot(ks, perps, marker='o', markerfacecolor='white', label=label, linewidth=lw)
    plt.xlabel("K topics")
    plt.ylabel("Perplexity (lower = better fit)")
    title = "Step 5: K-sweep" + (", per slice + joint" if n_slices > 1 else "")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: cross-slice heatmap (only meaningful with 2+ slices)
    slice_labels = [k for k in sweep.keys() if k != '_joint']
    if len(slice_labels) > 1:
        mat = np.zeros((len(slice_labels), len(ks)))
        for i, lbl in enumerate(slice_labels):
            row = np.array(sweep[lbl], dtype=float)
            rng = row.max() - row.min()
            mat[i] = (row - row.min()) / rng if rng > 0 else 0.0
        plt.figure(figsize=(max(8, len(ks) * 0.5), 0.6 * len(slice_labels) + 2))
        sns.heatmap(mat, xticklabels=ks, yticklabels=slice_labels,
                    cmap='viridis_r', annot=False, cbar_kws={'label': 'relative perplexity'})
        raw_mat = np.array([sweep[lbl] for lbl in slice_labels])
        median_per_k = np.median(raw_mat, axis=0)
        best_k_idx = int(np.argmin(median_per_k))
        plt.axvline(best_k_idx + 0.5, color='red', linestyle='--', linewidth=1.5)
        plt.title(f"Step 5: relative perplexity by K (red line = median-best K = {ks[best_k_idx]})")
        plt.xlabel("K")
        plt.ylabel("Slice")
        plt.tight_layout()
        plt.show()

    duration = time.perf_counter() - start_time
    print(f"Step 5 done in {duration:.2f}s")
    return {label: dict(zip(ks, perps)) for label, perps in sweep.items()}


# STEP 6: per-slice LDA -> Hungarian alignment -> consensus beta -> per-slice theta refit + projection

def step6_final_deconvolution(
    slices: Dict[str, SliceData],
    hvg_pack: HvgPack,
    manifold: Manifold,
    n_topics: int,
    n_iters: int = 30,
    k_neighbors: int = 3,
    min_sim: float = 0.05,
    doc_topic_prior: float = 0.1,
    topic_word_prior: float = 0.01,
    e_step_iters: int = 50,
) -> Model:
    """LDA fit per slice. With multiple slices, topics are aligned across slices via
    Hungarian matching on cosine similarity of betas, a mean-consensus beta is built,
    and per-slice theta is refit against the frozen consensus beta via variational E-step.
    With a single slice, the LDA beta is the model directly. Per-slice manifold-guided
    projection then extends consensus topics to each slice's full genes_clean.
    """
    start_time = time.perf_counter()
    n_slices = len(slices)
    n_hvg = len(hvg_pack.hvg_names)
    print(f"Step 6: LDA fit, K={n_topics}   ({n_slices} slice{'s' if n_slices > 1 else ''})")

    per_slice_betas_unaligned: Dict[str, np.ndarray] = {}
    per_slice_thetas_lda: Dict[str, np.ndarray] = {}
    for name in slices.keys():
        X = hvg_pack.hvg_per_slice[name]
        print(f"   [{name}] LDA on {X.shape[0]} spots")
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='batch',
            max_iter=n_iters,
            random_state=_SEED,
            n_jobs=-1,
            verbose=0,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
        )
        theta_lda = lda.fit_transform(X)
        beta = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
        per_slice_betas_unaligned[name] = beta
        per_slice_thetas_lda[name] = theta_lda

    if n_slices > 1:
        # Multi-slice path: align topics, build consensus beta, refit theta with frozen beta
        print("Step 6: Hungarian alignment of topics across slices")
        aligned_betas = _align_topics(per_slice_betas_unaligned)

        print("Step 6: building consensus beta (mean across aligned slices)")
        beta_stack = np.stack([aligned_betas[n] for n in slices.keys()], axis=0)
        beta_consensus = beta_stack.mean(axis=0)
        beta_consensus = beta_consensus / beta_consensus.sum(axis=1, keepdims=True)

        print(f"Step 6: per-slice theta refit (variational E-step, max_iter={e_step_iters})")
        for name, sd in slices.items():
            X = hvg_pack.hvg_per_slice[name]
            theta = _variational_e_step(X, beta_consensus, alpha=doc_topic_prior, max_iter=e_step_iters)
            sd.theta = theta
            print(f"   [{name}] theta {theta.shape}")
        # Multi-slice projection: per-slice with cosine fallback for slice-specific genes
        print("Step 6: projecting topics to full genes_clean per slice")
        inter_index = {g: i for i, g in enumerate(manifold.intersected_genes)}
        hvg_idx_in_intersected = manifold.hvg_indices_in_intersected
        hvg_coords_manifold = manifold.embedding[hvg_idx_in_intersected]

        per_slice_betas_full: Dict[str, np.ndarray] = {}
        for name, sd in slices.items():
            genes_s = sd.genes_clean
            n_genes_s = len(genes_s)
            beta_full = np.zeros((n_topics, n_genes_s))

            slice_gene_to_idx = {g: i for i, g in enumerate(genes_s)}
            hvg_idx_in_slice = [slice_gene_to_idx[g] for g in hvg_pack.hvg_names]

            norm_all = normalize(sd.counts_clean.T, axis=1)
            norm_hvg = normalize(sd.counts_clean[:, hvg_idx_in_slice].T, axis=1)
            sim_full = norm_all @ norm_hvg.T

            for i, g in enumerate(genes_s):
                sim_row = sim_full[i].toarray().flatten() if sp.issparse(sim_full) else sim_full[i]
                if g in inter_index:
                    gi = inter_index[g]
                    g_pos = manifold.embedding[gi]
                    dists = np.linalg.norm(hvg_coords_manifold - g_pos, axis=1)
                    neighbors = np.argsort(dists)[:k_neighbors]
                else:
                    neighbors = np.argsort(-sim_row)[:k_neighbors]

                weights = sim_row[neighbors]
                if np.max(weights) < min_sim:
                    continue
                proj = beta_consensus[:, neighbors] @ weights
                if proj.sum() > 0:
                    beta_full[:, i] = proj / proj.sum()

            sd.beta_final = beta_full
            per_slice_betas_full[name] = beta_full
            print(f"   [{name}] beta_final {beta_full.shape}")

        # Multi-slice QC: from beta_consensus on HVG (per-slice betas differ, HVG is the shared basis)
        topic_dict = {}
        for k in range(n_topics):
            top_idx = beta_consensus[k].argsort()[::-1][:15]
            topic_dict[f"Topic_{k}"] = [hvg_pack.hvg_names[i] for i in top_idx]
        qc_df = pd.DataFrame(topic_dict)
    else:
        # Single-slice path mirrors the legacy codeconv step6: LDA fit_transform
        # gives theta and beta_hvg, then a single cdist-based projection extends
        # beta to the full genes_clean. QC table is built from the projected beta_final.
        only_name = next(iter(slices.keys()))
        sd = slices[only_name]
        sd.theta = per_slice_thetas_lda[only_name]
        beta_hvg = per_slice_betas_unaligned[only_name]
        beta_consensus = beta_hvg
        print(f"   [{only_name}] theta {sd.theta.shape}")

        print("Step 6: projecting latent topics using UMAP manifold topology")
        genes_s = sd.genes_clean
        n_genes_s = len(genes_s)

        # Manifold positions: for a single slice, every gene in genes_clean is in
        # the intersected set (intersection of one set is itself), so all genes have
        # manifold coordinates.
        inter_index = {g: i for i, g in enumerate(manifold.intersected_genes)}
        hvg_coords_manifold = manifold.embedding[manifold.hvg_indices_in_intersected]
        all_coords = manifold.embedding[[inter_index[g] for g in genes_s]]
        dist_matrix = cdist(all_coords, hvg_coords_manifold, metric='euclidean')

        # Cosine similarity in expression space
        slice_gene_to_idx = {g: i for i, g in enumerate(genes_s)}
        hvg_idx_in_slice = [slice_gene_to_idx[g] for g in hvg_pack.hvg_names]
        norm_all = normalize(sd.counts_clean.T, axis=1)
        norm_hvg = normalize(sd.counts_clean[:, hvg_idx_in_slice].T, axis=1)
        similarity = norm_all @ norm_hvg.T

        beta_final = np.zeros((n_topics, n_genes_s))
        for i in range(n_genes_s):
            umap_neighbors_idx = np.argsort(dist_matrix[i])[:k_neighbors]
            sim_row = similarity[i].toarray().flatten() if sp.issparse(similarity) else similarity[i]
            weights = sim_row[umap_neighbors_idx]
            if np.max(weights) < min_sim:
                continue
            proj = beta_hvg[:, umap_neighbors_idx] @ weights
            if proj.sum() > 0:
                beta_final[:, i] = proj / proj.sum()

        sd.beta_final = beta_final
        per_slice_betas_full = {only_name: beta_final}
        print(f"   [{only_name}] beta_final {beta_final.shape}")

        # Single-slice QC: top genes per topic from beta_final on full genes_clean (matches legacy behavior)
        topic_dict = {}
        for k in range(n_topics):
            top_idx = beta_final[k].argsort()[::-1][:15]
            topic_dict[f"Topic_{k}"] = [genes_s[i] for i in top_idx]
        qc_df = pd.DataFrame(topic_dict)

    duration = time.perf_counter() - start_time
    print(f"Step 6 done in {duration:.2f}s")
    return Model(
        n_topics=n_topics,
        hvg_names=list(hvg_pack.hvg_names),
        beta_consensus=beta_consensus,
        qc_df=qc_df,
        per_slice_betas=per_slice_betas_full,
    )


# STEP 7: sampling engine (per-slice, with rescue)

def step7_sampling_engine(
    slices: Dict[str, SliceData],
    model: Model,
    config_path: str,
    species: str,
    low_slice_quality=False,
    min_topic_percentage: Optional[float] = None,
) -> Dict[str, dict]:
    """Per-slice hierarchical Bayesian sampling. Returns dict of per-slice cell outputs.

    Threshold-and-renormalize is always applied: any topic with theta < min_topic_percentage
    is zeroed and theta is renormalized. low_slice_quality=True per slice additionally
    enforces an inflate rescue: every surviving topic gets at least 1 cell.
    """
    start_time = time.perf_counter()
    profile = _load_config(config_path, species)
    cfg_min_pct = profile.get('min_topic_percentage', 0.05)
    if min_topic_percentage is None:
        min_topic_percentage = cfg_min_pct

    names = list(slices.keys())
    low_q_d = _broadcast(low_slice_quality, names, default=False)

    out: Dict[str, dict] = {}
    # Use the legacy RandomState (Mersenne Twister) so that multinomial / gamma
    # draws match the legacy codeconv step7 sequence sample-for-sample under the
    # same seed. PCG64 (np.random.default_rng) would draw a different sequence.
    rng = np.random.RandomState(_SEED)

    for name in names:
        sd = slices[name]
        slice_low_q = low_q_d[name]
        engine_params = sd.engine_params
        gamma_shape = 1.0 / engine_params['phi']
        gamma_scale = engine_params['mu'] * engine_params['phi']

        n_spots, n_genes = sd.counts_clean.shape
        n_topics = model.n_topics
        beta = sd.beta_final  # (K, n_genes_s)
        theta = sd.theta      # (n_spots, K)

        rows, cols, data = [], [], []
        cell_metadata = []
        global_cell_idx = 0
        n_rescued = 0

        print(f"\nStep 7 [{name}]: sampling   low_q={slice_low_q}   min_pct={min_topic_percentage}")
        for s in tqdm(range(n_spots), desc=f"[{name}] cells", unit="spot"):
            n_total = int(sd.n_cells[s])
            if n_total == 0:
                continue

            # Threshold-and-renormalize theta. Bypassed entirely when min_topic_percentage<=0
            # so that legacy single-slice runs see theta_eff identical to theta[s] (no copy,
            # no float drift from renormalization).
            if min_topic_percentage > 0:
                theta_eff = theta[s].copy()
                theta_eff[theta_eff < min_topic_percentage] = 0.0
                tsum = theta_eff.sum()
                if tsum <= 0:
                    # Spot has no surviving topics; fall back to argmax of original theta
                    theta_eff = np.zeros_like(theta_eff)
                    theta_eff[int(np.argmax(theta[s]))] = 1.0
                    tsum = 1.0
                theta_eff /= tsum
            else:
                theta_eff = theta[s]

            topic_dist = _safe_multinomial(rng, n_total, theta_eff)

            # Inflate rescue for low-quality slices
            if slice_low_q:
                surviving = np.where(theta_eff > 0)[0]
                for k in surviving:
                    if topic_dist[k] == 0:
                        topic_dist[k] = 1
                        n_rescued += 1

            # Generate in-silico cells with gamma weights per topic group
            cells_weights = {}
            cells_indices = {}
            current_spot_cell_idx = 0
            for k in range(n_topics):
                n_k = int(topic_dist[k])
                if n_k > 0:
                    w = rng.gamma(gamma_shape, gamma_scale, size=n_k)
                    if w.sum() == 0:
                        w = np.ones(n_k)
                    cells_weights[k] = w / w.sum()
                    for i in range(n_k):
                        cell_metadata.append({
                            'spot_idx': s,
                            'topic_idx': k,
                            'cell_num': current_spot_cell_idx + i + 1,
                        })
                    cells_indices[k] = [global_cell_idx + i for i in range(n_k)]
                    global_cell_idx += n_k
                    current_spot_cell_idx += n_k

            # UMI allocation per gene present in this spot
            spot_vec = sd.counts_clean[s].toarray().flatten()
            genes_present = np.where(spot_vec > 0)[0]
            for g in genes_present:
                count = int(spot_vec[g])
                p_topic = beta[:, g] * theta_eff
                if p_topic.sum() == 0:
                    # Match legacy codeconv: uniform fallback when projected beta is zero
                    # across topics for this gene.
                    p_topic = np.ones(n_topics) / n_topics
                else:
                    p_topic = p_topic / p_topic.sum()
                umi_per_topic = _safe_multinomial(rng, count, p_topic)

                for k in range(n_topics):
                    u_count = int(umi_per_topic[k])
                    if u_count > 0 and k in cells_weights:
                        umi_per_cell = _safe_multinomial(rng, u_count, cells_weights[k])
                        for ci, val in enumerate(umi_per_cell):
                            if val > 0:
                                rows.append(cells_indices[k][ci])
                                cols.append(g)
                                data.append(val)

        final_matrix = sp.csr_matrix((data, (rows, cols)), shape=(global_cell_idx, n_genes))
        print(f"   [{name}] generated {global_cell_idx} cells   rescued {n_rescued} topic slots")
        out[name] = {
            'matrix': final_matrix,
            'cell_metadata': cell_metadata,
            'n_cells': global_cell_idx,
            'n_rescued': n_rescued,
        }

    duration = time.perf_counter() - start_time
    print(f"\nStep 7 done in {duration:.2f}s")
    return out


# STEP 8: placement (per-slice)

def step8_geometry_and_placement(
    cells: Dict[str, dict],
    slices: Dict[str, SliceData],
):
    """Place each slice's in-silico cells around their spot centers using Fibonacci shells."""
    start_time = time.perf_counter()
    phi_golden = np.pi * (3. - np.sqrt(5.))

    for name, payload in cells.items():
        sd = slices[name]
        diameter = sd.scale_factors['spot_diameter_fullres']
        cell_metadata = payload['cell_metadata']

        spot_totals: Dict[int, int] = {}
        for meta in cell_metadata:
            spot_totals[meta['spot_idx']] = spot_totals.get(meta['spot_idx'], 0) + 1

        final_barcodes: List[str] = []
        final_coords: List[List[float]] = []
        spot_counters: Dict[int, int] = {}

        for meta in cell_metadata:
            s = meta['spot_idx']
            t = meta['topic_idx']
            idx = spot_counters.get(s, 0)
            spot_counters[s] = idx + 1
            n_total = spot_totals[s]

            cy = sd.coords[s, 0]
            cx = sd.coords[s, 1]
            r = diameter * np.sqrt(idx) / np.sqrt(n_total) if n_total > 1 else 0
            ang = idx * phi_golden
            py = cy + r * np.sin(ang)
            px = cx + r * np.cos(ang)
            ay, ax = sd.coords[s, 2], sd.coords[s, 3]

            final_coords.append([py, px, ay, ax])
            orig_bc = sd.barcodes[s]
            new_bc = f"{orig_bc}-Topic-{t}-Cell-{meta['cell_num']}"
            final_barcodes.append(new_bc)

        payload['final_barcodes'] = final_barcodes
        payload['final_coords'] = np.array(final_coords)
        print(f"Step 8 [{name}]: placed {len(final_barcodes)} cells")

    duration = time.perf_counter() - start_time
    print(f"Step 8 done in {duration:.2f}s")


# STEP 9: export to 10X-compatible format (per-slice)

def step9_export_results(
    cells: Dict[str, dict],
    slices: Dict[str, SliceData],
    output_folder: str,
):
    """Per-slice export to slice_<name>/deconvolved/ in 10X SpaceRanger HDF5 layout.

    For a single slice run, the slice_<name> wrapper is still applied so behavior is
    consistent across single and multi-slice modes.
    """
    start_time = time.perf_counter()
    os.makedirs(output_folder, exist_ok=True)

    summary = {}
    for name, payload in cells.items():
        sd = slices[name]
        slice_dir = os.path.join(output_folder, f"slice_{name}")
        out_dir = os.path.join(slice_dir, "deconvolved")
        spa_dir = os.path.join(out_dir, "spatial")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(spa_dir, exist_ok=True)

        h5_path = os.path.join(out_dir, "filtered_feature_bc_matrix.h5")
        print(f"Step 9 [{name}]: writing {h5_path}")
        genes = sd.genes_clean
        barcodes = payload['final_barcodes']
        final_matrix = payload['matrix']

        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("matrix")
            grp.create_dataset("features/id", data=np.array(genes, dtype='S'))
            grp.create_dataset("features/name", data=np.array(genes, dtype='S'))
            grp.create_dataset("features/feature_type",
                               data=np.array(["Gene Expression"] * len(genes), dtype='S'))
            grp.create_dataset("features/genome", data=np.array(["Genome"] * len(genes), dtype='S'))
            grp.create_dataset("features/_all_tag_keys", data=np.array([], dtype='S'))
            grp.create_dataset("barcodes", data=np.array(barcodes, dtype='S'))
            csc = final_matrix.T.tocsc()
            grp.create_dataset("data", data=csc.data)
            grp.create_dataset("indices", data=csc.indices)
            grp.create_dataset("indptr", data=csc.indptr)
            grp.create_dataset("shape", data=np.array(csc.shape, dtype='i4'))

        # Copy spatial assets
        for fname in ['tissue_lowres_image.png', 'tissue_hires_image.png', 'scalefactors_json.json']:
            src = os.path.join(sd.spatial_path, fname)
            dst = os.path.join(spa_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)

        coords = payload['final_coords']
        pos_df = pd.DataFrame(coords, columns=['pxl_row', 'pxl_col', 'array_row', 'array_col'])
        pos_df['barcode'] = barcodes
        pos_df['in_tissue'] = 1
        pos_df = pos_df[['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']]
        pos_df.to_csv(os.path.join(spa_dir, "tissue_positions.csv"), index=False)

        # Per-slice summary
        slice_summary = {
            'slice_name': name,
            'n_spots': int(sd.counts_clean.shape[0]),
            'n_cells_generated': int(payload['n_cells']),
            'n_topics': int(sd.theta.shape[1]) if sd.theta is not None else None,
            'n_rescued_topic_slots': int(payload.get('n_rescued', 0)),
            'gene_count_export': int(len(genes)),
        }
        with open(os.path.join(slice_dir, "run_summary.json"), 'w') as fout:
            json.dump(slice_summary, fout, indent=2)
        summary[name] = slice_summary
        print(f"   [{name}] cells={slice_summary['n_cells_generated']}   genes={slice_summary['gene_count_export']}")

    # Top-level summary across all slices
    with open(os.path.join(output_folder, "run_summary.json"), 'w') as fout:
        json.dump({'slices': summary, 'n_slices': len(summary)}, fout, indent=2)

    duration = time.perf_counter() - start_time
    print(f"\nStep 9 done in {duration:.2f}s   exports ready for Seurat::Load10X_Spatial()")
