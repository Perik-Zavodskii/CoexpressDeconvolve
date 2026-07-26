"""
CoexpressDeconvolve main module.

Multi-slice Visium deconvolution to in-silico single cells.
Pipeline: load -> density -> ODG -> manifold -> K-sweep -> LDA -> sampling -> placement -> export.

Multi-slice handling:
  visium_path can be a string (single slice) or a dict {name: path} / list (multi).
  Per-slice params (min_umi, anchor_mean_factor, low_slice_quality) accept either a
  scalar (broadcast to all slices) or a dict keyed by slice name.

LDA is fit per-slice and topics are aligned across slices by Hungarian matching on
cosine similarity of betas; the consensus beta is the mean of aligned betas.
Per-slice theta is then refit against the frozen consensus beta via a variational E-step.
The gene-coexpression manifold is joint over the intersected
gene set; per-slice beta projection extends consensus topics to each slice's full
cleaned gene list.
"""

import os
import re
import json
import shutil
import time
import warnings
from collections import defaultdict
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
    from sklearn.decomposition import FastICA, LatentDirichletAllocation, PCA
    from sklearn.preprocessing import StandardScaler, normalize
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    from scipy.special import digamma, polygamma
    from scipy.stats import mannwhitneyu
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
    # filled by Step 2
    n_cells: Optional[np.ndarray] = None
    engine_params: Optional[dict] = None
    # filled by Step 3 per-slice (post noise filter, slice-specific)
    counts_clean: Optional[sp.csr_matrix] = None
    genes_clean: Optional[List[str]] = None
    # filled by Step 6 per-slice
    theta: Optional[np.ndarray] = None
    beta_final: Optional[np.ndarray] = None


@dataclass
class OdgPack:
    """Output of Step 3. Joint overdispersed-gene selection on the intersected gene set."""
    intersected_genes: List[str]
    odg_names: List[str]
    odg_per_slice: Dict[str, sp.csr_matrix]
    odg_concat: sp.csr_matrix
    species: str


@dataclass
class Manifold:
    """Output of Step 4. Joint manifold on intersected genes.

    ``embedding`` is the 2D UMAP layout (visualization only). ``ica_coords`` is
    the higher-dimensional ICA feature matrix (genes x n_components) that the
    manifold was built from; Step 6 uses it for the projection k-NN because 2D
    UMAP distances are not quantitatively meaningful.
    """
    embedding: np.ndarray
    intersected_genes: List[str]
    odg_indices_in_intersected: List[int]
    species: str
    ica_coords: Optional[np.ndarray] = None


@dataclass
class Model:
    """Output of Step 6. Consensus beta + per-slice theta."""
    n_topics: int
    odg_names: List[str]
    beta_consensus: np.ndarray
    qc_df: pd.DataFrame
    per_slice_betas: Dict[str, np.ndarray]
    per_slice_stability: Optional[Dict[str, float]] = None


@dataclass
class KSweepResult:
    """Output of Step 5. Compact repr so notebook auto-display stays one line.

    All metrics remain accessible as attributes for programmatic inspection,
    custom plotting, or persistence — only the default display is suppressed.
    """
    perplexity: Dict[str, Dict[int, float]]
    rare_topics: Dict[str, Dict[int, int]]
    alpha_mean: Dict[str, Dict[int, float]]
    alpha_per_topic: Dict[str, Dict[int, np.ndarray]]
    perc_rare_thresh: float
    recommended_k: Dict[str, Dict[str, object]]

    def __repr__(self) -> str:
        if not self.perplexity:
            return "KSweepResult(empty)"
        targets = list(self.perplexity.keys())
        any_label = targets[0]
        ks = sorted(self.perplexity[any_label].keys())
        if ks:
            k_span = f"K={ks[0]}..{ks[-1]}"
        else:
            k_span = "K=[]"
        return f"KSweepResult(targets={targets}, {k_span})"


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

    Tries a direct rng.multinomial(n, p) call. Falls back to a clip + normalize + shave
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


def _inv_digamma(y, n_iter: int = 5):
    """Newton's method for the inverse of the digamma function.

    Initialization rule from Minka, "Estimating a Dirichlet distribution" (2000),
    Appendix C. Five Newton steps suffice for double precision.
    """
    y = np.asarray(y, dtype=float)
    x = np.where(y >= -2.22, np.exp(y) + 0.5, -1.0 / (y - digamma(1.0)))
    for _ in range(n_iter):
        x = x - (digamma(x) - y) / polygamma(1, x)
    return x


def _dirichlet_alpha_mle(
    theta: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-7,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate an asymmetric Dirichlet alpha from observed proportions.

    Implements Minka's fixed-point iteration (2000, eq. 9). Treats each row of
    theta as an observed sample from Dir(alpha). The iteration loops
        alpha_k <- digamma^{-1}( digamma(sum_k alpha_k) + E_d[log theta_{d,k}] )
    until convergence.

    Used in Step 5 as a post-hoc proxy for the alpha parameter that
    STdeconvolve's R LDA estimates directly (sklearn fixes doc_topic_prior, so
    we recover an alpha estimate from the fitted topic distributions instead).
    A fitted mean alpha < 1 indicates the model retained a sparse Dirichlet
    prior (each spot dominated by a few topics); values >= 1 mean topics smear
    across spots, which is the classical "K is too large" signature.

    Parameters
    ----------
    theta : (N, K) array of proportions, each row should sum to ~1.

    Returns
    -------
    alpha : (K,) array of per-topic concentration parameters.
    """
    theta = np.asarray(theta, dtype=float)
    theta = np.clip(theta, eps, 1.0)
    log_p_mean = np.mean(np.log(theta), axis=0)

    # Method-of-moments initialization (Minka eq. 23).
    p_mean = np.mean(theta, axis=0)
    p_var = np.maximum(np.mean(theta ** 2, axis=0) - p_mean ** 2, eps)
    s_per_k = (p_mean * (1.0 - p_mean) / p_var) - 1.0
    s = max(float(np.median(s_per_k)), 0.1)
    alpha = np.maximum(p_mean * s, 1e-3)

    for _ in range(max_iter):
        alpha_old = alpha.copy()
        alpha = _inv_digamma(digamma(alpha.sum()) + log_p_mean)
        alpha = np.maximum(alpha, 1e-6)
        if np.max(np.abs(alpha - alpha_old)) < tol:
            break
    return alpha


def _overdispersed_genes(
    counts: sp.csr_matrix,
    n_top: int,
    poly_deg: int = 3,
    fit_mask: Optional[np.ndarray] = None,
):
    """Select overdispersed genes via mean-variance trend residuals.

    Library-size-normalize counts to 10k, log1p-transform, compute per-gene
    mean and variance, fit a smoothed polynomial trend log10(var) ~
    poly(log10(mean)) across genes, and rank genes by the residual above the
    trend. The top n_top residuals are the overdispersed gene set.

    This mirrors STdeconvolve's restrictCorpus / getOverdispersedGenes: genes
    whose variance exceeds the global mean-variance relationship are the ones
    that carry biological signal beyond Poisson sampling noise. Replaces the
    earlier binned-dispersion-z-score selection inspired by Seurat: more robust
    to bin-edge effects and closer to the corpus-building procedure LDA-based
    spatial deconvolution expects.

    Parameters
    ----------
    fit_mask : Optional[np.ndarray]
        Boolean mask of length n_genes. If provided, the polynomial trend is
        fit using only the genes where fit_mask is True (e.g. to exclude
        pre-filtered rare or ubiquitous genes), AND ranking is restricted to
        those same genes. Residuals are still computed for all genes so the
        diagnostic plot can show filtered-out genes as context.

    Returns
    -------
    top_indices : np.ndarray (n_top_eff,)
        Gene indices sorted by residual, highest first. When fit_mask is given,
        all returned indices come from the masked-in set.
    log_mean, log_var, fitted_log_var, residuals : np.ndarray (n_genes,)
        Diagnostic arrays for plotting.
    """
    row_sums = np.array(counts.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    norm = counts.copy().astype(float)
    norm.data /= np.repeat(row_sums, np.diff(norm.indptr))
    norm.data *= 10000.0
    norm.data = np.log1p(norm.data)

    mean_expr = np.array(norm.mean(axis=0)).flatten()
    sq = norm.copy()
    sq.data **= 2
    mean_sq = np.array(sq.mean(axis=0)).flatten()
    var_expr = np.maximum(mean_sq - mean_expr ** 2, 0.0)

    eps = 1e-10
    log_mean = np.log10(mean_expr + eps)
    log_var = np.log10(var_expr + eps)
    valid = (mean_expr > 0) & (var_expr > 0)

    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        fit_valid = valid & fit_mask
        rank_pool = fit_valid
    else:
        fit_valid = valid
        rank_pool = valid

    if int(fit_valid.sum()) < poly_deg + 1:
        # Pathological corpus: fall back to top by raw variance within the pool.
        n_top_eff = min(n_top, int(rank_pool.sum()) if rank_pool.sum() > 0 else n_top)
        scores = np.where(rank_pool, var_expr, -np.inf)
        top = np.argsort(scores)[-n_top_eff:][::-1]
        fitted = np.full_like(log_var, np.nan)
        return top, log_mean, log_var, fitted, np.zeros_like(log_var)

    coeffs = np.polyfit(log_mean[fit_valid], log_var[fit_valid], deg=poly_deg)
    fitted_log_var = np.polyval(coeffs, log_mean)
    residuals = log_var - fitted_log_var
    residuals_for_ranking = np.where(rank_pool, residuals, -np.inf)

    n_top_eff = min(n_top, int(rank_pool.sum()))
    top = np.argsort(residuals_for_ranking)[-n_top_eff:][::-1]
    return top, log_mean, log_var, fitted_log_var, residuals


def _batched_multinomial(rng, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Batched multinomial sampler via sequential binomial decomposition.

    Each row is independently drawn from Multinomial(n[i], p[i, :]). Works on
    both ``np.random.RandomState`` and the newer Generator
    types because both expose a vectorized ``binomial`` with broadcasting.

    The recurrence is the standard composition: condition on the events
    already assigned and the remaining probability mass, draw the next column
    as a binomial of the remaining trials, then subtract.

    Parameters
    ----------
    rng : np.random.RandomState or np.random.Generator
        Source of randomness. Must support ``binomial(n, p)`` with vector args.
    n : (M,) array of nonnegative integer trial counts.
    p : (M, K) array of probability rows (each row should sum to ~1).

    Returns
    -------
    (M, K) integer array. Rows sum exactly to ``n`` and entries are ``>= 0``.
    """
    n = np.asarray(n, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    M, K = p.shape
    if M == 0 or K == 0:
        return np.zeros((M, K), dtype=np.int64)

    result = np.zeros((M, K), dtype=np.int64)
    remaining = n.copy()
    # cum_p[:, k] = sum_{j >= k} p[:, j], i.e. mass still available at column k.
    cum_p = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
    for k in range(K - 1):
        denom = cum_p[:, k]
        # When denom is zero (all remaining mass collapses to zero) skip the
        # binomial and keep result[:, k] = 0.
        with np.errstate(invalid='ignore', divide='ignore'):
            prob_k = np.where(denom > 0, p[:, k] / denom, 0.0)
        prob_k = np.clip(prob_k, 0.0, 1.0)
        draw = rng.binomial(remaining, prob_k)
        draw = np.minimum(draw, remaining)
        result[:, k] = draw
        remaining = remaining - draw
    result[:, K - 1] = remaining
    return result


def _topic_stability(aligned_betas: List[np.ndarray]) -> float:
    """Mean pairwise cosine similarity of aligned topic rows across replicates.

    Inputs are a list of (K, G) topic-gene matrices that have already been row-
    permuted into a common topic ordering (e.g. by Hungarian matching). For
    every pair of replicates, compute the per-topic cosine similarity and
    average across topics; then average across pairs.

    Returns 1.0 for a single replicate (trivially stable) or perfectly
    identical replicates; values approach 0 as replicates diverge.
    """
    n = len(aligned_betas)
    if n < 2:
        return 1.0
    sims = []
    for i in range(n):
        norm_i = np.linalg.norm(aligned_betas[i], axis=1)
        for j in range(i + 1, n):
            norm_j = np.linalg.norm(aligned_betas[j], axis=1)
            denom = np.maximum(norm_i * norm_j, 1e-12)
            cos = np.sum(aligned_betas[i] * aligned_betas[j], axis=1) / denom
            sims.append(float(np.mean(cos)))
    return float(np.mean(sims))


def _topic_log2fc(beta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-topic log2 fold change of beta vs the mean beta of all other topics.

    log2fc[k, g] = log2( beta[k, g] / mean_{k' != k}( beta[k', g] ) )

    Ranking genes by this quantity highlights what is *specific* to a topic
    rather than what is merely highly expressed everywhere, which is the
    interpretive lens STdeconvolve uses in its topGenes / getBetaTheta output.
    Falls through to zeros when there is only a single topic.

    Parameters
    ----------
    beta : (K, G) topic-gene distribution. Rows do not need to sum to 1 (we
        compare ratios, so any per-row scaling cancels).

    Returns
    -------
    log2fc : (K, G) array.
    """
    K = beta.shape[0]
    if K <= 1:
        return np.zeros_like(beta)
    total = beta.sum(axis=0, keepdims=True)
    mean_other = (total - beta) / (K - 1)
    return np.log2((beta + eps) / (mean_other + eps))


# Denoise (Step 7 optional GEX clean-up) helpers.
#
# A de novo, Seurat-mimicking clustering + marker pass on the generated cells,
# used to strip "bleed" of cluster-specific marker genes out of cells that don't
# belong to that gene's cluster(s) and reassign those UMIs back to the cells that
# do. No scanpy: every step below is implemented directly. The reassignment is a
# full transfer restricted to WITHIN EACH SPOT: for a specific marker gene, all of
# its UMIs in that spot's non-home cells are pooled and handed to that spot's
# home-cluster cells. If a spot has no home-cluster cell, those UMIs are discarded
# ("to trash") rather than moved to another spot — the count is never fabricated
# in a location where it was not measured. Per-gene totals are therefore conserved
# or reduced at the spot level, never inflated. Non-marker genes are untouched.

def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment of a p-value vector."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def _sc_normalize(counts: sp.csr_matrix, scale_factor: float = 1e4) -> sp.csr_matrix:
    """Seurat LogNormalize: log1p(count / library_size * scale_factor)."""
    counts = counts.tocsr().astype(np.float64)
    lib = np.asarray(counts.sum(axis=1)).flatten()
    lib[lib == 0] = 1.0
    norm = counts.copy()
    norm.data /= np.repeat(lib, np.diff(norm.indptr))
    norm.data *= scale_factor
    norm.data = np.log1p(norm.data)
    return norm


def _sc_hvg(counts: sp.csr_matrix, n_hvg: int) -> np.ndarray:
    """Seurat vst variable-feature selection.

    Fit a quadratic trend of log10(variance) ~ log10(mean) on raw counts,
    standardize each gene against its expected SD (clipped at sqrt(N)), and rank
    genes by the variance of the standardized values.
    """
    counts = counts.tocsc().astype(np.float64)
    n_cells, n_genes = counts.shape
    mean = np.asarray(counts.mean(axis=0)).flatten()
    sq = counts.copy(); sq.data **= 2
    mean_sq = np.asarray(sq.mean(axis=0)).flatten()
    var = np.maximum(mean_sq - mean ** 2, 0.0) * (n_cells / max(n_cells - 1, 1))
    std_var = np.zeros(n_genes)
    not_const = var > 0
    if int(not_const.sum()) > 3:
        coeffs = np.polyfit(np.log10(mean[not_const]), np.log10(var[not_const]), 2)
        pos = mean > 0
        expected_sd = np.zeros(n_genes)
        expected_sd[pos] = np.sqrt(10 ** np.polyval(coeffs, np.log10(mean[pos])))
        valid = pos & (expected_sd > 0)
        clip_max = np.sqrt(n_cells)

        # Fully vectorized standardized-variance: map each stored (nonzero) value
        # to its gene column, standardize + clip, and sum-of-squares per gene via
        # bincount; add the zero-entry contribution analytically.
        n_nz = np.diff(counts.indptr)                       # nonzeros per gene (CSC)
        gene_of_nz = np.repeat(np.arange(n_genes), n_nz)
        sd_nz = expected_sd[gene_of_nz]
        with np.errstate(divide='ignore', invalid='ignore'):
            z_nz = (counts.data - mean[gene_of_nz]) / sd_nz
        z_nz = np.clip(z_nz, None, clip_max)
        z_nz = np.where(np.isfinite(z_nz), z_nz, 0.0)
        sumsq_nz = np.bincount(gene_of_nz, weights=z_nz ** 2, minlength=n_genes)
        with np.errstate(divide='ignore', invalid='ignore'):
            z_zero = np.clip((0.0 - mean) / expected_sd, None, clip_max)
        z_zero = np.where(valid, z_zero, 0.0)
        ss = sumsq_nz + (n_cells - n_nz) * (z_zero ** 2)
        std_var = np.where(valid, ss / max(n_cells - 1, 1), 0.0)
    order = np.argsort(std_var)[::-1]
    return order[:min(n_hvg, n_genes)]


def _sc_scale(lognorm: sp.csr_matrix, hvg_idx: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """Seurat ScaleData: z-score each variable gene, clip at +/- clip."""
    sub = lognorm[:, hvg_idx].toarray()
    mu = sub.mean(axis=0)
    sd = sub.std(axis=0); sd[sd == 0] = 1.0
    return np.clip((sub - mu) / sd, -clip, clip)


def _sc_pca(scaled: np.ndarray, n_pcs: int, seed: int) -> np.ndarray:
    n_pcs = max(1, min(n_pcs, min(scaled.shape) - 1))
    return PCA(n_components=n_pcs, random_state=seed).fit_transform(scaled)


def _sc_snn(coords: np.ndarray, k: int, prune: float = 1.0 / 15.0) -> sp.csr_matrix:
    """Shared-nearest-neighbour graph with Jaccard edge weights (Seurat-style)."""
    n = coords.shape[0]
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k).fit(coords)
    knn = nn.kneighbors_graph(coords, mode='connectivity').astype(bool).astype(np.float64)
    inter = (knn @ knn.T).tocoo()
    jac = inter.data / (2 * k - inter.data)
    snn = sp.coo_matrix((jac, (inter.row, inter.col)), shape=(n, n)).tocsr()
    snn.setdiag(0.0); snn.eliminate_zeros()
    snn.data[snn.data < prune] = 0.0; snn.eliminate_zeros()
    return snn.maximum(snn.T)


def _louvain(adj: sp.csr_matrix, resolution: float = 1.0, seed: int = 42,
             max_levels: int = 20) -> np.ndarray:
    """Modularity Louvain with an RB resolution parameter. Self-contained.

    Higher resolution penalises large communities, yielding more clusters
    (matching Seurat's FindClusters resolution semantics). Returns a compact
    integer label per original node.
    """
    rng = np.random.default_rng(seed)
    adj = adj.tocsr().astype(np.float64)
    n0 = adj.shape[0]
    orig_to_cur = np.arange(n0)  # original node -> node index in current graph

    cur = adj
    for _level in range(max_levels):
        n = cur.shape[0]
        deg = np.asarray(cur.sum(axis=1)).flatten()
        m = deg.sum() / 2.0
        if m == 0:
            break
        comm = np.arange(n)
        comm_tot = deg.copy()
        indptr, indices, data = cur.indptr, cur.indices, cur.data

        improved_any = False
        improved = True
        while improved:
            improved = False
            for i in rng.permutation(n):
                ci = comm[i]
                ki = deg[i]
                s, e = indptr[i], indptr[i + 1]
                comm_w = defaultdict(float)
                for j, wij in zip(indices[s:e], data[s:e]):
                    if j != i:
                        comm_w[comm[j]] += wij
                comm_tot[ci] -= ki  # pull i out of its community
                best_c = ci
                # RB modularity gain: k_{i,in} - gamma * Sigma_tot * k_i / (2m).
                best_gain = comm_w.get(ci, 0.0) - resolution * ki * comm_tot[ci] / (2.0 * m)
                for c, wic in comm_w.items():
                    gain = wic - resolution * ki * comm_tot[c] / (2.0 * m)
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_c = c
                comm[i] = best_c
                comm_tot[best_c] += ki
                if best_c != ci:
                    improved = True
                    improved_any = True

        uniq, comm_new = np.unique(comm, return_inverse=True)  # compact labels
        orig_to_cur = comm_new[orig_to_cur]
        if not improved_any or uniq.shape[0] == n:
            break
        nc = uniq.shape[0]
        row = comm_new[np.repeat(np.arange(n), np.diff(indptr))]
        col = comm_new[indices]
        cur = sp.coo_matrix((data, (row, col)), shape=(nc, nc)).tocsr()

    _, labels = np.unique(orig_to_cur, return_inverse=True)
    return labels


def _sc_find_markers(lognorm: sp.csr_matrix, labels: np.ndarray, min_pct: float,
                     logfc: float, padj_thresh: float, label: str = ""):
    """FindAllMarkers analogue (Wilcoxon rank-sum, one-vs-rest).

    For each cluster, pre-filter genes by pct.1 >= min_pct and avg log2FC >= logfc
    (Seurat's min.pct / logfc.threshold gates), run a tie-corrected Wilcoxon
    rank-sum on the survivors, BH-adjust, and keep genes with padj < padj_thresh.

    Returns
    -------
    gene_to_home : dict{gene_idx -> set(cluster labels)}
        Every cluster for which the gene passed as an up-regulated marker.
    cluster_marker_profiles : dict{cluster -> (mean expm1 vector over all genes)}
        Used for the donor/home similarity QC flag.
    """
    lognorm = lognorm.tocsc()
    n_cells, n_genes = lognorm.shape
    clusters = np.unique(labels)
    binary = (lognorm > 0)
    expm = lognorm.copy(); expm.data = np.expm1(expm.data)
    gene_to_home = defaultdict(set)
    cluster_profile = {}
    for c in tqdm(clusters, desc=f"   [{label}] markers", unit="cluster", leave=False):
        in_c = labels == c
        n_in = int(in_c.sum()); n_out = n_cells - n_in
        cluster_profile[int(c)] = np.asarray(expm[in_c].mean(axis=0)).flatten()
        if n_in < 3 or n_out < 3:
            continue
        pct1 = np.asarray(binary[in_c].sum(axis=0)).flatten() / n_in
        m1 = cluster_profile[int(c)]
        m2 = np.asarray(expm[~in_c].mean(axis=0)).flatten()
        log2fc = np.log2(m1 + 1.0) - np.log2(m2 + 1.0)
        cand = np.where((pct1 >= min_pct) & (log2fc >= logfc))[0]
        if cand.size == 0:
            continue
        # Vectorized tie-corrected Wilcoxon rank-sum over all candidate genes at
        # once (normal approximation), replacing the per-gene Python loop.
        Dc = lognorm[:, cand].toarray()
        with np.errstate(all='ignore'):
            res = mannwhitneyu(Dc[in_c], Dc[~in_c], axis=0,
                               alternative='two-sided', method='asymptotic')
        pvals = np.asarray(res.pvalue, dtype=float)
        pvals = np.where(np.isfinite(pvals), pvals, 1.0)
        padj = _bh_adjust(pvals)
        for g in cand[padj < padj_thresh]:
            gene_to_home[int(g)].add(int(c))
    return gene_to_home, cluster_profile


def _sc_reassign(counts: sp.csr_matrix, labels: np.ndarray, spot_ids: np.ndarray,
                 gene_to_home: dict, seed: int, label: str = ""):
    """Within-spot full-transfer UMI reassignment of bled marker genes.

    Home clusters for each gene are defined globally (from the slice-wide marker
    call), but every UMI move is confined to the spot the cell came from:

      For each specific marker gene and each spot, pool that gene's UMIs sitting
      in the spot's non-home cells. If the spot contains home-cluster cell(s),
      redistribute the pool across them by a multinomial weighted by each cell's
      existing expression of the gene (uniform if all zero) — expression mass
      already encodes the per-cluster share when several home clusters coexist in
      the spot. If the spot has no home-cluster cell, the pooled UMIs are
      DISCARDED, not relocated to another spot.

    This never fabricates counts in a spot that did not measure them; per-gene
    totals are conserved or reduced per spot, never inflated. Non-marker genes
    are left untouched.

    Returns (cleaned_csr, umis_moved, umis_discarded, genes_cleaned).
    """
    rng = np.random.RandomState(seed)
    counts = counts.tocsc().astype(np.int64)
    labels = np.asarray(labels)
    spot_ids = np.asarray(spot_ids)
    n_all = np.unique(labels).shape[0]
    n_cells, n_genes = counts.shape

    # spot -> array of cell row indices (built once).
    spot_to_cells: Dict[int, list] = defaultdict(list)
    for i, s in enumerate(spot_ids):
        spot_to_cells[int(s)].append(i)
    spot_to_cells = {s: np.asarray(v) for s, v in spot_to_cells.items()}

    moved_total = 0
    discarded_total = 0
    genes_cleaned = 0
    discarded_per_spot: Dict[int, int] = defaultdict(int)

    # Modified columns are stashed as sparse triplets and the whole matrix is
    # rebuilt ONCE at the end. In-place CSC column assignment is O(nnz) per gene
    # (it rebuilds the index arrays on every call) and was ~96% of the runtime;
    # stashing avoids it entirely. RNG usage and iteration order are unchanged, so
    # the output is bit-identical to the in-place version, just far faster.
    changed_mask = np.zeros(n_genes, dtype=bool)
    new_rows: List[np.ndarray] = []
    new_cols: List[np.ndarray] = []
    new_data: List[np.ndarray] = []

    for g, home in tqdm(gene_to_home.items(), total=len(gene_to_home),
                        desc=f"   [{label}] de-bleed", unit="gene", leave=False):
        home = set(home)
        if len(home) == 0 or len(home) == n_all:
            continue  # not specific / expressed everywhere -> nothing bleeds
        col = counts.getcol(int(g)).toarray().flatten().astype(np.int64)
        home_mask = np.isin(labels, list(home))
        # Only spots that actually hold bled UMIs of this gene need visiting.
        donor_cells = np.where((~home_mask) & (col > 0))[0]
        if donor_cells.size == 0:
            continue
        touched = False
        for s in np.unique(spot_ids[donor_cells]):
            cells = spot_to_cells[int(s)]
            spot_home = home_mask[cells]
            donor_local = cells[~spot_home]
            pool = int(col[donor_local].sum())
            if pool == 0:
                continue
            col[donor_local] = 0  # strip bled UMIs from this spot's wrong cells
            recip = cells[spot_home]
            if recip.size == 0:
                discarded_total += pool  # no target population in this spot
                discarded_per_spot[int(s)] += pool
                touched = True
                continue
            w = col[recip].astype(np.float64)
            if w.sum() == 0:
                w = np.ones(recip.size)
            col[recip] += rng.multinomial(pool, w / w.sum())
            moved_total += pool
            touched = True
        if touched:
            nz = np.flatnonzero(col)
            new_rows.append(nz)
            new_cols.append(np.full(nz.shape[0], int(g), dtype=np.int64))
            new_data.append(col[nz])
            changed_mask[int(g)] = True
            genes_cleaned += 1

    # Rebuild once: keep every original entry whose column was NOT modified, then
    # append the stashed entries for the modified columns. O(nnz), done a single
    # time instead of once per gene.
    col_of_nz = np.repeat(np.arange(n_genes), np.diff(counts.indptr))
    keep = ~changed_mask[col_of_nz]
    rows = np.concatenate([counts.indices[keep]] + new_rows)
    cols = np.concatenate([col_of_nz[keep]] + new_cols)
    data = np.concatenate([counts.data[keep]] + new_data)
    cleaned = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes))
    return cleaned, moved_total, discarded_total, dict(discarded_per_spot), genes_cleaned


def _plot_denoise_drop(spot_before: np.ndarray, spot_dropped: np.ndarray, label: str):
    """Two-panel denoise diagnostic.

    Left: distribution over spots of the percentage of a spot's UMIs discarded
    by the clean-up (only spots that held UMIs are shown). Right: overall share
    of total UMIs retained vs dropped across the whole slice.
    """
    spot_before = np.asarray(spot_before, dtype=float)
    spot_dropped = np.asarray(spot_dropped, dtype=float)
    has_umi = spot_before > 0
    pct_per_spot = np.zeros_like(spot_before)
    pct_per_spot[has_umi] = 100.0 * spot_dropped[has_umi] / spot_before[has_umi]

    total_before = spot_before.sum()
    total_dropped = spot_dropped.sum()
    pct_dropped = 100.0 * total_dropped / max(total_before, 1.0)
    pct_retained = 100.0 - pct_dropped
    n_spots_hit = int((spot_dropped > 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

    ax = axes[0]
    vals = pct_per_spot[has_umi]
    ax.hist(vals, bins=40, color='#e74c3c', edgecolor='black', alpha=0.75)
    if vals.size:
        ax.axvline(float(np.mean(vals)), color='black', linestyle='--',
                   linewidth=1.5, label=f'mean {np.mean(vals):.2f}%')
        ax.axvline(float(np.median(vals)), color='navy', linestyle=':',
                   linewidth=1.5, label=f'median {np.median(vals):.2f}%')
        ax.legend()
    ax.set_xlabel('% of spot UMIs dropped')
    ax.set_ylabel('# spots')
    ax.set_title(f'[{label}] per-spot UMI drop distribution\n'
                 f'{n_spots_hit}/{int(has_umi.sum())} spots affected')
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    bars = ax2.bar(['Retained', 'Dropped'], [pct_retained, pct_dropped],
                   color=['#27ae60', '#e74c3c'], edgecolor='black', width=0.6)
    for b, v in zip(bars, [pct_retained, pct_dropped]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.0, f'{v:.2f}%',
                 ha='center', fontweight='bold')
    ax2.set_ylabel('% of total UMIs')
    ax2.set_ylim(0, 105)
    ax2.set_title(f'[{label}] overall UMI retained vs dropped\n'
                  f'{int(total_before - total_dropped):,} kept / {int(total_dropped):,} dropped')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def _denoise_matrix(counts: sp.csr_matrix, spot_ids: np.ndarray, min_pct: float,
                    logfc: float, resolution: float, n_hvg: int, n_pcs: int,
                    k_neighbors: int, padj_thresh: float, seed: int,
                    label: str = "", make_plot: bool = True) -> tuple:
    """Orchestrate the Step 7 GEX clean-up on one slice's cell x gene matrix.

    ``spot_ids`` gives the source spot index of each cell (row); UMI moves are
    confined to within a spot. Returns (cleaned_csr, qc_dict).
    """
    n_cells = counts.shape[0]
    if n_cells < 10:
        return counts.tocsr(), {'skipped': 'too few cells', 'n_cells': n_cells}

    print(f"   [{label}] denoise: log-normalizing...")
    lognorm = _sc_normalize(counts)
    print(f"   [{label}] denoise: finding variable genes (vst)...")
    hvg = _sc_hvg(counts, n_hvg)
    print(f"   [{label}] denoise: scaling...")
    scaled = _sc_scale(lognorm, hvg)
    print(f"   [{label}] denoise: computing PCA ({n_pcs} components)...")
    pca = _sc_pca(scaled, n_pcs, seed)
    print(f"   [{label}] denoise: building SNN graph (k={k_neighbors})...")
    snn = _sc_snn(pca, k_neighbors)
    print(f"   [{label}] denoise: clustering (Louvain, res={resolution})...")
    labels = _louvain(snn, resolution=resolution, seed=seed)
    n_clusters = int(np.unique(labels).size)
    print(f"   [{label}] denoise: {n_clusters} clusters found")

    if n_clusters < 2:
        return counts.tocsr(), {'n_clusters': n_clusters, 'umis_moved': 0,
                                'genes_cleaned': 0,
                                'note': 'single cluster; nothing to de-bleed'}

    print(f"   [{label}] denoise: finding markers (Wilcoxon, one-vs-rest)...")
    gene_to_home, cluster_profile = _sc_find_markers(
        lognorm, labels, min_pct=min_pct, logfc=logfc, padj_thresh=padj_thresh, label=label
    )
    n_specific = sum(1 for h in gene_to_home.values() if 0 < len(h) < n_clusters)
    print(f"   [{label}] denoise: reassigning bled UMIs within spots...")
    cleaned, moved, discarded, discarded_per_spot, genes_cleaned = _sc_reassign(
        counts, labels, spot_ids, gene_to_home, seed, label=label
    )

    # Per-spot UMI accounting for the drop diagnostic.
    spot_ids = np.asarray(spot_ids, dtype=np.int64)
    row_tot = np.asarray(counts.sum(axis=1)).flatten()
    spot_before = np.bincount(spot_ids, weights=row_tot).astype(np.int64)
    spot_dropped = np.zeros_like(spot_before)
    for s, v in discarded_per_spot.items():
        spot_dropped[int(s)] += int(v)
    total_before = int(spot_before.sum())
    pct_dropped = 100.0 * discarded / max(total_before, 1)
    pct_retained = 100.0 - pct_dropped

    # QC: flag cluster pairs whose expression profiles correlate highly (>0.9) —
    # the clean-up may be collapsing genuine subclusters there. Correlation is
    # computed on the variable genes (HVGs) only: on the full gene set the shared
    # background makes almost every pair look ~1.0, which is uninformative.
    prof_ids = sorted(cluster_profile.keys())
    similar_pairs = []
    if len(prof_ids) > 1:
        P = np.vstack([cluster_profile[c] for c in prof_ids])[:, hvg]
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(P)
        for a in range(len(prof_ids)):
            for b in range(a + 1, len(prof_ids)):
                if np.isfinite(corr[a, b]) and corr[a, b] > 0.9:
                    similar_pairs.append((prof_ids[a], prof_ids[b], round(float(corr[a, b]), 3)))
    if similar_pairs:
        print(f"   [{label}] denoise QC: {len(similar_pairs)} highly-correlated cluster "
              f"pair(s) (r>0.9) — check for subcluster collapse: {similar_pairs}")

    _, sizes = np.unique(labels, return_counts=True)
    qc = {
        'n_clusters': n_clusters,
        'cluster_sizes': sizes.tolist(),
        'n_specific_genes': int(n_specific),
        'genes_cleaned': int(genes_cleaned),
        'umis_moved': int(moved),
        'umis_discarded': int(discarded),
        'umis_total_before': total_before,
        'pct_umi_retained': round(pct_retained, 4),
        'pct_umi_dropped': round(pct_dropped, 4),
        'umis_per_spot_before': spot_before.tolist(),
        'umis_dropped_per_spot': spot_dropped.tolist(),
        'labels': labels.tolist(),
        'similar_cluster_pairs': similar_pairs,
    }
    print(f"   [{label}] denoise: {moved} UMIs reassigned within-spot, "
          f"{discarded} discarded (no in-spot target) across {genes_cleaned} "
          f"marker genes ({n_specific} specific of {len(gene_to_home)} markers)")
    print(f"   [{label}] denoise: UMI retained {pct_retained:.2f}% / dropped {pct_dropped:.2f}% "
          f"({(spot_dropped > 0).sum()} spots affected)")
    if make_plot:
        _plot_denoise_drop(spot_before, spot_dropped, label)
    return cleaned, qc


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
    anchor_blend_alpha=0.6, 
    low_slice_quality=False,
    colormap='hsv',
) -> Dict[str, SliceData]:
    """Estimate per-spot cell counts from Hybrid HK-UMI calibration.

    Per-slice min_umi, anchor_mean_factor, and alpha accepted as scalar or dict.
    alpha=1.0 is pure UMI, alpha=0.0 is pure Housekeeping.
    """
    start_time = time.perf_counter()
    profile = _load_config(config_path, species)
    hk_reference = profile['hk_profiles']
    engine_params = profile.get('engine_parameters', {})

    if not hk_reference:
        raise ValueError(f"Species '{species}' has no hk_profiles in config.")

    names = list(slices.keys())
    min_umi_d = _broadcast(min_umi, names)
    anchor_d = _broadcast(anchor_mean_factor, names)
    alpha_d = _broadcast(anchor_blend_alpha, names)
    low_q_d = _broadcast(low_slice_quality, names, default=False)

    for name in names:
        sd = slices[name]
        slice_min_umi = min_umi_d[name]
        slice_anchor = anchor_d[name]
        slice_alpha = alpha_d[name]
        slice_low_q = low_q_d[name]

        print(f"\nStep 2 [{name}]: Hybrid Calibration (alpha={slice_alpha}, factor={slice_anchor})")

        common_hk = [g for g in hk_reference.keys() if g in sd.gene_names]
        hk_indices = [sd.gene_names.index(g) for g in common_hk]
        ref_values_log = np.array([hk_reference[g] for g in common_hk])

        # 1. Housekeeping Signal (The Biological Anchor)
        safe_total = sd.total_umi.copy()
        safe_total[safe_total == 0] = 1
        hk_counts_raw = sd.counts[:, hk_indices].toarray()
        normalized_hk_log = np.log1p((hk_counts_raw / safe_total[:, np.newaxis]) * 10000)

        spot_hk_log_means = np.mean(normalized_hk_log, axis=1)
        standard_anchor_log_mean = np.mean(ref_values_log)
        adjusted_standard_log_mean = standard_anchor_log_mean + np.log(slice_anchor)

        spot_signal_linear = np.expm1(spot_hk_log_means)
        standard_signal_linear = np.expm1(adjusted_standard_log_mean)
        if standard_signal_linear < 0.001: standard_signal_linear = 0.001

        hk_cells = spot_signal_linear / standard_signal_linear

        # 2. Total UMI Signal (The Statistical Stabilizer)
        # We anchor the UMI-per-cell ratio to the valid HK-estimated spots
        valid_gate = sd.total_umi >= slice_min_umi
        if np.any(valid_gate) and np.sum(hk_cells[valid_gate]) > 0:
            global_umi_per_cell = np.sum(sd.total_umi[valid_gate]) / np.sum(hk_cells[valid_gate])
        else:
            global_umi_per_cell = np.mean(sd.total_umi) / 5.0 # Fallback
            
        umi_cells = sd.total_umi / global_umi_per_cell

        # 3. The Hybrid Blend (Weighted Geometric Mean)
        raw_n_cells = (umi_cells ** slice_alpha) * (hk_cells ** (1.0 - slice_alpha))
        
        # 4. Discrete Mapping and Filtering
        n_cells = np.round(raw_n_cells).astype(int)
        is_low_quality = sd.total_umi < slice_min_umi
        n_cells[is_low_quality] = 0
        
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


# STEP 3: ODG selection (joint, on intersected gene set)

def step3_feature_selection(
    slices: Dict[str, SliceData],
    config_path: str,
    species: str,
    n_odg: int = 2000,
    od_poly_deg: int = 3,
    min_pct_spots: float = 0.05,
    max_pct_spots: float = 0.95,
    pct_filter_mode: str = "all",
) -> OdgPack:
    """Per-slice noise filter, intersect across slices, restrict the candidate
    gene pool by spot-presence fraction, then overdispersed-gene select on the
    candidates.

    Pipeline inside this step:

      1. Per-slice noise regex filter from the species profile.
      2. Intersect gene sets across slices.
      3. Restrict the candidate pool by spot-presence fraction (mimics
         STdeconvolve's restrictCorpus): per slice, the fraction of spots in
         which a gene has count > 0 must lie within
         ``[min_pct_spots, max_pct_spots]``. With multiple slices, ``"any"``
         mode keeps a gene that passes in any one slice; ``"all"`` mode
         requires it to pass in every slice. The presence filter only restricts
         the pool considered for overdispersed selection — the per-slice
         ``genes_clean`` and ``OdgPack.intersected_genes`` keep the full
         noise-filtered intersection, so Steps 4 / 6 / 7 / 9 still see all
         genes.
      4. Overdispersed-gene selection via mean-variance trend residual: the
         polynomial trend ``log10(var) ~ poly(log10(mean))`` is fit on
         candidates only (so rare and ubiquitous filtered-out genes don't pull
         the curve), and the top ``n_odg`` candidates by residual become the
         overdispersed gene set.

    The OdgPack output carries the selected ``odg_names`` / ``odg_per_slice``
    / ``odg_concat``; Steps 4-9 consume these directly as the overdispersed
    gene set.

    Parameters
    ----------
    n_odg : int
        Number of top overdispersed genes to retain.
    od_poly_deg : int
        Polynomial degree for the smoothed mean-variance trend (default 3).
    min_pct_spots, max_pct_spots : float
        Fraction-of-spots window for the presence filter. A gene passes if its
        spot-occupancy lies in [min_pct_spots, max_pct_spots] in (any | all)
        slice(s). Defaults 0.05 / 0.95.
    pct_filter_mode : str
        ``"any"`` (default) or ``"all"`` — see above.
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

    # Concatenate (rows = spots) for joint trend / overdispersed calculation.
    inter_concat = sp.vstack([inter_counts_per_slice[n] for n in slices.keys()]).tocsr()

    # Presence filter (restrictCorpus). Per slice, compute the fraction of spots
    # in which each gene has count > 0; gene "passes" if the fraction is in
    # [min_pct_spots, max_pct_spots]. Combine across slices via the requested
    # mode. This filter only restricts the candidate pool for the overdispersed
    # selection — sd.genes_clean and OdgPack.intersected_genes remain untouched.
    if pct_filter_mode not in ("any", "all"):
        raise ValueError(
            f"pct_filter_mode must be 'any' or 'all', got {pct_filter_mode!r}"
        )

    n_intersect = len(intersected)
    pass_matrix = np.zeros((len(slices), n_intersect), dtype=bool)
    print(f"Step 3: presence filter [{min_pct_spots:.2f}, {max_pct_spots:.2f}] "
          f"({pct_filter_mode} across slices)")
    for i, (name, _sd) in enumerate(slices.items()):
        X_s = inter_counts_per_slice[name]
        presence = np.asarray((X_s > 0).sum(axis=0)).flatten() / float(X_s.shape[0])
        passes = (presence >= min_pct_spots) & (presence <= max_pct_spots)
        pass_matrix[i] = passes
        print(f"   [{name}] presence pass: {int(passes.sum())}/{n_intersect}")

    if pct_filter_mode == "any":
        candidate_mask = pass_matrix.any(axis=0)
    else:  # "all"
        candidate_mask = pass_matrix.all(axis=0)
    n_candidates = int(candidate_mask.sum())
    print(f"   combined: {n_candidates}/{n_intersect} candidate genes after filter")

    if n_candidates < max(10, od_poly_deg + 1):
        raise ValueError(
            f"Presence filter left only {n_candidates} candidate genes — "
            "loosen min_pct_spots / max_pct_spots or check the data."
        )

    # Overdispersed gene selection: trend fit and ranking restricted to candidates.
    print(f"Step 3: selecting top {n_odg} overdispersed genes from {n_candidates} candidates...")
    n_odg_eff = min(n_odg, n_candidates)
    odg_indices_local, log_mean, log_var, fitted_log_var, residuals = _overdispersed_genes(
        inter_concat, n_top=n_odg_eff, poly_deg=od_poly_deg, fit_mask=candidate_mask
    )
    odg_names = [intersected[i] for i in odg_indices_local]

    # Per-slice overdispersed-gene count matrices.
    odg_per_slice: Dict[str, sp.csr_matrix] = {
        name: inter_counts_per_slice[name][:, odg_indices_local] for name in slices.keys()
    }
    odg_concat = inter_concat[:, odg_indices_local]

    # Diagnostic plot: residuals above the smoothed mean-variance trend, split
    # into three populations: filtered-out (failed presence filter), candidates
    # not selected, and selected overdispersed. x-axis clipped at the 5th
    # percentile so the +eps padded zero-expression tail doesn't blow out view.
    is_top = np.zeros(n_intersect, dtype=bool)
    is_top[odg_indices_local] = True
    cand_not_top = candidate_mask & (~is_top)
    filt_out = ~candidate_mask

    plt.figure(figsize=(9, 6))
    plt.scatter(log_mean[filt_out], residuals[filt_out], s=2, color='lightgrey', alpha=0.35,
                label=f'Filtered out (presence): {int(filt_out.sum())}')
    plt.scatter(log_mean[cand_not_top], residuals[cand_not_top], s=2, color='steelblue', alpha=0.45,
                label=f'Candidates (not selected): {int(cand_not_top.sum())}')
    plt.scatter(log_mean[is_top], residuals[is_top], s=4, color='red',
                label=f'Overdispersed (selected): {n_odg_eff}')
    plt.axhline(0.0, color='black', linewidth=1.0, linestyle='--')
    
    cand_mean = log_mean[candidate_mask]
    if cand_mean.size > 0:
        x_low = float(np.min(cand_mean)) - 0.2
        x_high = float(np.max(log_mean)) + 0.2
        plt.xlim(x_low, x_high)

    cand_residuals = residuals[candidate_mask]
    if cand_residuals.size > 0:
        y_low = float(np.percentile(cand_residuals, 0.5))
        y_high = float(np.percentile(cand_residuals, 99.9))
        y_range = max(y_high - y_low, 1e-6)
        
        pad = 0.1 * y_range
        plt.ylim(min(-0.5, y_low - pad), max(1.0, y_high + pad))

    plt.xlabel('log10(mean expression)')
    plt.ylabel('Residual log10(variance) — observed minus trend')
    plt.title(f'Step 3: overdispersion score (top {n_odg_eff} of {n_candidates} candidates; '
              f'{n_intersect} intersected genes)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    duration = time.perf_counter() - start_time
    print(f"Step 3 done in {duration:.2f}s   joint overdispersed-gene matrix: {odg_concat.shape}")
    return OdgPack(
        intersected_genes=intersected,
        odg_names=odg_names,
        odg_per_slice=odg_per_slice,
        odg_concat=odg_concat,
        species=species,
    )


# STEP 4: joint manifold

def step4_gene_manifold(
    slices: Dict[str, SliceData],
    odg_pack: OdgPack,
    config_path: str,
    n_components: int = 30,
) -> Manifold:
    """ICA + UMAP on the gene matrix to produce a gene manifold.

    Single slice: uses the slice's counts_clean directly in its native gene order
    (matches legacy codeconv Step 4 behavior bit-for-bit modulo upstream tie-breaking).
    Multi-slice: uses the alphabetical-intersected stacked matrix so all slices share
    a common gene index space.
    """
    start_time = time.perf_counter()
    species = odg_pack.species
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
            ord_idx = [gene_to_idx[g] for g in odg_pack.intersected_genes]
            inter_concat_per_slice.append(sd.counts_clean[:, ord_idx])
        inter_concat = sp.vstack(inter_concat_per_slice).tocsr()
        manifold_genes = list(odg_pack.intersected_genes)
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

    # ODG positions within the manifold's gene index space
    inter_index = {g: i for i, g in enumerate(manifold_genes)}
    odg_indices_in_intersected = [inter_index[g] for g in odg_pack.odg_names]

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
        odg_indices_in_intersected=odg_indices_in_intersected,
        species=species,
        ica_coords=X_ica,
    )


# STEP 5: K sweep

def step5_ksweep(
    odg_pack: OdgPack,
    min_k: int = 3,
    max_k: int = 20,
    step: int = 1,
    subsample_frac: float = 1.0,
    doc_topic_prior: float = 0.1,
    topic_word_prior: float = 0.01,
    perc_rare_thresh: float = 0.05,
    alpha_mle_max_iter: int = 200,
    holdout_frac: float = 0.3,
) -> "KSweepResult":
    """Run an LDA K-sweep and visualize held-out perplexity alongside rare-topic count.

    For every K we fit LDA on a random 70/30 train/test split of the spots
    (split fixed across K so the K values are honestly comparable) and record:

      1. Held-out perplexity, evaluated on the test split. Lower is a better
         out-of-sample fit. The train/test split prevents the same-data
         optimism the in-sample perplexity would have.
      2. Rare-topic count: number of topics whose mean proportion across the
         train spots is below ``perc_rare_thresh`` (default 0.05). Rare topics
         indicate the model is over-splitting into spurious cell types — K is
         "too high" once rare topics appear.
      3. Mean Dirichlet alpha, estimated post-hoc from the fitted theta via
         Minka's fixed-point iteration. Used internally by the recommended-K
         rule as a soft constraint (alpha < 1 means each spot stays dominated
         by a few topics). Not plotted because sklearn's fixed doc_topic_prior
         keeps the post-hoc estimate well below 1 across most K, so the metric
         is rarely informative in this setting; the values remain available on
         the returned object for inspection.

    A combined optimal-K rule is computed and stored on the returned
    ``recommended_k`` field but not printed, since the right K rarely lines up
    with a single hard rule across datasets — use it as a starting point and
    pick the final K from the plots and the per-topic content in Step 6.

    With multiple slices, also runs a joint sweep on the concatenated
    overdispersed-gene matrix and shows two cross-slice heatmaps: relative
    perplexity and rare-topic count. Single-slice runs skip the joint sweep and
    the heatmaps (both redundant).

    Parameters
    ----------
    perc_rare_thresh : float
        Threshold below which a topic's mean proportion across spots flags it
        as rare. Default 0.05.
    alpha_mle_max_iter : int
        Maximum iterations for the Minka fixed-point alpha estimator per K.
    holdout_frac : float
        Fraction of spots held out for perplexity evaluation. Default 0.3
        (70/30 train/test split). Pass 0.0 to evaluate perplexity on the full
        training matrix (the legacy in-sample behavior).

    Returns
    -------
    KSweepResult
        Dataclass with a compact ``__repr__`` so notebook auto-display stays one
        line. Fields: ``perplexity``, ``rare_topics``, ``alpha_mean``,
        ``alpha_per_topic``, ``perc_rare_thresh``, ``recommended_k``.
    """
    start_time = time.perf_counter()
    ks = list(range(min_k, max_k + 1, step))
    sweep_perp: Dict[str, List[float]] = {}
    sweep_rare: Dict[str, List[int]] = {}
    sweep_alpha_mean: Dict[str, List[float]] = {}
    sweep_alpha_full: Dict[str, List[np.ndarray]] = {}

    n_slices = len(odg_pack.odg_per_slice)
    targets = dict(odg_pack.odg_per_slice)
    if n_slices > 1:
        targets['_joint'] = odg_pack.odg_concat

    for label, X in targets.items():
        n_spots = X.shape[0]
        if subsample_frac < 1.0 and n_spots > 1000:
            n_sub = int(n_spots * subsample_frac)
            sub_rng = np.random.default_rng(_SEED)
            idx = sub_rng.choice(n_spots, n_sub, replace=False)
            X_use = X[idx, :]
            print(f"Step 5 [{label}]: {n_sub}/{n_spots} subsample")
        else:
            X_use = X
            print(f"Step 5 [{label}]: full {n_spots} spots")

        # Train/test split. Fixed across K so the K values are honestly
        # comparable. holdout_frac=0.0 falls back to the legacy in-sample path.
        n_use = X_use.shape[0]
        if holdout_frac > 0.0 and n_use >= 10:
            split_rng = np.random.default_rng(_SEED + 7919)
            n_test = max(1, int(round(n_use * holdout_frac)))
            all_idx = np.arange(n_use)
            split_rng.shuffle(all_idx)
            test_idx = all_idx[:n_test]
            train_idx = all_idx[n_test:]
            X_train = X_use[train_idx, :]
            X_test = X_use[test_idx, :]
            print(f"   train/test split: {len(train_idx)}/{len(test_idx)} ({(1 - holdout_frac):.2f}/{holdout_frac:.2f})")
        else:
            X_train = X_use
            X_test = X_use
            if holdout_frac > 0.0:
                print(f"   too few spots ({n_use}) for a hold-out split; evaluating in-sample")

        perps: List[float] = []
        rares: List[int] = []
        alpha_means: List[float] = []
        alpha_fulls: List[np.ndarray] = []
        for k in tqdm(ks, desc=f"LDA sweep [{label}]"):
            lda = LatentDirichletAllocation(
                n_components=k,
                learning_method='online',
                learning_offset=50.,
                max_iter=5,
                random_state=_SEED,
                n_jobs=1,  # deterministic: fixed E-step chunking across machines
                doc_topic_prior=doc_topic_prior,
                topic_word_prior=topic_word_prior,
            )
            # Fit on train; transform gives the train theta for the in-sample
            # metrics. Perplexity is evaluated on the held-out test split.
            theta_lda = lda.fit_transform(X_train)
            row_sum = theta_lda.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            theta_props = theta_lda / row_sum

            # Metric 1: held-out perplexity (on the test split).
            perps.append(lda.perplexity(X_test))
            # Metric 2: rare-topic count from train theta.
            mean_theta = theta_props.mean(axis=0)
            rares.append(int(np.sum(mean_theta < perc_rare_thresh)))
            # Metric 3: Minka MLE Dirichlet alpha from train theta.
            alpha_vec = _dirichlet_alpha_mle(theta_props, max_iter=alpha_mle_max_iter)
            alpha_fulls.append(alpha_vec)
            alpha_means.append(float(alpha_vec.mean()))

        sweep_perp[label] = perps
        sweep_rare[label] = rares
        sweep_alpha_mean[label] = alpha_means
        sweep_alpha_full[label] = alpha_fulls

    # Combined optimal-K recommendation per target.
    recommended_k: Dict[str, Dict[str, object]] = {}
    for label in sweep_perp.keys():
        perp_arr = np.array(sweep_perp[label])
        rare_arr = np.array(sweep_rare[label])
        alpha_arr = np.array(sweep_alpha_mean[label])
        both = (alpha_arr < 1.0) & (rare_arr == 0)
        if both.any():
            best_local = int(np.argmin(np.where(both, perp_arr, np.inf)))
            criterion = "alpha<1 AND rare==0; lowest perplexity"
        elif (alpha_arr < 1.0).any():
            mask = alpha_arr < 1.0
            best_local = int(np.argmin(np.where(mask, perp_arr, np.inf)))
            criterion = "alpha<1 only; lowest perplexity (no K reached rare==0)"
        else:
            best_local = int(np.argmin(perp_arr))
            criterion = "fallback: lowest perplexity (no K reached alpha<1)"
        recommended_k[label] = {'k': int(ks[best_local]), 'criterion': criterion}

    # Plot 1: dual-axis line plot — perplexity (solid o) + rare topics (right, dashed s)
    from matplotlib.ticker import MaxNLocator
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    label_color = {lbl: color_cycle[i % len(color_cycle)] for i, lbl in enumerate(sweep_perp.keys())}

    for label, perps in sweep_perp.items():
        lw = 2.5 if label == '_joint' else 1.5
        ax1.plot(ks, perps, marker='o', markerfacecolor='white',
                 color=label_color[label], linewidth=lw)
    ax1.set_xlabel("K topics")
    ax1.set_ylabel("Held-out perplexity (lower = better fit)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    for label, rares in sweep_rare.items():
        lw = 2.5 if label == '_joint' else 1.5
        ax2.plot(ks, rares, marker='s', linestyle='--',
                 color=label_color[label], linewidth=lw, alpha=0.85)
    ax2.set_ylabel(f"# rare topics (mean θ < {perc_rare_thresh:.2f})")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Compact legend: one entry per slice (color) + a linestyle key for metric.
    slice_handles = [plt.Line2D([0], [0], color=label_color[lbl], linewidth=2, label=lbl)
                     for lbl in sweep_perp.keys()]
    style_handles = [
        plt.Line2D([0], [0], color='black', linewidth=1.5,
                   marker='o', markerfacecolor='white', label='Perplexity'),
        plt.Line2D([0], [0], color='black', linewidth=1.5,
                   linestyle='--', marker='s', label='Rare topics'),
    ]
    ax1.legend(handles=slice_handles + style_handles, loc='best', fontsize=8, ncol=2)

    title = "Step 5: K-sweep — Perplexity + Rare topics" + (
        ", per slice + joint" if n_slices > 1 else ""
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Cross-slice heatmaps (only meaningful with 2+ slices).
    slice_labels = [k for k in sweep_perp.keys() if k != '_joint']
    if len(slice_labels) > 1:
        # Plot 2a: relative perplexity heatmap
        mat = np.zeros((len(slice_labels), len(ks)))
        for i, lbl in enumerate(slice_labels):
            row = np.array(sweep_perp[lbl], dtype=float)
            row_range = row.max() - row.min()
            mat[i] = (row - row.min()) / row_range if row_range > 0 else 0.0
        plt.figure(figsize=(max(8, len(ks) * 0.5), 0.6 * len(slice_labels) + 2))
        sns.heatmap(mat, xticklabels=ks, yticklabels=slice_labels,
                    cmap='viridis_r', annot=False, cbar_kws={'label': 'relative perplexity'})
        raw_mat = np.array([sweep_perp[lbl] for lbl in slice_labels])
        median_per_k = np.median(raw_mat, axis=0)
        best_k_idx = int(np.argmin(median_per_k))
        plt.axvline(best_k_idx + 0.5, color='red', linestyle='--', linewidth=1.5)
        plt.title(f"Step 5: relative perplexity by K (red line = median-best K = {ks[best_k_idx]})")
        plt.xlabel("K")
        plt.ylabel("Slice")
        plt.tight_layout()
        plt.show()

        # Plot 2b: rare-topic count heatmap (raw counts, annotated)
        rare_mat = np.array([sweep_rare[lbl] for lbl in slice_labels], dtype=int)
        plt.figure(figsize=(max(8, len(ks) * 0.5), 0.6 * len(slice_labels) + 2))
        sns.heatmap(rare_mat, xticklabels=ks, yticklabels=slice_labels,
                    cmap='Reds', annot=True, fmt='d',
                    cbar_kws={'label': f'# rare topics (mean θ < {perc_rare_thresh:.2f})'})
        median_rare_per_k = np.median(rare_mat, axis=0)
        zero_rare = np.where(median_rare_per_k == 0)[0]
        if len(zero_rare) > 0:
            last_clean_k_idx = int(zero_rare.max())
            plt.axvline(last_clean_k_idx + 0.5, color='blue', linestyle='--', linewidth=1.5)
            rare_guidance = f"largest K with median rare=0 is K={ks[last_clean_k_idx]}"
        else:
            rare_guidance = "all K have rare topics — consider lowering K"
        plt.title(f"Step 5: rare topics by K  ({rare_guidance})")
        plt.xlabel("K")
        plt.ylabel("Slice")
        plt.tight_layout()
        plt.show()

    duration = time.perf_counter() - start_time
    print(f"\nStep 5 done in {duration:.2f}s")
    return KSweepResult(
        perplexity={label: dict(zip(ks, perps)) for label, perps in sweep_perp.items()},
        rare_topics={label: dict(zip(ks, rares)) for label, rares in sweep_rare.items()},
        alpha_mean={label: dict(zip(ks, alphas)) for label, alphas in sweep_alpha_mean.items()},
        alpha_per_topic={label: dict(zip(ks, sweep_alpha_full[label])) for label in sweep_alpha_full},
        perc_rare_thresh=perc_rare_thresh,
        recommended_k=recommended_k,
    )


# STEP 6: per-slice LDA -> Hungarian alignment -> consensus beta -> per-slice theta refit + projection

def step6_final_deconvolution(
    slices: Dict[str, SliceData],
    odg_pack: OdgPack,
    manifold: Manifold,
    n_topics: int,
    n_iters: int = 30,
    k_neighbors: int = 3,
    min_sim: float = 0.05,
    doc_topic_prior: float = 0.1,
    topic_word_prior: float = 0.01,
    e_step_iters: int = 50,
    n_seeds: int = 1,
) -> Model:
    """LDA fit per slice with optional multi-seed consensus to stabilize topics.

    Per slice, fit LDA with ``n_seeds`` independent random initializations
    (seeds ``_SEED`` ... ``_SEED + n_seeds - 1``), Hungarian-align all replicate
    betas (and thetas) to the seed-0 ordering, and take their mean as that
    slice's beta. Single LDA fits are highly sensitive to initialization, so
    a 5-10 seed consensus gives substantially more reproducible topic content
    at the cost of an N-times longer Step 6.

    A per-slice topic stability score — the mean pairwise cosine similarity of
    aligned topic rows across seeds — is computed and stored on the returned
    Model. 1.0 means seeds are identical; values closer to 0 mean topics drift
    across seeds and the K may be too high (or the data underdetermines that
    many topics).

    With multiple slices, the per-slice consensus betas are then Hungarian-
    aligned across slices and meaned to form ``beta_consensus``, and per-slice
    theta is refit against the frozen consensus beta via the variational E-step.
    With a single slice, the per-slice consensus beta IS the model.
    Per-slice manifold-guided projection extends consensus topics to each
    slice's full ``genes_clean``.

    Parameters
    ----------
    n_seeds : int
        Number of LDA seeds per slice to consensus-average. Default 1 keeps the
        legacy single-seed behavior. 5-10 is recommended for stable topics.
    """
    start_time = time.perf_counter()
    n_slices = len(slices)
    n_odg = len(odg_pack.odg_names)
    n_seeds = max(1, int(n_seeds))
    print(f"Step 6: LDA fit, K={n_topics}   ({n_slices} slice{'s' if n_slices > 1 else ''}, n_seeds={n_seeds})")

    per_slice_betas_unaligned: Dict[str, np.ndarray] = {}
    per_slice_thetas_lda: Dict[str, np.ndarray] = {}
    per_slice_stability: Dict[str, float] = {}
    for name in slices.keys():
        X = odg_pack.odg_per_slice[name]
        if n_seeds == 1:
            print(f"   [{name}] LDA on {X.shape[0]} spots (single seed)")
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                learning_method='batch',
                max_iter=n_iters,
                random_state=_SEED,
                n_jobs=1,  # deterministic: fixed E-step chunking across machines
                verbose=0,
                doc_topic_prior=doc_topic_prior,
                topic_word_prior=topic_word_prior,
            )
            theta_lda = lda.fit_transform(X)
            beta = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
            per_slice_betas_unaligned[name] = beta
            per_slice_thetas_lda[name] = theta_lda
            per_slice_stability[name] = 1.0
        else:
            print(f"   [{name}] LDA on {X.shape[0]} spots ({n_seeds}-seed consensus)")
            seed_betas: List[np.ndarray] = []
            seed_thetas: List[np.ndarray] = []
            for i in range(n_seeds):
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    learning_method='batch',
                    max_iter=n_iters,
                    random_state=_SEED + i,
                    n_jobs=1,  # deterministic: fixed E-step chunking across machines
                    verbose=0,
                    doc_topic_prior=doc_topic_prior,
                    topic_word_prior=topic_word_prior,
                )
                t = lda.fit_transform(X)
                b = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
                seed_betas.append(b)
                seed_thetas.append(t)

            # Hungarian-align each seed to seed 0 on cosine of beta rows. Apply
            # the same column permutation to the corresponding theta.
            anchor = seed_betas[0]
            anchor_norms = np.linalg.norm(anchor, axis=1, keepdims=True)
            anchor_norms[anchor_norms == 0] = 1.0
            anchor_norm = anchor / anchor_norms
            aligned_b: List[np.ndarray] = [anchor]
            aligned_t: List[np.ndarray] = [seed_thetas[0]]
            for i in range(1, n_seeds):
                b = seed_betas[i]
                b_norms = np.linalg.norm(b, axis=1, keepdims=True)
                b_norms[b_norms == 0] = 1.0
                b_norm = b / b_norms
                sim = anchor_norm @ b_norm.T
                row_ind, col_ind = linear_sum_assignment(-sim)
                new_b = np.zeros_like(b)
                new_b[row_ind] = b[col_ind]
                new_t = seed_thetas[i][:, col_ind]
                aligned_b.append(new_b)
                aligned_t.append(new_t)

            beta_mean = np.mean(np.stack(aligned_b, axis=0), axis=0)
            beta_mean = beta_mean / beta_mean.sum(axis=1, keepdims=True)
            theta_mean = np.mean(np.stack(aligned_t, axis=0), axis=0)
            t_sum = theta_mean.sum(axis=1, keepdims=True)
            t_sum[t_sum == 0] = 1.0
            theta_mean = theta_mean / t_sum

            stability = _topic_stability(aligned_b)
            per_slice_betas_unaligned[name] = beta_mean
            per_slice_thetas_lda[name] = theta_mean
            per_slice_stability[name] = stability
            print(f"      topic stability across {n_seeds} seeds: {stability:.3f} (1.0 = identical)")

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
            X = odg_pack.odg_per_slice[name]
            theta = _variational_e_step(X, beta_consensus, alpha=doc_topic_prior, max_iter=e_step_iters)
            sd.theta = theta
            print(f"   [{name}] theta {theta.shape}")
        # Multi-slice projection: neighbors in ICA space (cosine), with an
        # expression-cosine fallback for slice-specific genes absent from the joint
        # manifold. UMAP embedding is not used here (2D layout, non-metric).
        print("Step 6: projecting topics to full genes_clean per slice (ICA-space cosine k-NN)")
        inter_index = {g: i for i, g in enumerate(manifold.intersected_genes)}
        odg_idx_in_intersected = manifold.odg_indices_in_intersected
        odg_ica = manifold.ica_coords[odg_idx_in_intersected]
        odg_ica_norm = np.linalg.norm(odg_ica, axis=1)
        odg_ica_norm[odg_ica_norm == 0] = 1.0

        per_slice_betas_full: Dict[str, np.ndarray] = {}
        for name, sd in slices.items():
            genes_s = sd.genes_clean
            n_genes_s = len(genes_s)
            beta_full = np.zeros((n_topics, n_genes_s))

            slice_gene_to_idx = {g: i for i, g in enumerate(genes_s)}
            odg_idx_in_slice = [slice_gene_to_idx[g] for g in odg_pack.odg_names]

            norm_all = normalize(sd.counts_clean.T, axis=1)
            norm_odg = normalize(sd.counts_clean[:, odg_idx_in_slice].T, axis=1)
            sim_full = norm_all @ norm_odg.T

            for i, g in enumerate(genes_s):
                sim_row = sim_full[i].toarray().flatten() if sp.issparse(sim_full) else sim_full[i]
                if g in inter_index:
                    g_vec = manifold.ica_coords[inter_index[g]]
                    gn = np.linalg.norm(g_vec) or 1.0
                    cos = (odg_ica @ g_vec) / (odg_ica_norm * gn)
                    neighbors = np.argsort(-cos)[:k_neighbors]
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

        # Multi-slice QC: rank genes per topic by log2 fold change of beta vs
        # the mean beta of the other topics. This highlights topic-specific
        # markers rather than genes that are simply highly expressed everywhere.
        log2fc_consensus = _topic_log2fc(beta_consensus)
        topic_dict = {}
        for k in range(n_topics):
            top_idx = log2fc_consensus[k].argsort()[::-1][:15]
            topic_dict[f"Topic_{k}"] = [odg_pack.odg_names[i] for i in top_idx]
        qc_df = pd.DataFrame(topic_dict)
    else:
        # Single-slice path mirrors the legacy codeconv Step 6: LDA fit_transform
        # gives theta and beta_odg, then a single cdist-based projection extends
        # beta to the full genes_clean. QC table is built from the projected beta_final.
        only_name = next(iter(slices.keys()))
        sd = slices[only_name]
        sd.theta = per_slice_thetas_lda[only_name]
        beta_odg = per_slice_betas_unaligned[only_name]
        beta_consensus = beta_odg
        print(f"   [{only_name}] theta {sd.theta.shape}")

        print("Step 6: projecting latent topics using ICA manifold topology (cosine k-NN)")
        genes_s = sd.genes_clean
        n_genes_s = len(genes_s)

        # Neighbors are found in the ICA feature space (cosine distance), NOT in the
        # 2D UMAP embedding: the UMAP layout is for visualization and does not
        # preserve quantitative distances, whereas the ICA coordinates are the space
        # the manifold was actually built from. For a single slice every gene in
        # genes_clean is in the intersected set, so all have ICA coordinates.
        inter_index = {g: i for i, g in enumerate(manifold.intersected_genes)}
        odg_coords_manifold = manifold.ica_coords[manifold.odg_indices_in_intersected]
        all_coords = manifold.ica_coords[[inter_index[g] for g in genes_s]]
        dist_matrix = cdist(all_coords, odg_coords_manifold, metric='cosine')

        # Cosine similarity in expression space
        slice_gene_to_idx = {g: i for i, g in enumerate(genes_s)}
        odg_idx_in_slice = [slice_gene_to_idx[g] for g in odg_pack.odg_names]
        norm_all = normalize(sd.counts_clean.T, axis=1)
        norm_odg = normalize(sd.counts_clean[:, odg_idx_in_slice].T, axis=1)
        similarity = norm_all @ norm_odg.T

        beta_final = np.zeros((n_topics, n_genes_s))
        for i in range(n_genes_s):
            neighbor_idx = np.argsort(dist_matrix[i])[:k_neighbors]
            sim_row = similarity[i].toarray().flatten() if sp.issparse(similarity) else similarity[i]
            weights = sim_row[neighbor_idx]
            if np.max(weights) < min_sim:
                continue
            proj = beta_odg[:, neighbor_idx] @ weights
            if proj.sum() > 0:
                beta_final[:, i] = proj / proj.sum()

        sd.beta_final = beta_final
        per_slice_betas_full = {only_name: beta_final}
        print(f"   [{only_name}] beta_final {beta_final.shape}")

        # Single-slice QC: rank genes per topic by log2 fold change of beta_final
        # vs the mean beta of the other topics, on the full genes_clean basis.
        log2fc_final = _topic_log2fc(beta_final)
        topic_dict = {}
        for k in range(n_topics):
            top_idx = log2fc_final[k].argsort()[::-1][:15]
            topic_dict[f"Topic_{k}"] = [genes_s[i] for i in top_idx]
        qc_df = pd.DataFrame(topic_dict)

    duration = time.perf_counter() - start_time
    print(f"Step 6 done in {duration:.2f}s")
    return Model(
        n_topics=n_topics,
        odg_names=list(odg_pack.odg_names),
        beta_consensus=beta_consensus,
        qc_df=qc_df,
        per_slice_betas=per_slice_betas_full,
        per_slice_stability=dict(per_slice_stability) if per_slice_stability else None,
    )


# STEP 7: sampling engine (per-slice, with rescue)

def step7_sampling_engine(
    slices: Dict[str, SliceData],
    model: Model,
    config_path: str,
    species: str,
    low_slice_quality=False,
    min_topic_percentage: Optional[float] = 0.0,
    denoise: bool = False,
    denoise_min_pct: float = 0.25,
    denoise_logfc: float = 1.0,
    denoise_resolution: float = 0.5,
    denoise_n_hvg: int = 2000,
    denoise_n_pcs: int = 30,
    denoise_k: int = 20,
    denoise_padj: float = 0.05,
    denoise_plot: bool = True,
) -> Dict[str, dict]:
    """Per-slice hierarchical Bayesian sampling. Returns dict of per-slice cell outputs.

    Threshold-and-renormalize: any topic with theta < min_topic_percentage is
    zeroed and theta is renormalized. Default is 0.0 (no thresholding); pass a
    value > 0 to prune minor topics per spot, or pass None to fall back to the
    config's ``min_topic_percentage``. low_slice_quality=True per slice additionally
    enforces an inflate rescue: every surviving topic gets at least 1 cell.

    The inner gene-and-cell allocation is vectorized via a sequential-binomial
    batched multinomial sampler, so the draws are statistically equivalent to
    the legacy per-gene-per-cell loop but the RNG draw order differs: outputs
    will not be bit-for-bit identical to the legacy implementation under the
    same seed. The marginals (per-spot UMI conservation, per-cell topic
    assignment, gamma cell weights) are preserved.

    Optional GEX clean-up (denoise)
    -------------------------------
    When ``denoise=True``, each slice's generated cell x gene matrix is put
    through a de novo Seurat-style pass — LogNormalize, vst variable features,
    scale, PCA, SNN graph, Louvain clustering, and a Wilcoxon FindAllMarkers
    analogue — and then *cleaned*. Marker clusters are defined slice-wide, but the
    UMI reassignment is confined to within each spot: for every cluster-specific
    marker gene, its UMIs sitting in a spot's non-home cells are pooled and handed
    to that same spot's home-cluster cells (multinomial weighted by existing
    expression). If a spot has no home-cluster cell, those pooled UMIs are
    DISCARDED rather than moved to another spot, so nothing is fabricated in a
    location that never measured it. Non-marker genes are untouched; per-gene
    totals are conserved-or-reduced per spot, never inflated. The returned
    ``matrix`` for the slice is the cleaned matrix (there is no separate raw
    copy); a ``denoise_qc`` entry with cluster labels, UMIs moved/discarded, and a
    subcluster-collapse warning is attached to the slice payload.

    Parameters
    ----------
    denoise : bool
        Master toggle for the clean-up. Default False (raw generated matrix).
    denoise_min_pct, denoise_logfc, denoise_resolution : float
        Seurat min.pct, logfc.threshold (log2), and FindClusters resolution.
        Defaults 0.25 / 1.0 / 0.5.
    denoise_n_hvg, denoise_n_pcs, denoise_k : int
        Variable-feature count, PCA dimensions, and kNN neighbours for the SNN
        graph. Defaults 2000 / 30 / 20.
    denoise_padj : float
        BH-adjusted p-value cutoff for a gene to count as a cluster marker.
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
    # draws match the legacy codeconv Step 7 sequence sample-for-sample under the
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
        counts_csr = sd.counts_clean.tocsr()

        # Triplet buffers built in chunks per spot; concatenated at the end.
        rows_chunks: List[np.ndarray] = []
        cols_chunks: List[np.ndarray] = []
        data_chunks: List[np.ndarray] = []
        cell_metadata = []
        global_cell_idx = 0
        n_rescued = 0

        print(f"\nStep 7 [{name}]: sampling   low_q={slice_low_q}   min_pct={min_topic_percentage}")
        for s in tqdm(range(n_spots), desc=f"[{name}] cells", unit="spot"):
            n_total = int(sd.n_cells[s])
            if n_total == 0:
                continue

            # Threshold-and-renormalize theta. Bypassed entirely when
            # min_topic_percentage <= 0 so theta_eff stays identical to theta[s].
            if min_topic_percentage > 0:
                theta_eff = theta[s].copy()
                theta_eff[theta_eff < min_topic_percentage] = 0.0
                tsum = theta_eff.sum()
                if tsum <= 0:
                    theta_eff = np.zeros_like(theta_eff)
                    theta_eff[int(np.argmax(theta[s]))] = 1.0
                    tsum = 1.0
                theta_eff /= tsum
            else:
                theta_eff = theta[s]

            topic_dist = _safe_multinomial(rng, n_total, theta_eff)

            # Inflate rescue for low-quality slices.
            if slice_low_q:
                surviving = np.where(theta_eff > 0)[0]
                for k in surviving:
                    if topic_dist[k] == 0:
                        topic_dist[k] = 1
                        n_rescued += 1

            # Per-topic cell groups: draw gamma cell weights, register the
            # in-silico cell metadata, record the global cell indices.
            cells_weights: Dict[int, np.ndarray] = {}
            cells_indices: Dict[int, np.ndarray] = {}
            current_spot_cell_idx = 0
            for k in range(n_topics):
                n_k = int(topic_dist[k])
                if n_k <= 0:
                    continue
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
                cells_indices[k] = np.arange(global_cell_idx, global_cell_idx + n_k, dtype=np.int64)
                global_cell_idx += n_k
                current_spot_cell_idx += n_k

            # Pull the spot's nonzero (gene, count) entries directly from CSR —
            # no full-row toarray(), no per-gene Python loop.
            start, end = counts_csr.indptr[s], counts_csr.indptr[s + 1]
            if start == end:
                continue
            gene_idx_arr = counts_csr.indices[start:end]
            gene_count_arr = counts_csr.data[start:end].astype(np.int64)
            n_present = gene_idx_arr.shape[0]

            # p_topic for every (gene, topic) in one matmul. Rows that sum to
            # zero (no beta mass for any topic) fall back to uniform, matching
            # the legacy per-gene uniform fallback.
            beta_sub = beta[:, gene_idx_arr].T  # (n_present, K)
            p_topic_mat = beta_sub * theta_eff[None, :]
            row_sums = p_topic_mat.sum(axis=1)
            zero_rows = row_sums <= 0
            if zero_rows.any():
                p_topic_mat[zero_rows] = 1.0 / n_topics
                row_sums[zero_rows] = 1.0
            p_topic_mat = p_topic_mat / row_sums[:, None]

            # Batched UMI-per-topic for every present gene at once.
            umi_per_topic_mat = _batched_multinomial(rng, gene_count_arr, p_topic_mat)
            # umi_per_topic_mat: (n_present, K)

            # Per topic, batch-allocate the topic UMIs across the topic's cells.
            for k in range(n_topics):
                if k not in cells_indices:
                    continue
                n_k = cells_indices[k].shape[0]
                u_counts_k = umi_per_topic_mat[:, k]
                nonzero_g = np.flatnonzero(u_counts_k > 0)
                if nonzero_g.size == 0:
                    continue
                # Broadcast cell weights across the genes-with-UMIs in this topic.
                w_tiled = np.broadcast_to(
                    cells_weights[k][None, :], (nonzero_g.size, n_k)
                ).copy()
                cell_alloc = _batched_multinomial(
                    rng, u_counts_k[nonzero_g], w_tiled
                )  # (n_genes_with_umi, n_k)

                # Emit (row, col, data) triplets via np.where on the (gene, cell)
                # block — no Python-side cell loop.
                gi_idx, ci_idx = np.where(cell_alloc > 0)
                if gi_idx.size == 0:
                    continue
                rows_chunks.append(cells_indices[k][ci_idx])
                cols_chunks.append(gene_idx_arr[nonzero_g[gi_idx]])
                data_chunks.append(cell_alloc[gi_idx, ci_idx].astype(np.int64))

        if rows_chunks:
            rows = np.concatenate(rows_chunks)
            cols = np.concatenate(cols_chunks)
            data = np.concatenate(data_chunks)
        else:
            rows = np.zeros(0, dtype=np.int64)
            cols = np.zeros(0, dtype=np.int64)
            data = np.zeros(0, dtype=np.int64)

        final_matrix = sp.csr_matrix((data, (rows, cols)), shape=(global_cell_idx, n_genes))
        print(f"   [{name}] generated {global_cell_idx} cells   rescued {n_rescued} topic slots")

        denoise_qc = None
        if denoise:
            spot_ids = np.array([m['spot_idx'] for m in cell_metadata], dtype=np.int64)
            final_matrix, denoise_qc = _denoise_matrix(
                final_matrix,
                spot_ids,
                min_pct=denoise_min_pct,
                logfc=denoise_logfc,
                resolution=denoise_resolution,
                n_hvg=denoise_n_hvg,
                n_pcs=denoise_n_pcs,
                k_neighbors=denoise_k,
                padj_thresh=denoise_padj,
                seed=_SEED,
                label=name,
                make_plot=denoise_plot,
            )

        out[name] = {
            'matrix': final_matrix,
            'cell_metadata': cell_metadata,
            'n_cells': global_cell_idx,
            'n_rescued': n_rescued,
            'denoise_qc': denoise_qc,
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
        dqc = payload.get('denoise_qc')
        if dqc:
            slice_summary['denoise'] = {
                'n_clusters': dqc.get('n_clusters'),
                'genes_cleaned': dqc.get('genes_cleaned'),
                'umis_moved': dqc.get('umis_moved'),
                'umis_discarded': dqc.get('umis_discarded'),
                'pct_umi_retained': dqc.get('pct_umi_retained'),
                'pct_umi_dropped': dqc.get('pct_umi_dropped'),
                'similar_cluster_pairs': dqc.get('similar_cluster_pairs'),
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
