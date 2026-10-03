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
import glob
import gzip
import json
import shutil
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

try:
    import numpy as np
    import pandas as pd
    import scipy.io
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    import seaborn as sns
    # tqdm.auto, not tqdm.notebook: notebook mode raises ImportError (IProgress
    # not found) the moment the module is used outside Jupyter: in a plain
    # script, a CI run, or a terminal session. auto picks the widget bar inside
    # a notebook and the text bar everywhere else.
    from tqdm.auto import tqdm
    from sklearn.decomposition import FastICA, LatentDirichletAllocation
    from sklearn.preprocessing import StandardScaler, normalize
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree
    from scipy.optimize import linear_sum_assignment
    from scipy.special import digamma, polygamma
    import h5py
    import umap
except ImportError as e:
    missing_pkg = str(e).split()[-1]
    raise ImportError(
        f"Missing package: {missing_pkg}. "
        f"Please install required dependencies via:\n"
        f"pip install numpy pandas scipy matplotlib seaborn scikit-learn h5py tqdm umap-learn"
    )


# --- Profiling & figure-output helpers (added for STAR Protocol reproducibility) ---
import sys as _sys
import threading as _threading
import functools as _functools
try:
    import resource as _resource
except Exception:  # pragma: no cover
    _resource = None

# Every step figure is written here (600 dpi) in addition to inline display.
# Override via set_figure_output().
FIGURE_DIR = "./figures"
FIGURE_DPI = 600

# Set by step5_ksweep to the number of topics it recommends (or to the manual
# n_topics= override, if one was given), so Step 6 reads:
#     model = codeconv.step6_final_deconvolution(..., n_topics=codeconv.recommended_K)
# None until a sweep has been run.
recommended_K: Optional[int] = None


def set_figure_output(directory: str = "./figures", dpi: int = 600):
    """Configure where step figures are written and at what resolution."""
    global FIGURE_DIR, FIGURE_DPI
    FIGURE_DIR = str(directory)
    FIGURE_DPI = int(dpi)


def _show(fig_name: str):
    """Save the current figure to FIGURE_DIR at FIGURE_DPI, then show it.

    Drop-in replacement for plt.show(): inline notebook display is unchanged, but a
    high-resolution PNG is also written to ./figures for the manuscript.
    """
    try:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(fig_name)).strip("_")
        plt.savefig(os.path.join(FIGURE_DIR, f"{safe}.png"),
                    dpi=FIGURE_DPI, bbox_inches="tight")
    except Exception as exc:  # never let figure I/O break a pipeline run
        print(f"   [figure not saved: {exc}]")
    plt.show()


def _current_rss_bytes() -> int:
    """Best-effort current resident-set size of this process, in bytes."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as fh:  # Linux
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    if _resource is not None:
        maxrss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        return int(maxrss) if _sys.platform == "darwin" else int(maxrss) * 1024
    return 0


class _PeakRSS:
    """Sample process RSS in a background thread to capture a per-step peak."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = 0
        self._stop = _threading.Event()
        self._thread = None

    def start(self):
        self.peak = _current_rss_bytes()
        self._thread = _threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            rss = _current_rss_bytes()
            if rss > self.peak:
                self.peak = rss

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        rss = _current_rss_bytes()
        if rss > self.peak:
            self.peak = rss
        return self.peak


def _profile_step(fn):
    """Decorator: report peak process RAM for a step, mirroring the per-step timing."""
    @_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        monitor = _PeakRSS().start()
        try:
            return fn(*args, **kwargs)
        finally:
            peak_gb = monitor.stop() / (1024 ** 3)
            print(f"   [{fn.__name__}] peak RAM: {peak_gb:.2f} GB")
    return wrapper


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
    # source platform key, see PLATFORM_PROFILES
    platform: str = 'visium'
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
    """Output of Step 4. Joint manifold on intersected genes."""
    embedding: np.ndarray
    intersected_genes: List[str]
    odg_indices_in_intersected: List[int]
    species: str


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
    custom plotting, or persistence; only the default display is suppressed.
    """
    perplexity: Dict[str, Dict[int, float]]
    rare_topics: Dict[str, Dict[int, int]]
    alpha_mean: Dict[str, Dict[int, float]]
    alpha_per_topic: Dict[str, Dict[int, np.ndarray]]
    perc_rare_thresh: float
    recommended_k: Dict[str, Dict[str, object]]
    k_tol: float = 0.01
    k_band: float = 0.05
    k_rule: str = "percentile"
    chosen_k: Optional[int] = None

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
        rec_txt = ""
        if self.recommended_k:
            label = '_joint' if '_joint' in self.recommended_k else next(iter(self.recommended_k))
            rec_txt = f", recommended_K={self.recommended_k[label]['k']}"
        chosen_txt = f", chosen_K={self.chosen_k}" if self.chosen_k is not None else ""
        return f"KSweepResult(targets={targets}, {k_span}{rec_txt}{chosen_txt})"

    @property
    def k(self) -> Optional[int]:
        """The single recommended K to carry into Step 6.

        Uses the joint sweep when several slices were swept, otherwise the only
        slice. ``chosen_k`` (a manual override passed to step5_ksweep) always
        wins, so ``ksweep.k`` is the value the run actually proceeds with.
        """
        if self.chosen_k is not None:
            return int(self.chosen_k)
        if not self.recommended_k:
            return None
        label = '_joint' if '_joint' in self.recommended_k else next(iter(self.recommended_k))
        return int(self.recommended_k[label]['k'])

    def summary(self) -> pd.DataFrame:
        """Per-K sweep metrics as a tidy DataFrame, for the record or for replotting."""
        rows = []
        for label, perps in self.perplexity.items():
            p = np.array([perps[k] for k in sorted(perps)], dtype=float)
            span = float(p.max() - p.min())
            in_set = set(self.recommended_k.get(label, {}).get('k_in_band', []))
            for i, k in enumerate(sorted(perps)):
                rows.append({
                    'target': label,
                    'K': k,
                    'perplexity': float(p[i]),
                    'perplexity_norm': float((p[i] - p.min()) / span) if span > 0 else 0.0,
                    'pct_above_best': float(p[i] / p.min() - 1.0) if p.min() > 0 else 0.0,
                    'in_low_band': bool(k in in_set),
                    'rare_topics': int(self.rare_topics[label][k]),
                    'alpha_mean': float(self.alpha_mean[label][k]),
                })
        return pd.DataFrame(rows)


def recommend_k(
    perplexity: Dict[int, float],
    rule: str = "percentile",
    tol: float = 0.01,
    band: float = 0.05,
    rare_topics: Optional[Dict[int, int]] = None,
    alpha_mean: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    """Pick the number of topics K from a held-out-perplexity sweep, automatically.

    The protocol's guidance is "take the highest K that still sits in the
    low-perplexity regime, because rare topics are wanted here": rare topics
    correspond to minor cell populations we intend to recover as individual
    cells, not to over-splitting. This function is that sentence written as a
    rule, so the choice stops being a visual judgement call.

    Every rule returns the **largest** K inside a low-perplexity set; they differ
    only in how that set is defined.

    Rule ``percentile`` (default, ``band = 0.05``). Keep the K values whose
    held-out perplexity falls in the best ``band`` fraction of the swept
    perplexities by rank, the 95th-percentile rule, and return the largest.
    At band = 0.05 the cut sits essentially at the minimum, so in practice this
    reads as "among the K values statistically tied with the best fit, take the
    most resolved one". Widening the band past ~0.10 starts admitting K values
    that are visibly worse, which is what over-calls K.

    Rule ``relative_tolerance``. Keep every K whose held-out perplexity is within
    ``tol`` of the best K,

        P(K) <= min(P) * (1 + tol)                      tol = 0.01 -> within 1%

    and return the largest. Its advantage is that it is invariant to the sweep's
    extent by construction: the answer does not move when ``min_k`` or ``max_k``
    change, only when the fit genuinely changes, and it is directly
    interpretable in units of fit quality rather than rank. Worth comparing
    against the default on any new dataset.

    Rule ``largest_in_band``. Min-max normalize the sweep,
    ``p_norm(K) = (P(K) - min P) / (max P - min P)``, and keep every K with
    ``p_norm(K) <= band``. Intuitive to read off the plot, but the normalization
    is anchored to ``max P``, which is whatever the smallest swept K happened to
    score, so the recommendation shifts when the sweep range changes even though
    the curve has not. Kept for reading the shaded band on Figure 5.

    Rule ``lowest_perplexity``. Plain argmin, the conventional choice, kept for
    comparison. It cannot return a K larger than the best-fitting one, so it
    systematically under-calls K in this setting.

    Rule ``conservative``. Lowest perplexity among K with mean Dirichlet
    alpha < 1 and no rare topics; needs ``rare_topics`` and ``alpha_mean``. This
    was the module's original rule. It optimizes for topics that are all
    well-populated, which is the right target for proportion-style deconvolution
    and the wrong one here, so it is no longer the default.

    Whatever the rule, a recommendation equal to ``max_k`` means perplexity had
    not started to rise again and the sweep is too narrow to bound K from above;
    the returned ``at_max_k`` flag says so.

    Parameters
    ----------
    perplexity : Dict[int, float]
        Held-out perplexity keyed by K, e.g. ``ksweep.perplexity['Glioblastoma']``.
    rule : str
        ``percentile`` (default) | ``relative_tolerance`` | ``largest_in_band`` |
        ``lowest_perplexity`` | ``conservative``.
    tol : float
        Relative tolerance on the best perplexity, for ``relative_tolerance``.
        Default 0.01 (1%). Lower it for a stricter, smaller K.
    band : float
        Band width as a fraction, for ``percentile`` and ``largest_in_band``.
        Default 0.05 (the 95th-percentile rule). Raise it for a larger K.
    rare_topics, alpha_mean : Optional[Dict[int, float]]
        Required by ``conservative`` only.

    Returns
    -------
    dict
        ``k``, ``criterion``, ``rule``, ``tol``, ``band``, ``at_max_k``,
        ``k_in_band`` (every K in the low-perplexity set) and ``perplexity_norm``.

    Examples
    --------
    >>> ksweep = codeconv.step5_ksweep(odg_pack, min_k=3, max_k=15)
    >>> recommend_k(ksweep.perplexity['Glioblastoma'], tol=0.01)['k']
    """
    if not perplexity:
        raise ValueError("recommend_k: empty perplexity sweep.")

    ks = sorted(perplexity.keys())
    p = np.array([float(perplexity[k]) for k in ks])
    span = float(p.max() - p.min())
    p_norm = (p - p.min()) / span if span > 0 else np.zeros_like(p)
    norm_map = {int(k): float(v) for k, v in zip(ks, p_norm)}

    if rule == "relative_tolerance":
        if tol < 0.0:
            raise ValueError(f"recommend_k: tol must be >= 0, got {tol}.")
        threshold = float(p.min()) * (1.0 + tol)
        in_band = [int(k) for k, v in zip(ks, p) if v <= threshold]
        k_star = max(in_band)
        criterion = (
            f"largest K with held-out perplexity within {tol:.1%} of the best "
            f"(<= {threshold:.1f}; best = {p.min():.1f} at K = {ks[int(np.argmin(p))]})"
        )
    elif rule == "largest_in_band":
        if not (0.0 <= band <= 1.0):
            raise ValueError(f"recommend_k: band must be in [0, 1], got {band}.")
        in_band = [int(k) for k, v in zip(ks, p_norm) if v <= band]
        if not in_band:  # only possible if band < 0 rounding; keep it safe
            in_band = [int(ks[int(np.argmin(p))])]
        k_star = max(in_band)
        criterion = (
            f"largest K with normalized held-out perplexity <= {band:.2f} "
            f"(low-perplexity band = best {band * 100:.0f}% of the sweep range)"
        )
    elif rule == "percentile":
        if not (0.0 <= band <= 1.0):
            raise ValueError(f"recommend_k: band must be in [0, 1], got {band}.")
        threshold = float(np.percentile(p, band * 100.0))
        in_band = [int(k) for k, v in zip(ks, p) if v <= threshold]
        if not in_band:
            in_band = [int(ks[int(np.argmin(p))])]
        k_star = max(in_band)
        criterion = (
            f"largest K in the best {band:.0%} of swept perplexities by rank "
            f"(<= {threshold:.1f})"
        )
    elif rule == "lowest_perplexity":
        in_band = [int(ks[int(np.argmin(p))])]
        k_star = in_band[0]
        criterion = "lowest held-out perplexity"
    elif rule == "conservative":
        if rare_topics is None or alpha_mean is None:
            raise ValueError("recommend_k: rule='conservative' needs rare_topics and alpha_mean.")
        rare_arr = np.array([rare_topics[k] for k in ks])
        alpha_arr = np.array([alpha_mean[k] for k in ks])
        both = (alpha_arr < 1.0) & (rare_arr == 0)
        if both.any():
            idx = int(np.argmin(np.where(both, p, np.inf)))
            criterion = "alpha<1 AND rare==0; lowest perplexity"
        elif (alpha_arr < 1.0).any():
            idx = int(np.argmin(np.where(alpha_arr < 1.0, p, np.inf)))
            criterion = "alpha<1 only; lowest perplexity (no K reached rare==0)"
        else:
            idx = int(np.argmin(p))
            criterion = "fallback: lowest perplexity (no K reached alpha<1)"
        k_star = int(ks[idx])
        in_band = [k_star]
    else:
        raise ValueError(
            f"recommend_k: unknown rule {rule!r}; expected 'relative_tolerance', "
            f"'largest_in_band', 'percentile', 'lowest_perplexity' or 'conservative'."
        )

    return {
        'k': int(k_star),
        'criterion': criterion,
        'rule': rule,
        'tol': float(tol),
        'band': float(band),
        'k_in_band': in_band,
        'at_max_k': bool(k_star == ks[-1]),
        'perplexity_norm': norm_map,
    }


def cellchat_spatial_factors(
    cells: Dict[str, dict],
    slices: Dict[str, "SliceData"],
    spot_pitch_um: float = 100.0,
    interaction_range_um: Optional[float] = 50.0,
) -> Dict[str, Dict[str, float]]:
    """Derive CellChat's ``spatial.factors`` (ratio, tol) for the reconstructed cells.

    **The pixel-to-micron ratio is calibrated from the spot pitch, not from
    ``spot_diameter_fullres``.** CellChat's vignette computes it as
    ``spot.size / spot_diameter_fullres`` with ``spot.size = 65``, described as
    "the theoretical spot size (um) in 10X Visium", while 10x's physical capture
    spot is 55 um. Neither constant reproduces the real geometry: measured on the
    worked example (11 mm CytAssist, ``spot_diameter_fullres = 219.17 px``, true
    spot pitch 362.93 px), 55 um implies a 91 um pitch and 65 um implies 108 um,
    against a documented pitch of 100 um. The pitch is the unambiguous number:
    it is fixed by the array, verifiable from the spot coordinates, and identical
    across Visium capture areas, so the ratio is derived from it directly::

        ratio = spot_pitch_um / median nearest-neighbour distance between spot centres

    which for the worked example gives 0.2755 um/pixel and implies
    ``spot_diameter_fullres`` spans ~60 um, i.e. between the two conventions.

    ``tol``, half the characteristic centre-to-centre distance. CellChat suggests
    ``spot_size/2`` for Visium spots, but after deconvolution the objects are
    reconstructed cells packed *inside* each spot, so the characteristic spacing
    is the cell-to-cell nearest-neighbour distance, not the spot pitch. Returned
    as half the median nearest-neighbour distance, following the documented
    fallback ("if the cell/spot size is not known ... tol can be the half value of
    the minimum centre-to-centre distance"). Passing the spot-scale
    ``55/2 = 27.5 um`` instead silently widens every range by that amount.

    Sub-spot geometry check. Within a spot, cells are positioned by the
    Fibonacci/Vogel packing of ``step8_geometry_and_placement``, not measured. The
    question that matters is therefore not how many neighbours share a spot, but
    whether the packing **decides which pairs count**. It does not, as long as the
    cutoff covers the whole spot footprint: then every intra-spot pair is inside
    the range whatever the packing did with it, and the inference reduces to
    co-occupancy in the same capture spot, which is measured. So the diagnostic
    reported here is ``intra_spot_pairs_captured``, the fraction of within-spot
    cell pairs falling inside ``interaction_range_um + tol``. At 100% the result is
    invariant to the packing; below ~95% the packing starts selecting pairs and
    short-range results become an artefact of the placement model. On the worked
    example (spot footprint 60.4 um) the coverage is 100% from
    ``interaction_range_um = 50`` upward and falls to 81% at 30 um.

    ``same_spot_fraction`` is also returned, as context for how much of the signal
    is spot co-occupancy versus between-spot architecture, but it is not an error
    condition.

    What this function deliberately does **not** return is ``scale.distance``.
    That factor is applied to ``d.spatial``, which in ``computeCommunProb`` is a
    *cell-group x cell-group* matrix of trimmed-mean neighbour distances built by
    ``computeRegionDistance``, not a cell-to-cell distance matrix. Its minimum
    therefore depends on the cell-type labels, which do not exist until the
    annotation step in R. Derive it there, with CellChat's own function::

        res <- computeRegionDistance(coordinates = spatial.locs, meta = meta.t,
                                     interaction.range = 100, ratio = ratio, tol = tol,
                                     contact.dependent = TRUE, contact.range = 100)
        scale.distance <- 1.5 / min(res$d.spatial, na.rm = TRUE)

    Call after ``step8_geometry_and_placement``. ``step9_export_results`` also
    calls it and records the values in ``run_summary.json``.

    Parameters
    ----------
    spot_pitch_um : float
        Centre-to-centre distance between capture spots, in microns. 100.0 for
        every 10x Visium capture area; pass the array pitch for other platforms.
    interaction_range_um : Optional[float]
        The CellChat ``interaction.range`` you intend to use, in microns.
        Default 50.0: with a 60 um Visium spot footprint this is the shortest
        range that still captures every intra-spot pair, so it is the most
        conservative packing-invariant choice. Diagnostics are reported at a
        cutoff of ``interaction_range_um + tol``. Pass None to skip them.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Per slice: ``ratio`` (um per pixel), ``tol``, ``min_distance_um``,
        ``median_nn_distance_um``, ``spot_pitch_px``, ``spot_diameter_um``,
        ``n_cells``, and, unless ``interaction_range_um`` is None,
        ``interaction_range_um``, ``cutoff_um``, ``mean_neighbours``,
        ``same_spot_fraction`` and ``intra_spot_pairs_captured``.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, payload in cells.items():
        sd = slices[name]
        coords = payload.get('final_coords')
        if coords is None:
            raise ValueError(
                f"cellchat_spatial_factors [{name}]: cells have no coordinates yet: "
                f"run step8_geometry_and_placement first."
            )
        if np.asarray(coords).shape[0] < 2:
            raise ValueError(f"cellchat_spatial_factors [{name}]: need at least 2 cells.")

        # Ratio from the spot pitch, measured on the original spot centres.
        spot_xy = np.asarray(sd.coords[:, :2], dtype=float)
        if spot_xy.shape[0] < 2:
            raise ValueError(f"cellchat_spatial_factors [{name}]: need at least 2 spots.")
        d_spot, _ = cKDTree(spot_xy).query(spot_xy, k=2)
        pitch_px = float(np.median(d_spot[:, 1]))
        if not np.isfinite(pitch_px) or pitch_px <= 0:
            raise ValueError(f"cellchat_spatial_factors [{name}]: degenerate spot pitch.")
        ratio = float(spot_pitch_um) / pitch_px

        xy_um = np.asarray(coords[:, :2], dtype=float) * ratio

        # k-d tree nearest-neighbour query, not a dense distance matrix: at
        # 45,530 cells the dense form would be ~16.6 GB.
        tree = cKDTree(xy_um)
        k_query = min(8, xy_um.shape[0])
        dists, _ = tree.query(xy_um, k=k_query)
        neighbours = dists[:, 1:]
        nonzero = neighbours[neighbours > 0]
        if nonzero.size == 0:
            raise ValueError(
                f"cellchat_spatial_factors [{name}]: all cells share coordinates."
            )
        d_min = float(nonzero.min())
        per_cell_nn = np.where(neighbours[:, 0] > 0, neighbours[:, 0], np.nan)
        d_med = float(np.nanmedian(per_cell_nn))
        tol = d_med / 2.0

        spot_diameter_um = float(sd.scale_factors['spot_diameter_fullres']) * ratio

        res = {
            'ratio': ratio,
            'tol': tol,
            'min_distance_um': d_min,
            'median_nn_distance_um': d_med,
            'spot_pitch_px': pitch_px,
            'spot_diameter_um': spot_diameter_um,
            'n_cells': int(xy_um.shape[0]),
        }
        print(
            f"CellChat [{name}]: spot pitch {pitch_px:.1f} px = {spot_pitch_um:.0f} um   "
            f"-> ratio={ratio:.5f} um/pixel   spot footprint {spot_diameter_um:.1f} um   "
            f"cell nearest-neighbour distance min={d_min:.1f} median={d_med:.1f} um   "
            f"-> spatial.factors(ratio={ratio:.5f}, tol={tol:.2f})"
        )

        if interaction_range_um is not None:
            cutoff = float(interaction_range_um) + tol
            counts = np.asarray(
                tree.query_ball_point(xy_um, r=cutoff, return_length=True)
            ) - 1

            same_frac = float('nan')
            intra_cov = float('nan')
            barcodes = payload.get('final_barcodes')
            if barcodes is not None:
                parent = np.array([str(b).split('-Topic-')[0] for b in barcodes])
                _, spot_of_cell = np.unique(parent, return_inverse=True)

                # Does the packing decide which intra-spot pairs count? Spots hold
                # a handful of cells, so the exact per-spot pair enumeration is cheap.
                order = np.argsort(spot_of_cell, kind='stable')
                bounds = np.cumsum(np.bincount(spot_of_cell))[:-1]
                n_pairs = n_inside = 0
                for grp in np.split(order, bounds):
                    if grp.size < 2:
                        continue
                    pts = xy_um[grp]
                    dd = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
                    iu = np.triu_indices(grp.size, 1)
                    n_pairs += iu[0].size
                    n_inside += int((dd[iu] <= cutoff).sum())
                intra_cov = n_inside / n_pairs if n_pairs else float('nan')

                # Same-spot share of neighbours, sampled to bound the cost.
                rng_s = np.random.default_rng(0)
                probe = rng_s.choice(xy_um.shape[0], min(4000, xy_um.shape[0]), replace=False)
                same = tot = 0
                for i in probe:
                    nb = [j for j in tree.query_ball_point(xy_um[i], r=cutoff) if j != i]
                    tot += len(nb)
                    if nb:
                        same += int((spot_of_cell[nb] == spot_of_cell[i]).sum())
                same_frac = same / tot if tot else float('nan')

            res.update({
                'interaction_range_um': float(interaction_range_um),
                'cutoff_um': cutoff,
                'mean_neighbours': float(counts.mean()),
                'same_spot_fraction': same_frac,
                'intra_spot_pairs_captured': intra_cov,
            })
            print(
                f"   interaction.range={interaction_range_um:g} um "
                f"(effective cutoff {cutoff:.1f} um): {counts.mean():.1f} neighbours per cell, "
                f"{same_frac:.0%} of them in the same spot, "
                f"{intra_cov:.1%} of intra-spot pairs captured"
            )
            if np.isfinite(intra_cov) and intra_cov < 0.95:
                print(
                    f"   WARNING: only {intra_cov:.0%} of within-spot cell pairs fall inside this "
                    f"range, so the Fibonacci/Vogel packing of Step 8 is selecting which pairs are "
                    f"counted and the result is partly an artefact of the placement model. Use "
                    f"interaction.range >= {spot_diameter_um - tol:.0f} um to cover the whole "
                    f"{spot_diameter_um:.0f} um spot footprint and keep the inference "
                    f"packing-invariant."
                )
        out[name] = res
    return out


# --- Platform profiles ---------------------------------------------------------
#
# Everything downstream of Step 1 only needs four things per slice: a counts
# matrix, per-barcode coordinates, a spot footprint, and a spot pitch. Visium
# supplies all four in the SpaceRanger layout; other spot-based platforms supply
# the first two and leave the geometry to be filled in from what is known about
# the array. This table is that knowledge.
#
# One calibration rule covers every platform: the physical scale of the
# coordinates is recovered from the array's *known* centre-to-centre spacing
# divided by the spacing actually measured in the file,
#
#     um_per_unit = pitch_um / median nearest-neighbour distance between capture units
#
# This avoids depending on any per-vendor field whose meaning is ambiguous. On
# the worked Visium example it recovers 0.2755 um/pixel, against the array's
# documented 100 um spot pitch.
#
# `coord_units`:
#   'grid'  , coordinates are integer array indices; multiply by `pitch_um`.
#   'scaled', coordinates are in a linear unit of unknown scale; calibrate with
#              the rule above.
PLATFORM_PROFILES: dict = {
    'visium': {
        'label': '10x Visium',
        'spot_diameter_um': 55.0,
        'pitch_um': 100.0,
        'coord_units': 'scaled',
        'has_image': True,
    },
    'dbit_seq': {
        'label': 'DBiT-seq',
        # Microchannel width; the 10 / 25 / 50 um variants are distinguished by
        # `pixel_um`, read from the sample name when it carries one. Pitch is
        # twice the width, because channels alternate with equal spacing.
        'spot_diameter_um': 50.0,
        'pitch_um': 100.0,
        'coord_units': 'grid',
        'has_image': False,
    },
}


def detect_platform(data_path: str) -> str:
    """Identify which spot-based platform a dataset directory or file came from.

    Detection is by layout, in order of specificity:

    ==================  =========================================================
    ``visium``          a ``spatial/`` directory with ``scalefactors_json.json``
    ``dbit_seq``        integer ``AxB`` labels on the 50 x 50 microfluidic grid
    ==================  =========================================================

    Returns the platform key; raises if nothing matches. Detection is a
    heuristic; pass ``platform=`` to ``step1_acquisition_and_anchoring`` to
    override it.
    """
    if os.path.isdir(data_path):
        if os.path.exists(os.path.join(data_path, 'spatial', 'scalefactors_json.json')):
            return 'visium'
        tables = _find_count_tables(data_path)
        if not tables:
            raise FileNotFoundError(
                f"detect_platform: {data_path} has no spatial/ directory and no "
                f"counts table (.tsv/.csv/.txt, optionally gzipped)."
            )
        table = tables[0]
    else:
        table = data_path

    labels = _peek_table_labels(table)
    if labels is None:
        raise ValueError(
            f"detect_platform: could not read coordinate labels from {table}. "
            f"Pass platform= explicitly."
        )
    parsed = _parse_coord_labels(labels)
    if parsed is None:
        raise ValueError(
            f"detect_platform: labels in {table} are not coordinate-like "
            f"(expected 'AxB' or 'A_B'). Pass platform= explicitly."
        )
    arr = np.asarray(parsed, dtype=float)
    integral = np.allclose(arr, np.round(arr))
    if integral and arr.min() >= 0 and arr.max() <= 50:
        return 'dbit_seq'
    raise ValueError(
        f"detect_platform: coordinate labels in {table} do not match a known "
        f"platform (integer indices within a 50 x 50 grid are expected for "
        f"DBiT-seq). Pass platform= explicitly."
    )


def _find_count_tables(directory: str) -> List[str]:
    """Delimited counts tables in a directory.

    When several candidates exist the largest is taken, because the counts
    matrix is invariably the biggest table in a sample directory.
    """
    pats = ('*.tsv', '*.tsv.gz', '*.csv', '*.csv.gz', '*.txt', '*.txt.gz')
    hits: List[str] = []
    for p in pats:
        hits.extend(glob.glob(os.path.join(directory, p)))
    return sorted(hits, key=lambda h: -os.path.getsize(h))


def _read_counts_table(path: str, sep: str, chunk_rows: int = 2000):
    """Read a delimited dense counts table into a sparse matrix, in chunks.

    Returns ``(csr, row_labels, col_labels)`` with rows and columns in file
    order; the caller decides which axis is which.

    Dense text matrices from these platforms can be large: a whole-transcriptome
    grid is easily a gigabyte of text, which as a dense float64 frame would not
    open on the 16 GB workstation this protocol targets. Parsing line by line
    and keeping only the non-zeros holds peak memory at roughly one row, and the
    result is a few hundred megabytes at most because these matrices are 95-99%
    zeros. Files under 100 MB skip the machinery and are read in one go.
    """
    size_mb = os.path.getsize(path) / 1e6

    # Large text matrices are parsed once and cached next to the source, because
    # streaming 1.8 GB of text takes minutes while reloading the compressed
    # sparse form takes under a second. The cache is invalidated whenever the
    # source file is newer.
    cache = path + '.codeconv_cache.npz'
    if size_mb >= 100 and os.path.exists(cache) and \
            os.path.getmtime(cache) >= os.path.getmtime(path):
        z = np.load(cache, allow_pickle=False)
        mat = sp.csr_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
        print(f"   cached parse: {cache}")
        return mat, [str(x) for x in z['rows']], [str(x) for x in z['cols']]

    if size_mb < 100:
        df = pd.read_csv(path, sep=sep, index_col=0)
        if df.shape[1] == 0:  # wrong delimiter guess
            df = pd.read_csv(path, sep=None, engine='python', index_col=0)
        # A trailing delimiter on the header line yields a blank all-NaN column.
        df = df.loc[:, [c for c in df.columns if str(c).strip() != '' and not df[c].isna().all()]]
        return sp.csr_matrix(df.to_numpy(dtype=np.float64)), \
            [str(i) for i in df.index], [str(c) for c in df.columns]

    # Streaming line-by-line parse. pandas carries a large per-column overhead,
    # which dominates on matrices tens of thousands of columns wide, so each
    # line is parsed straight into a numeric array with
    # np.fromstring and reduced to its non-zeros immediately. Peak memory is one
    # row, and only the non-zeros are retained.
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        header = f.readline().rstrip('\r\n').split(sep)
        col_labels = [str(c) for c in header[1:] if str(c).strip() != '']
        n_cols = len(col_labels)
        print(f"   large table ({size_mb:.0f} MB, {n_cols} columns); streaming")

        indptr = [0]
        indices: List[np.ndarray] = []
        values: List[np.ndarray] = []
        row_labels: List[str] = []
        nnz = 0
        for line in f:
            label, _, rest = line.rstrip('\r\n').partition(sep)
            if not label:
                continue
            row = np.fromstring(rest, dtype=np.float32, sep=sep)
            if row.size > n_cols:
                row = row[:n_cols]
            elif row.size < n_cols:
                row = np.pad(row, (0, n_cols - row.size))
            nz = np.flatnonzero(row)
            indices.append(nz.astype(np.int32))
            values.append(row[nz])
            nnz += nz.size
            indptr.append(nnz)
            row_labels.append(str(label))
            if len(row_labels) % 2000 == 0:
                print(f"      {len(row_labels)} rows, {nnz:,} non-zeros", end='\r')

    print(f"      {len(row_labels)} rows, {nnz:,} non-zeros      ")
    mat = sp.csr_matrix(
        (np.concatenate(values) if values else np.zeros(0, np.float32),
         np.concatenate(indices) if indices else np.zeros(0, np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(row_labels), n_cols),
    )
    try:
        np.savez_compressed(
            cache, data=mat.data, indices=mat.indices, indptr=mat.indptr,
            shape=np.asarray(mat.shape), rows=np.asarray(row_labels),
            cols=np.asarray(col_labels),
        )
        print(f"   cached the parse to {os.path.basename(cache)} for future runs")
    except Exception as exc:  # a read-only data directory must not be fatal
        print(f"   (could not write parse cache: {exc})")
    return mat, row_labels, col_labels


def _dbit_pixel_um(name: str, default: float = 50.0) -> float:
    """Infer the DBiT-seq microchannel width from a sample name.

    GEO sample names in the DBiT-seq series encode the resolution, e.g.
    ``GSM4189611_50t`` is the 50 um tail section and ``GSM4189615`` variants
    carry ``10``. Returns the default when the name says nothing.
    """
    m = re.search(r'(?<!\d)(10|25|50)(?=[^\d]|$)', str(name))
    return float(m.group(1)) if m else float(default)


def _peek_table_labels(path: str, n: int = 200) -> Optional[List[str]]:
    """First-column labels of a delimited table, without reading the whole file."""
    base = os.path.basename(path).lower()
    try:
        sep = ',' if '.csv' in base else '\t'
        head = pd.read_csv(path, sep=sep, index_col=0, nrows=n)
        if head.shape[1] == 0:  # wrong delimiter guess, e.g. a comma-separated .txt
            head = pd.read_csv(path, sep=None, engine='python', index_col=0, nrows=n)
    except Exception:
        return None
    labels = [str(x) for x in head.index]
    # Coordinate labels may live in the header instead (genes x spots orientation).
    if _parse_coord_labels(labels) is None:
        cols = [str(c) for c in head.columns]
        if _parse_coord_labels(cols) is not None:
            return cols
    return labels


def _parse_coord_labels(labels) -> Optional[np.ndarray]:
    """Parse ``AxB`` / ``A_B`` / ``A,B`` labels into an (n, 2) float array, or None."""
    pat = re.compile(r'^\s*(-?\d+(?:\.\d+)?)\s*[x_,\-]\s*(-?\d+(?:\.\d+)?)\s*$', re.I)
    out = []
    for lab in labels:
        m = pat.match(str(lab))
        if m is None:
            return None
        out.append((float(m.group(1)), float(m.group(2))))
    return np.asarray(out, dtype=float) if out else None


def make_dummy_tissue_image(
    coords_fullres: np.ndarray,
    spatial_dir: str,
    spot_diameter_fullres: Optional[float] = None,
    margin_frac: float = 0.03,
    hires_px: int = 2000,
    lowres_px: int = 600,
    quiet: bool = False,
) -> dict:
    """Write a black placeholder tissue image and matching scale factors.

    Several spot-based platforms ship no registered histology at all (DBiT-seq
    samples vary), but ``Seurat::Load10X_Spatial`` and the rest of the 10x ecosystem
    expect an image to exist. Rather than fail, write a black canvas sized to the
    spot grid so the object loads and every spatial plot still works; the tissue
    background is simply blank.

    The canvas is anchored at the coordinate origin and extended to the maximum
    coordinate plus a small margin, so the existing pixel coordinates stay valid
    and no barcode-to-position mapping has to be rewritten.

    Parameters
    ----------
    coords_fullres : np.ndarray
        (n, >=2) array of full-resolution pixel coordinates as (row, col).
    spatial_dir : str
        Directory to write ``tissue_hires_image.png``, ``tissue_lowres_image.png``
        into. Created if missing.
    spot_diameter_fullres : Optional[float]
        Spot footprint in full-resolution pixels, for the returned scale factors.
        Defaults to 1/40th of the larger canvas dimension.
    margin_frac : float
        Blank margin beyond the outermost spot, as a fraction of the extent.

    Returns
    -------
    dict
        Scale-factor entries (``tissue_hires_scalef``, ``tissue_lowres_scalef``,
        ``spot_diameter_fullres``, ``fiducial_diameter_fullres``) to write into
        ``scalefactors_json.json``.
    """
    import matplotlib.image as mpimg

    xy = np.asarray(coords_fullres, dtype=float)[:, :2]
    if xy.size == 0:
        raise ValueError("make_dummy_tissue_image: no coordinates given.")

    extent = np.nanmax(xy, axis=0)
    pad = margin_frac * float(np.nanmax(extent)) if np.nanmax(extent) > 0 else 1.0
    h_full = float(extent[0] + pad)
    w_full = float(extent[1] + pad)
    if not (np.isfinite(h_full) and np.isfinite(w_full)) or min(h_full, w_full) <= 0:
        raise ValueError("make_dummy_tissue_image: degenerate coordinate extent.")

    if spot_diameter_fullres is None:
        spot_diameter_fullres = max(h_full, w_full) / 40.0

    os.makedirs(spatial_dir, exist_ok=True)
    scalefs = {}
    for fname, target in (("tissue_hires_image.png", hires_px),
                          ("tissue_lowres_image.png", lowres_px)):
        scalef = float(target) / max(h_full, w_full)
        h_px = max(1, int(round(h_full * scalef)))
        w_px = max(1, int(round(w_full * scalef)))
        mpimg.imsave(os.path.join(spatial_dir, fname),
                     np.zeros((h_px, w_px, 3), dtype=np.uint8))
        key = 'tissue_hires_scalef' if 'hires' in fname else 'tissue_lowres_scalef'
        scalefs[key] = scalef

    scalefs['spot_diameter_fullres'] = float(spot_diameter_fullres)
    scalefs['fiducial_diameter_fullres'] = float(spot_diameter_fullres) * 1.75

    if not quiet:
        print(
            f"   WARNING: no tissue image detected, providing a dummy one. "
            f"Wrote a black {int(w_full)} x {int(h_full)} full-resolution canvas "
            f"({margin_frac:.0%} margin) covering {xy.shape[0]} spots. Spatial plots will "
            f"render on a blank background; H&E overlays are not available for this sample."
        )
    return scalefs


# --- Built-in species configuration -------------------------------------------
#
# The human and mouse profiles ship inside the module so a fresh install runs
# without any extra download: pass config_path=None (or point it at a file that
# does not exist) and these are used. An external codeconv_config.json always
# wins when it is present, so existing runs and hand-tuned profiles are
# unaffected. Regenerate a species block with the CoexpressDeconvolve Config
# Maker and write it to a JSON file to override.
#
# NOTE: the mouse hk_profiles and engine_parameters below mirror the human
# reference and are placeholders. Recalibrate them from a mouse scRNA-seq
# reference before publication-grade mouse runs. The 'other' block is
# intentionally empty: non-hs/mm organisms must supply their own profile.
DEFAULT_CONFIG: dict = {
    "min_topic_percentage": 0.05,
    "species_profiles": {
        "hs": {
            "noise_regex": "^MT-|^RP[SL][0-9]+|^LINC|^MIR|^AC[0-9]+",
            "hk_profiles": {
                "RPL19": 2.529822467672188, "TUBB": 0.6653868555153559,
                "EEF1G": 1.409845725236883, "PPIA": 1.4689432838260164,
                "GAPDH": 1.9870339974714675, "ABCF1": 0.2683943094020617,
                "SDHA": 0.22436799098809676, "OAZ1": 1.5212192863178522,
                "G6PD": 0.10922542882001043, "ALAS1": 0.10135028595398016,
                "GUSB": 0.1690932556829679, "HPRT1": 0.18642984163014426,
                "POLR2A": 0.36647764940764155, "POLR1B": 0.040193901737717995,
                "TBP": 0.060567363961918315,
            },
            "qc_markers": [
                "CD3E", "CD4", "CD8A", "MS4A1", "CD19", "PTPRC",
                "HBB", "CD14", "FCGR3A", "CD34", "NCAM1", "JCHAIN",
                "EPCAM", "KRT18", "COL1A1", "DCN", "PECAM1", "ERBB2",
            ],
            "engine_parameters": {"mu": 5483.714684902277, "phi": 0.7002926708915271},
        },
        "mm": {
            "noise_regex": "^mt-|^Rp[sl][0-9]+|^Gm[0-9]+|^Mir|^Rik$",
            "hk_profiles": {
                "Rpl19": 2.529822467672188, "Tubb5": 0.6653868555153559,
                "Eef1g": 1.409845725236883, "Ppia": 1.4689432838260164,
                "Gapdh": 1.9870339974714675, "Abcf1": 0.2683943094020617,
                "Sdha": 0.22436799098809676, "Oaz1": 1.5212192863178522,
                "G6pdx": 0.10922542882001043, "Alas1": 0.10135028595398016,
                "Gusb": 0.1690932556829679, "Hprt": 0.18642984163014426,
                "Polr2a": 0.36647764940764155, "Polr1b": 0.040193901737717995,
                "Tbp": 0.060567363961918315,
            },
            "qc_markers": [
                "Cd3e", "Cd4", "Cd8a", "Ms4a1", "Cd19", "Ptprc",
                "Hbb-bs", "Cd14", "Fcgr3", "Cd34", "Ncam1", "Jchain",
                "Epcam", "Krt18", "Col1a1", "Dcn", "Pecam1", "Erbb2",
            ],
            "engine_parameters": {"mu": 5483.714684902277, "phi": 0.7002926708915271},
        },
        "other": {
            "noise_regex": "",
            "hk_profiles": {},
            "qc_markers": [],
            "engine_parameters": {"mu": 5000.0, "phi": 0.7},
        },
    },
}


def write_default_config(path: str = "codeconv_config.json", overwrite: bool = False) -> str:
    """Write the built-in species configuration to a JSON file, for editing.

    Use this when you want to hand-tune a profile or add an organism: dump the
    defaults, edit the file, then pass its path as ``config_path``.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")
    with open(path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    print(f"Wrote built-in species configuration to {path}")
    return path


# Helpers

def _load_config(config_path: Optional[str], species: str) -> dict:
    """Load a species profile, falling back to the built-in configuration.

    An external JSON at ``config_path`` wins when it exists. Passing None, or a
    path that is not there, uses ``DEFAULT_CONFIG``, so a fresh install runs
    without downloading anything.
    """
    import copy

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    else:
        if config_path:
            print(f"   config '{config_path}' not found; using the built-in species profiles.")
        cfg = copy.deepcopy(DEFAULT_CONFIG)

    if species not in cfg["species_profiles"]:
        raise ValueError(
            f"Species '{species}' not in config. Available: {list(cfg['species_profiles'].keys())}"
        )
    profile = copy.deepcopy(cfg["species_profiles"][species])
    if not profile.get("hk_profiles"):
        raise ValueError(
            f"Species '{species}' has an empty hk_profiles block, so cell-density calibration "
            f"has nothing to calibrate against. Built-in profiles exist for 'hs' and 'mm'; for "
            f"any other organism, generate a species block with the CoexpressDeconvolve Config "
            f"Maker ('Estimate single cell parameters.ipynb'), save it with write_default_config() "
            f"as a starting point, and pass that file as config_path."
        )
    profile["min_topic_percentage"] = cfg.get("min_topic_percentage", 0.05)
    profile["species"] = species
    return profile


def _normalize_paths(spatial_path) -> Dict[str, str]:
    """Coerce a string / list / dict input into a {name: path} dict.

    Single string -> {basename: path}.
    List of strings -> {basename(p): p for p in list}; duplicate basenames are an error.
    Dict -> passthrough.
    """
    if isinstance(spatial_path, str):
        name = os.path.basename(spatial_path.rstrip('/')) or spatial_path
        return {name: spatial_path}
    if isinstance(spatial_path, dict):
        return dict(spatial_path)
    if isinstance(spatial_path, (list, tuple)):
        names = [os.path.basename(p.rstrip('/')) or p for p in spatial_path]
        if len(set(names)) != len(names):
            raise ValueError(
                "Duplicate slice basenames detected in spatial_path list; "
                "use the dict form to disambiguate: {name: path, ...}."
            )
        return dict(zip(names, spatial_path))
    raise TypeError(f"spatial_path must be str, list, or dict, got {type(spatial_path)}")


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


# STEP 1: load

@_profile_step
def step1_acquisition_and_anchoring(
    spatial_path=None,
    platform: str = "auto",
    visium_path=None,
) -> Dict[str, SliceData]:
    """Load spot-based spatial transcriptomics data per slice.

    Accepts a str, list, or dict of paths, exactly as before. The platform is
    detected from the directory layout and announced; pass ``platform=`` to force
    it. Everything downstream is platform-agnostic, because Step 1 normalizes all
    inputs to the same four things: counts, coordinates, spot footprint, pitch.

    Supported platforms (see :data:`PLATFORM_PROFILES` and :func:`detect_platform`):

    ==============  ==========================  ========================================
    ``visium``      55 um spots, 100 um pitch   SpaceRanger layout, native, unchanged
    ``dbit_seq``    10/25/50 um, 2x the width   grid-indexed table on a 50 x 50 array
    ==============  ==========================  ========================================

    Platforms whose capture units are already at or below the size of one cell -
    Slide-seq beads at 10 um, Visium HD 8 and 16 um bins, and the subcellular
    Stereo-seq bin1, Xenium and CosMx, are deliberately out of scope: a capture
    unit that holds at most one cell has nothing to deconvolve, and segmentation
    rather than a mixture model is the appropriate step there.

    Platforms without histology get a black placeholder image and synthesized
    scale factors, so the exported object still loads in Seurat
    (see :func:`make_dummy_tissue_image`).

    Parameters
    ----------
    spatial_path : str | list | dict
        Dataset directory (or a {name: path} dict for multi-slice runs).
    platform : str
        ``"auto"`` (default) or one of the keys of ``PLATFORM_PROFILES``. May also
        be a dict keyed by slice name for mixed-platform runs.
    visium_path
        Deprecated alias for ``spatial_path``, kept so existing notebooks keep
        working.

    Returns
    -------
    Dict[str, SliceData]
    """
    if visium_path is not None:
        if spatial_path is not None:
            raise TypeError(
                "step1_acquisition_and_anchoring: pass either spatial_path or the "
                "deprecated visium_path, not both."
            )
        warnings.warn(
            "visium_path is deprecated and will be removed in a future release; "
            "use spatial_path (the pipeline is no longer Visium-only).",
            DeprecationWarning, stacklevel=2,
        )
        spatial_path = visium_path
    if spatial_path is None:
        raise TypeError("step1_acquisition_and_anchoring: spatial_path is required.")

    start_time = time.perf_counter()
    paths = _normalize_paths(spatial_path)
    platforms = _broadcast(platform, list(paths.keys()), default="auto")
    out: Dict[str, SliceData] = {}

    for name, data_path in paths.items():
        plat = platforms[name]
        if plat == "auto":
            plat = detect_platform(data_path)
        if plat not in PLATFORM_PROFILES:
            raise ValueError(
                f"[{name}] unknown platform '{plat}'. "
                f"Known: {sorted(PLATFORM_PROFILES)}"
            )
        prof = PLATFORM_PROFILES[plat]
        print(f"\nStep 1 [{name}]: loading from {data_path}")
        print(
            f"   Detected platform: {prof['label']}, "
            f"{prof['spot_diameter_um']:g} um capture units, "
            f"{prof['pitch_um']:g} um centre-to-centre"
        )

        if plat != 'visium':
            out[name] = _load_generic_platform(name, data_path, plat)
            continue

        out[name] = _load_visium_slice(name, data_path)

    duration = time.perf_counter() - start_time
    print(f"\nStep 1 done in {duration:.2f}s   {len(out)} slice(s) loaded.")
    return out


def _load_visium_slice(name: str, data_path: str) -> SliceData:
    """Load one 10x Visium slice from the SpaceRanger layout."""
    if True:
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
        _show(f"step1_qc_umi_{name}")

        return SliceData(
            name=name,
            spatial_path=spatial_dir,
            counts=counts,
            gene_names=gene_names,
            barcodes=valid_barcodes,
            coords=coords,
            total_umi=total_counts_orig,
            scale_factors=scale_factors,
            platform='visium',
        )


def _load_generic_platform(name: str, data_path: str, plat: str) -> SliceData:
    """Load a grid-indexed spot-based platform into the same SliceData shape.

    Expects a single delimited counts table whose row (or column) labels encode
    the array coordinates as ``AxB`` / ``A_B``, which is the DBiT-seq layout.

    Coordinates are converted to microns using the platform pitch and then used
    as the working pixel space at 1 px = 1 um, so ``spot_diameter_fullres`` and
    everything derived from it stay meaningful. No histology exists for these
    platforms, so ``spatial_path`` is left pointing at a directory that Step 9
    will populate with a black placeholder.
    """
    prof = dict(PLATFORM_PROFILES[plat])

    if os.path.isdir(data_path):
        tables = _find_count_tables(data_path)
        if not tables:
            raise FileNotFoundError(f"[{name}] no counts table found in {data_path}")
        table_path = tables[0]
    else:
        table_path = data_path

    # DBiT-seq ships 10, 25 and 50 um variants; take the width from the sample
    # name when it carries one, and derive the pitch as twice the width.
    if plat == 'dbit_seq':
        px = _dbit_pixel_um(os.path.basename(table_path) or name, prof['spot_diameter_um'])
        prof['spot_diameter_um'], prof['pitch_um'] = px, 2.0 * px
        print(f"   microchannel width {px:g} um -> pitch {2 * px:g} um")

    sep = ',' if '.csv' in os.path.basename(table_path).lower() else '\t'
    print(f"   > counts table: {table_path}")
    mat, row_labels, col_labels = _read_counts_table(table_path, sep)

    # Orient to capture-units x genes: the capture-unit axis is the one whose
    # labels parse as coordinates, whichever way round the file was written.
    if _parse_coord_labels(row_labels[:50]) is None:
        mat, row_labels, col_labels = mat.T.tocsr(), col_labels, row_labels
    barcodes = row_labels
    xy = _parse_coord_labels(barcodes)
    if xy is None:
        raise ValueError(
            f"[{name}] labels of {table_path} are not coordinate-like "
            f"('AxB' or 'A_B') on either axis."
        )

    gene_names = col_labels
    counts = mat.tocsr()

    # Physical calibration. Grid indices scale by the known pitch directly;
    # otherwise recover the unit from the measured spacing (see PLATFORM_PROFILES).
    if prof['coord_units'] == 'grid':
        um_per_unit = float(prof['pitch_um'])
        array_idx = xy.copy()
    else:
        d_nn, _ = cKDTree(xy).query(xy, k=2)
        measured = float(np.median(d_nn[:, 1]))
        if not np.isfinite(measured) or measured <= 0:
            raise ValueError(f"[{name}] degenerate coordinates; cannot calibrate scale.")
        um_per_unit = float(prof['pitch_um']) / measured
        array_idx = xy / measured
        print(f"   calibration: median nearest-neighbour spacing {measured:.2f} units "
              f"= {prof['pitch_um']:g} um -> {um_per_unit:.4f} um/unit")
    xy_um = xy * um_per_unit
    # Anchor at the origin so the placeholder canvas fits tightly.
    xy_um = xy_um - xy_um.min(axis=0) + float(prof['spot_diameter_um'])

    coords = np.c_[xy_um[:, 0], xy_um[:, 1], array_idx[:, 0], array_idx[:, 1]]
    total_umi = np.asarray(counts.sum(axis=1)).flatten()

    # Working pixel scale is 1 px = 1 um for these platforms, so the footprint in
    # "pixels" is the footprint in microns and cellchat_spatial_factors recovers
    # the ratio back as 1.0 from the same pitch rule.
    scale_factors = {
        'spot_diameter_fullres': float(prof['spot_diameter_um']),
        'fiducial_diameter_fullres': float(prof['spot_diameter_um']) * 1.75,
        'tissue_hires_scalef': 1.0,
        'tissue_lowres_scalef': 1.0,
    }

    print(
        f"   {counts.shape[0]} capture units x {counts.shape[1]} genes   "
        f"median UMI/unit: {np.median(total_umi):.0f}   mean: {total_umi.mean():.0f}   "
        f"1st pct: {np.percentile(total_umi, 1):.0f}   "
        f"extent {np.ptp(xy_um[:, 0]):.0f} x {np.ptp(xy_um[:, 1]):.0f} um"
    )
    # Only flag images that could plausibly be histology; instrument output
    # (bead frames, base-call images) is not tissue and naming it would mislead.
    folder = data_path if os.path.isdir(data_path) else os.path.dirname(data_path)
    sibling = [
        f for f in glob.glob(os.path.join(folder, '*'))
        if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg')
        and not re.search(r'worker|beadimage|calls|thumb', os.path.basename(f), re.I)
    ]
    print(
        f"   WARNING: no registered tissue image; a black placeholder will be written "
        f"at export so the object still loads in Seurat."
    )
    if sibling:
        print(
            f"   (found {os.path.basename(sibling[0])} in the sample folder, but no "
            f"registration to the capture grid is provided, so it is not used.)"
        )

    plt.figure(figsize=(10, 6), dpi=100)
    lo = max(1.0, float(total_umi.min()))
    bins = np.logspace(np.log10(lo), np.log10(max(lo + 1, float(total_umi.max()))), 50)
    plt.hist(total_umi, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
    plt.axvline(np.median(total_umi), color='green', linestyle='--',
                label=f'Median: {int(np.median(total_umi))}')
    plt.axvline(np.percentile(total_umi, 1), color='purple', linestyle=':',
                label=f'1st pct: {int(np.percentile(total_umi, 1))}')
    plt.xscale('log')
    plt.title(f"[{name}] UMI per capture unit   N={counts.shape[0]}   {prof['label']}")
    plt.xlabel("Total UMI (log)")
    plt.ylabel("Capture units")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    _show(f"step1_qc_umi_{name}")

    return SliceData(
        name=name,
        spatial_path=os.path.join(data_path if os.path.isdir(data_path)
                                  else os.path.dirname(data_path), "spatial"),
        counts=counts,
        gene_names=gene_names,
        barcodes=barcodes,
        coords=coords,
        total_umi=total_umi,
        scale_factors=scale_factors,
        platform=plat,
    )


# STEP 2: density

@_profile_step
def step2_estimate_cell_density(
    slices: Dict[str, SliceData],
    config_path: Optional[str] = None,
    species: str = "hs",
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

        # anchor_blend_alpha is an internal, tuned constant and is deliberately
        # not reported: it is not a knob users should be reaching for.
        print(f"\nStep 2 [{name}]: Hybrid Calibration (factor={slice_anchor})")

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
        _show(f"step2_calibration_{name}")

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
        _show(f"step2_density_hist_{name}")

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
        _show(f"step2_spatial_density_{name}")

        print(f"   total cells on slide: {int(sum(n_cells))}")

        sd.n_cells = n_cells
        sd.engine_params = engine_params

    duration = time.perf_counter() - start_time
    print(f"\nStep 2 done in {duration:.2f}s")
    return slices


# STEP 3: ODG selection (joint, on intersected gene set)

@_profile_step
def step3_feature_selection(
    slices: Dict[str, SliceData],
    config_path: Optional[str] = None,
    species: str = "hs",
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
         the pool considered for overdispersed selection, the per-slice
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
        ``"any"`` (default) or ``"all"``, see above.
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
    # selection, sd.genes_clean and OdgPack.intersected_genes remain untouched.
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
            f"Presence filter left only {n_candidates} candidate genes, "
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
    plt.ylabel('Residual log10(variance): observed minus trend')
    plt.title(f'Step 3: overdispersion score (top {n_odg_eff} of {n_candidates} candidates; '
              f'{n_intersect} intersected genes)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _show("step3_overdispersion")

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

@_profile_step
def step4_gene_manifold(
    slices: Dict[str, SliceData],
    odg_pack: OdgPack,
    config_path: Optional[str] = None,
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
        print(f"Step 4: building gene manifold over {inter_concat.shape[1]} genes...")
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
    _show("step4_gene_manifold")

    duration = time.perf_counter() - start_time
    print(f"Step 4 done in {duration:.2f}s")
    return Manifold(
        embedding=embedding,
        intersected_genes=manifold_genes,
        odg_indices_in_intersected=odg_indices_in_intersected,
        species=species,
    )


# STEP 5: K sweep

@_profile_step
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
    k_tol: float = 0.01,
    k_band: float = 0.05,
    k_rule: str = "percentile",
    n_topics: Optional[int] = None,
) -> "KSweepResult":
    """Run an LDA K-sweep and visualize held-out perplexity alongside rare-topic count.

    For every K we fit LDA on a random 70/30 train/test split of the spots
    (split fixed across K so the K values are honestly comparable) and record:

      1. Held-out perplexity, evaluated on the test split. Lower is a better
         out-of-sample fit. The train/test split prevents the same-data
         optimism the in-sample perplexity would have.
      2. Rare-topic count: number of topics whose mean proportion across the
         train spots is below ``perc_rare_thresh`` (default 0.05). Rare topics
         indicate the model is over-splitting into spurious cell types, K is
         "too high" once rare topics appear.
      3. Mean Dirichlet alpha, estimated post-hoc from the fitted theta via
         Minka's fixed-point iteration. Used internally by the recommended-K
         rule as a soft constraint (alpha < 1 means each spot stays dominated
         by a few topics). Not plotted because sklearn's fixed doc_topic_prior
         keeps the post-hoc estimate well below 1 across most K, so the metric
         is rarely informative in this setting; the values remain available on
         the returned object for inspection.

    A K is recommended automatically, printed, and marked on the plot. The
    default rule is ``percentile`` at ``k_band=0.05``: take the largest K whose
    held-out perplexity is in the best 5% of the sweep by rank, the
    95th-percentile rule (see :func:`recommend_k`). This is the protocol's
    written guidance, highest K still in the low-perplexity regime, rare topics
    being desirable rather than penalized, expressed as a rule instead of a
    visual judgement call. Pass ``k_rule='relative_tolerance'`` for the
    sweep-extent-invariant variant, ``k_rule='largest_in_band'`` for the min-max
    band shaded on the plot, ``k_rule='conservative'`` for the module's original
    rule (lowest perplexity among K with alpha < 1 and no rare topics), or
    ``k_rule='lowest_perplexity'`` for plain argmin.

    The recommendation is a starting point, not a verdict: confirm it against
    the per-topic content in Step 6. To record a manual choice, pass it as
    ``n_topics=``, it is echoed on the plot next to the recommendation and
    returned as ``chosen_k``, so ``ksweep.k`` is always the K the run proceeds
    with.

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
    k_band : float
        Band width as a fraction, for the default ``percentile`` rule (and for
        ``largest_in_band`` and the shaded region on the plot). Default 0.05,
        the 95th-percentile rule. Raise it for a larger K.
    k_tol : float
        Relative tolerance on the best held-out perplexity, for the
        ``relative_tolerance`` rule. Default 0.01 (within 1% of the best fit).
    k_rule : str
        ``percentile`` (default) | ``relative_tolerance`` | ``largest_in_band`` |
        ``lowest_perplexity`` | ``conservative``.
    n_topics : Optional[int]
        A manual K, if you have already decided. Does not change the sweep; it is
        recorded as ``chosen_k``, marked on the plot alongside the automatic
        recommendation, and returned by ``ksweep.k``.

    Returns
    -------
    KSweepResult
        Dataclass with a compact ``__repr__`` so notebook auto-display stays one
        line. Fields: ``perplexity``, ``rare_topics``, ``alpha_mean``,
        ``alpha_per_topic``, ``perc_rare_thresh``, ``recommended_k``, ``k_band``,
        ``k_rule``, ``chosen_k``; ``.k`` gives the K to carry into Step 6 and
        ``.summary()`` returns the whole sweep as a DataFrame.
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
                n_jobs=-1,
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

    # Automatic K recommendation per target (see recommend_k for the rules).
    recommended_k: Dict[str, Dict[str, object]] = {}
    for label in sweep_perp.keys():
        recommended_k[label] = recommend_k(
            perplexity=dict(zip(ks, sweep_perp[label])),
            rule=k_rule,
            tol=k_tol,
            band=k_band,
            rare_topics=dict(zip(ks, sweep_rare[label])),
            alpha_mean=dict(zip(ks, sweep_alpha_mean[label])),
        )

    # Report it. The previous version computed this silently, which made the
    # choice of K look like an unaided visual judgement call.
    headline_label = '_joint' if '_joint' in recommended_k else next(iter(recommended_k))
    headline = recommended_k[headline_label]

    # Publish the choice as a module-level name as well as on the result object,
    # so Step 6 can read `codeconv.recommended_K` (or a bare `recommended_K`
    # after `from codeconv import *`) without threading the result through.
    global recommended_K
    recommended_K = int(n_topics) if n_topics is not None else int(headline['k'])

    print(f"\nStep 5: recommended K = {headline['k']}   [{headline['criterion']}]")
    if len(recommended_k) > 1:
        per_target = "   ".join(f"{lbl}: K={rec['k']}" for lbl, rec in recommended_k.items())
        print(f"   per target -> {per_target}")
    if headline['at_max_k']:
        print(
            f"   WARNING: the recommendation sits at max_k={ks[-1]}, so perplexity had not "
            f"started to rise again, the sweep is too narrow to bound K from above. "
            f"Re-run with a larger max_k before trusting this value (Troubleshooting problem 7)."
        )
    if n_topics is not None:
        print(f"   manual override: n_topics={int(n_topics)} will be used for Step 6.")

    # Plot 1: dual-axis line plot, perplexity (solid o) + rare topics (right, dashed s)
    from matplotlib.ticker import MaxNLocator
    fig, ax1 = plt.subplots(figsize=(11, 6.3))
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

    # Show the selection rule on the figure itself: the low-perplexity band as a
    # shaded region, the recommended K as a vertical line, a manual override (if
    # any) as a second line. Makes Figure 5 self-documenting.
    head_perp = np.array(sweep_perp[headline_label], dtype=float)
    p_lo, p_hi = float(head_perp.min()), float(head_perp.max())
    if p_hi > p_lo:
        if k_rule == "relative_tolerance":
            cutoff = p_lo * (1.0 + k_tol)
            band_label = f'within {k_tol:.1%} of best perplexity'
        elif k_rule == "percentile":
            cutoff = float(np.percentile(head_perp, k_band * 100.0))
            band_label = f'best {k_band:.0%} of sweep by rank'
        else:
            cutoff = p_lo + k_band * (p_hi - p_lo)
            band_label = f'low-perplexity band (≤{k_band:.0%} of range)'
        ax1.axhspan(p_lo, min(cutoff, p_hi), color='tab:green', alpha=0.08, zorder=0)
        ax1.axhline(cutoff, color='tab:green', linestyle=':', linewidth=1.2,
                    label=band_label)
    ax1.axvline(headline['k'], color='tab:red', linestyle='--', linewidth=1.5,
                label=f"recommended K = {headline['k']}")
    if n_topics is not None and int(n_topics) != int(headline['k']):
        ax1.axvline(int(n_topics), color='tab:blue', linestyle='-.', linewidth=1.5,
                    label=f"chosen K = {int(n_topics)}")

    # Compact legend: one entry per slice (color) + a linestyle key for metric.
    slice_handles = [plt.Line2D([0], [0], color=label_color[lbl], linewidth=2, label=lbl)
                     for lbl in sweep_perp.keys()]
    style_handles = [
        plt.Line2D([0], [0], color='black', linewidth=1.5,
                   marker='o', markerfacecolor='white', label='Perplexity'),
        plt.Line2D([0], [0], color='black', linewidth=1.5,
                   linestyle='--', marker='s', label='Rare topics'),
    ]
    rule_handles, _ = ax1.get_legend_handles_labels()
    all_handles = slice_handles + style_handles + rule_handles
    # Centred above the axes so it never lands on the curves or the shaded band.
    ax1.legend(handles=all_handles, loc='lower center',
               bbox_to_anchor=(0.5, 1.03), ncol=min(4, len(all_handles)),
               fontsize=8, frameon=False)

    title = (
        f"Step 5: K-sweep, perplexity and rare topics   (recommended K = {headline['k']})"
        + (", per slice + joint" if n_slices > 1 else "")
    )
    # Title above the legend, which itself sits above the axes.
    ax1.set_title(title, pad=52)
    plt.tight_layout()
    _show("step5_ksweep_perplexity_rare")

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
        _show("step5_relative_perplexity_heatmap")

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
            rare_guidance = "all K have rare topics, consider lowering K"
        plt.title(f"Step 5: rare topics by K  ({rare_guidance})")
        plt.xlabel("K")
        plt.ylabel("Slice")
        plt.tight_layout()
        _show("step5_rare_topics_heatmap")

    duration = time.perf_counter() - start_time
    print(f"\nStep 5 done in {duration:.2f}s")
    return KSweepResult(
        perplexity={label: dict(zip(ks, perps)) for label, perps in sweep_perp.items()},
        rare_topics={label: dict(zip(ks, rares)) for label, rares in sweep_rare.items()},
        alpha_mean={label: dict(zip(ks, alphas)) for label, alphas in sweep_alpha_mean.items()},
        alpha_per_topic={label: dict(zip(ks, sweep_alpha_full[label])) for label in sweep_alpha_full},
        perc_rare_thresh=perc_rare_thresh,
        recommended_k=recommended_k,
        k_tol=k_tol,
        k_band=k_band,
        k_rule=k_rule,
        chosen_k=int(n_topics) if n_topics is not None else None,
    )


# STEP 6: per-slice LDA -> Hungarian alignment -> consensus beta -> per-slice theta refit + projection

@_profile_step
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

    A per-slice topic stability score, the mean pairwise cosine similarity of
    aligned topic rows across seeds, is computed and stored on the returned
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
                n_jobs=-1,
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
                    n_jobs=-1,
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
        # Multi-slice projection: per-slice with cosine fallback for slice-specific genes
        print("Step 6: projecting topics to full genes_clean per slice")
        inter_index = {g: i for i, g in enumerate(manifold.intersected_genes)}
        odg_idx_in_intersected = manifold.odg_indices_in_intersected
        odg_coords_manifold = manifold.embedding[odg_idx_in_intersected]

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
                    gi = inter_index[g]
                    g_pos = manifold.embedding[gi]
                    dists = np.linalg.norm(odg_coords_manifold - g_pos, axis=1)
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

        print("Step 6: projecting latent topics using UMAP manifold topology")
        genes_s = sd.genes_clean
        n_genes_s = len(genes_s)

        # Manifold positions: for a single slice, every gene in genes_clean is in
        # the intersected set (intersection of one set is itself), so all genes have
        # manifold coordinates.
        inter_index = {g: i for i, g in enumerate(manifold.intersected_genes)}
        odg_coords_manifold = manifold.embedding[manifold.odg_indices_in_intersected]
        all_coords = manifold.embedding[[inter_index[g] for g in genes_s]]
        dist_matrix = cdist(all_coords, odg_coords_manifold, metric='euclidean')

        # Cosine similarity in expression space
        slice_gene_to_idx = {g: i for i, g in enumerate(genes_s)}
        odg_idx_in_slice = [slice_gene_to_idx[g] for g in odg_pack.odg_names]
        norm_all = normalize(sd.counts_clean.T, axis=1)
        norm_odg = normalize(sd.counts_clean[:, odg_idx_in_slice].T, axis=1)
        similarity = norm_all @ norm_odg.T

        beta_final = np.zeros((n_topics, n_genes_s))
        for i in range(n_genes_s):
            umap_neighbors_idx = np.argsort(dist_matrix[i])[:k_neighbors]
            sim_row = similarity[i].toarray().flatten() if sp.issparse(similarity) else similarity[i]
            weights = sim_row[umap_neighbors_idx]
            if np.max(weights) < min_sim:
                continue
            proj = beta_odg[:, umap_neighbors_idx] @ weights
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

@_profile_step
def step7_sampling_engine(
    slices: Dict[str, SliceData],
    model: Model,
    config_path: Optional[str] = None,
    species: str = "hs",
    low_slice_quality=False,
    min_topic_percentage: Optional[float] = None,
) -> Dict[str, dict]:
    """Per-slice hierarchical Bayesian sampling. Returns dict of per-slice cell outputs.

    Threshold-and-renormalize is always applied: any topic with theta < min_topic_percentage
    is zeroed and theta is renormalized. low_slice_quality=True per slice additionally
    enforces an inflate rescue: every surviving topic gets at least 1 cell.

    The inner gene-and-cell allocation is vectorized via a sequential-binomial
    batched multinomial sampler, so the draws are statistically equivalent to
    the legacy per-gene-per-cell loop but the RNG draw order differs: outputs
    will not be bit-for-bit identical to the legacy implementation under the
    same seed. The marginals (per-spot UMI conservation, per-cell topic
    assignment, gamma cell weights) are preserved.
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

            # Pull the spot's nonzero (gene, count) entries directly from CSR -
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
                # block, no Python-side cell loop.
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

# STEP 8: placement (per-slice)

@_profile_step
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
            # Vogel sunflower confined to the spot footprint: the max radius is the
            # spot radius (diameter / 2), NOT the diameter, otherwise cells reach a
            # full diameter out and overlap neighbouring spots. (idx + 0.5) / n gives
            # an area-uniform fill of the disk.
            r = 0.5 * diameter * np.sqrt((idx + 0.5) / n_total) if n_total > 1 else 0.0
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

@_profile_step
def step9_export_results(
    cells: Dict[str, dict],
    slices: Dict[str, SliceData],
    output_folder: str,
    spot_pitch_um: float = 100.0,
    interaction_range_um: float = 50.0,
):
    """Per-slice export to slice_<name>/deconvolved/ in 10X SpaceRanger HDF5 layout.

    For a single slice run, the slice_<name> wrapper is still applied so behavior is
    consistent across single and multi-slice modes.

    The per-slice ``run_summary.json`` carries a ``cellchat`` block with everything
    the downstream R session needs, ``ratio``, ``tol``, ``interaction_range``,
    ``contact_range``, so the CellChat step reads one file instead of
    re-deriving unit conversions by hand::

        cc <- jsonlite::fromJSON("slice_<name>/run_summary.json")$cellchat
        spatial.factors <- data.frame(ratio = cc$ratio, tol = cc$tol)

    Parameters
    ----------
    spot_pitch_um : float
        Centre-to-centre spot distance in microns, used to calibrate the
        pixel-to-micron ratio. 100.0 for all 10x Visium capture areas;
        pass the array pitch for other platforms.
    interaction_range_um : float
        The CellChat ``interaction.range`` to record and to run the sub-spot
        diagnostics against. Default 50.0, the shortest range that still covers
        the whole ~60 um Visium spot footprint, so every intra-spot pair is
        counted and the inference stays invariant to the Step 8 packing.
        See :func:`cellchat_spatial_factors`.
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

        # No registered histology (DBiT-seq samples vary): write a black
        # placeholder so Load10X_Spatial and every spatial plot still work.
        if not any(os.path.exists(os.path.join(spa_dir, f))
                   for f in ('tissue_hires_image.png', 'tissue_lowres_image.png')):
            dummy_sf = make_dummy_tissue_image(
                coords, spa_dir,
                spot_diameter_fullres=sd.scale_factors.get('spot_diameter_fullres'),
            )
            merged = dict(sd.scale_factors or {})
            merged.update(dummy_sf)
            with open(os.path.join(spa_dir, 'scalefactors_json.json'), 'w') as fsf:
                json.dump(merged, fsf, indent=2)
        pos_df = pd.DataFrame(coords, columns=['pxl_row', 'pxl_col', 'array_row', 'array_col'])
        pos_df['barcode'] = barcodes
        pos_df['in_tissue'] = 1
        pos_df = pos_df[['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']]
        pos_df.to_csv(os.path.join(spa_dir, "tissue_positions.csv"), index=False)

        # Per-slice summary. The CellChat block is recorded here so the downstream
        # R session can read spatial.factors straight out of run_summary.json
        # instead of guessing them (see cellchat_spatial_factors).
        slice_summary = {
            'slice_name': name,
            'n_spots': int(sd.counts_clean.shape[0]),
            'n_cells_generated': int(payload['n_cells']),
            'n_topics': int(sd.theta.shape[1]) if sd.theta is not None else None,
            'n_rescued_topic_slots': int(payload.get('n_rescued', 0)),
            'gene_count_export': int(len(genes)),
        }
        try:
            cc = cellchat_spatial_factors(
                {name: payload}, {name: sd},
                spot_pitch_um=spot_pitch_um,
                interaction_range_um=interaction_range_um,
            )[name]
            slice_summary['cellchat'] = {
                'ratio': round(cc['ratio'], 6),
                'tol': round(cc['tol'], 4),
                'interaction_range': round(float(interaction_range_um), 2),
                'contact_range': round(float(interaction_range_um), 2),
                'spot_pitch_um': float(spot_pitch_um),
                'spot_pitch_px': round(cc['spot_pitch_px'], 3),
                'spot_diameter_um': round(cc['spot_diameter_um'], 3),
                'min_intercell_distance_um': round(cc['min_distance_um'], 4),
                'median_nn_distance_um': round(cc['median_nn_distance_um'], 4),
                'mean_neighbours': round(cc.get('mean_neighbours', float('nan')), 3),
                'same_spot_fraction': round(cc.get('same_spot_fraction', float('nan')), 4),
                'intra_spot_pairs_captured': round(cc.get('intra_spot_pairs_captured', float('nan')), 4),
            }
        except Exception as exc:  # never let a diagnostic break the export
            print(f"   [{name}] could not derive CellChat spatial factors: {exc}")
        with open(os.path.join(slice_dir, "run_summary.json"), 'w') as fout:
            json.dump(slice_summary, fout, indent=2)
        summary[name] = slice_summary
        print(f"   [{name}] cells={slice_summary['n_cells_generated']}   genes={slice_summary['gene_count_export']}")

    # Top-level summary across all slices
    with open(os.path.join(output_folder, "run_summary.json"), 'w') as fout:
        json.dump({'slices': summary, 'n_slices': len(summary)}, fout, indent=2)

    duration = time.perf_counter() - start_time
    print(f"\nStep 9 done in {duration:.2f}s   exports ready for Seurat::Load10X_Spatial()")
