# CoexpressDeconvolve

CoexpressDeconvolve is a computational framework designed to resolve the multiple cell per spot problem inherent in spot-based spatial transcriptomics (such as 10x Genomics Visium). Unlike other deconvolution methods that focus on providing cell-type proportions, this tool focuses on the reconstruction of high-fidelity, whole-transcriptome gene expression profiles for individual cell types directly from mixed spatial signals without requiring an external single-cell RNA-seq reference.

The pipeline supports multi-slice runs out of the box: pass a single Visium directory to deconvolve one slice, or pass a dict of named directories to jointly deconvolve several slices with shared topics, comparable across slices in downstream analysis.

## Getting Started

### 1. Requirements

To run the pipeline, ensure you have the following files in your working directory:

- `codeconv.py`: The core library.
- `codeconv_config.json`: The configuration file containing universal single-cell parameters, housekeeping gene standards, and species profiles for human (`hs`), mouse (`mm`), and a user-extensible `other` block.
- `filtered_feature_bc_matrix.h5` and `spatial` folder: your Visium data, one set per slice.

### 2. Installation

Install the necessary Python dependencies:

```
pip install h5py tqdm numpy pandas scipy scikit-learn umap-learn matplotlib seaborn
```

## Usage

Run the tool using the provided Jupyter notebook `CoexpressDeconvolve.ipynb`. Species selection now lives in the config block at the top of the notebook, so a single notebook handles human, mouse, and other organisms:

```python
visium_path   = "."                          # single slice
# or, multi-slice:
# visium_path = {"sample_A": "./path/A", "sample_B": "./path/B"}
output_folder = "."
config_path   = "codeconv_config.json"
species       = "hs"                         # "hs" | "mm" | "other"
```

Per-slice parameters (`min_umi`, `anchor_mean_factor`, `low_slice_quality`) accept either a scalar (broadcast to all slices) or a dict keyed by slice name. Universal parameters (`n_hvg`, `n_components`, `SELECTED_K`) are shared across slices.

For non-standard organisms, set `species = "other"` and populate the `other` block in `codeconv_config.json` with your own housekeeping gene reference and engine parameters.

The notebook guides you through the 9-step pipeline:

1. **Data Acquisition** — loading H5/MTX matrices and spatial metadata, per slice.
2. **Density Estimation** — calculating cell counts via housekeeping gene calibration. Set `low_slice_quality=True` per slice to enforce a floor of one cell on every spot that passes the UMI gate.
3. **Feature Selection** — filtering noise genes and identifying Highly Variable Genes on the intersected gene set across slices.
4. **Manifold Construction** — building the joint spatial gene co-expression topology via ICA + UMAP.
5. **K-Sweep** — optimizing the number of latent topics. Diagnostic plots show per-slice and joint perplexity sweeps as both an overlaid line plot and a relative-perplexity heatmap.
6. **Deconvolution** — per-slice LDA, Hungarian topic alignment across slices, mean-consensus β, per-slice θ refit via variational E-step with frozen consensus β, and per-slice projection of consensus topics onto each slice's full gene list.
7. **Sampling** — generating single-cell-like expression profiles. Topics with `θ < min_topic_percentage` are zeroed and renormalized; setting `low_slice_quality=True` additionally rescues every surviving topic with at least one cell to preserve rare programs in low-quality spots.
8. **Spatial Placement** — placing reconstructed single-cell-like profiles within their physical spot boundaries via Fibonacci shells.
9. **Export** — saving results in a 10x-compatible format under `output_folder/slice_<name>/deconvolved/`, one folder per slice.

## Downstream Analysis

Each slice's output folder contains a `filtered_feature_bc_matrix.h5` and a `spatial` folder. You can load each slice directly into Seurat using `Load10X_Spatial()`, e.g. via the `Seurat Spatial.ipynb` notebook to perform clustering or cell-cell communication analysis as if you had single-cell resolution. Topic identities are aligned across slices, so cluster comparison across samples is meaningful.

![Figure 6 copy](https://github.com/user-attachments/assets/7dab151f-4c59-408b-88a0-689b108b5e95)
(**a**) UMAP projection of Visium spot-level transcriptomes before deconvolution. (**b**) Spatial mapping of these clusters onto the histological section. (**c**) Dot plot of representative marker genes. (**d**) UMAP projection following computational deconvolution. (**e**) Spatial distribution of deconvolved cells. (**f**) Dot plot of marker genes of the deconvolved cell populations.
