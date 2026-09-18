# CoexpressDeconvolve

[![PyPI](https://img.shields.io/pypi/v/codeconv.svg)](https://pypi.org/project/codeconv/)
[![Python](https://img.shields.io/pypi/pyversions/codeconv.svg)](https://pypi.org/project/codeconv/)
[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21860849-blue)](https://doi.org/10.5281/zenodo.21860849)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Spot-based spatial transcriptomics measures whole transcriptomes in place, but every spot pools
several cells, so a spot profile is a mixture rather than a cell. CoexpressDeconvolve resolves that
mixture into single-cell-like profiles, without needing a matched single-cell dataset to learn from.

The problem it addresses is that the usual route out of the mixture is a reference atlas, and a
reference is often unavailable, mismatched to the tissue, or missing the very populations that were
lost during dissociation. CoexpressDeconvolve works from the slide alone.

Three quantities carry the method. How many cells a spot captured is estimated from the
housekeeping-gene signal together with the total UMI count. Which expression programs are present is
learned de novo from gene co-expression across the whole slide. How the spot's transcripts belong to
those cells follows from distributing its UMIs among them, conserving the per-spot total exactly: the
reconstructed cells of a spot add back up to the transcriptome measured there.

Each cell comes out with a whole transcriptome, absolute counts and a coordinate inside its parent
spot, written in the standard 10x SpaceRanger layout, so the result opens in Seurat or scanpy like
any other single-cell object.

Method and benchmarks: Perik-Zavodskaia, O., Perik-Zavodskii, R., Alrhmoun, S. & Sennikov, S.
*iScience* **29**, 116824 (2026). https://doi.org/10.1016/j.isci.2026.116824

## Supported platforms

The platform is detected from the input layout and announced at load. Everything downstream is
platform-agnostic: Step 1 normalizes every input to counts, coordinates, a capture footprint and a
pitch.

| Platform | Capture unit | Pitch | Layout expected | Histology |
|---|---|---|---|---|
| **10x Visium** (any capture area) | 55 µm | 100 µm | SpaceRanger: `filtered_feature_bc_matrix.h5` (or the `.mtx` folder) + `spatial/` | yes |
| **DBiT-seq** | 10 / 25 / 50 µm | 2 × channel width | one count table on the 50 × 50 grid, labels `AxB` | no |

The DBiT-seq channel width is read from the sample name when it carries one (`GSM4189611_50t` -> 50 µm),
and the pitch is twice the width. Platforms that ship no tissue image get a black placeholder canvas
sized to the capture grid, with synthesized scale factors, so `Seurat::Load10X_Spatial()` and every
spatial plot still work.

Detection is by layout rather than by file name, so an unusual folder can be read the wrong way. Pass
`platform="visium"` or `platform="dbit_seq"` to `step1_acquisition_and_anchoring` to settle it, or
call `codeconv.detect_platform(path)` on its own to see what the layout looks like from here.

Capture units at or below the size of one cell are out of scope: there is nothing to deconvolve, and
segmentation rather than a mixture model is the right step there.

## Installation

```bash
pip install codeconv
```

Python 3.9 or newer. Dependency versions are pinned by the package, so there is nothing else to
manage. Human (`hs`) and mouse (`mm`) species profiles are built into the module, so a fresh install
runs with no further downloads.

## Usage - The 9 steps

1. **Acquisition** - detect the platform, load counts and coordinates, per-spot UMI QC.
2. **Density** - cells per spot from a hybrid housekeeping/UMI calibration. `low_slice_quality=True`
   enforces a floor of one cell on every spot passing the UMI gate.
3. **Feature selection** - noise-gene regex, presence filter, overdispersed genes from the
   mean-variance trend.
4. **Manifold** - joint gene co-expression topology via ICA + UMAP.
5. **K-sweep** - held-out perplexity across K, rare-topic counts, automatic recommendation.
6. **Deconvolution** - per-slice LDA, Hungarian topic alignment across slices, mean-consensus beta,
   per-slice theta refit against the frozen consensus, projection onto each slice's full gene list.
7. **Sampling** - discrete cells drawn per spot; per-spot UMI totals are conserved exactly. Topics
   below `min_topic_percentage` are zeroed and renormalized; `low_slice_quality=True` rescues every
   surviving topic with at least one cell, to keep rare programs in weak spots.
8. **Placement** - cells positioned inside their parent spot footprint by Vogel packing.
9. **Export** - 10x layout under `output_folder/slice_<name>/deconvolved/`, one folder per slice.

`CoexpressDeconvolve.ipynb` in this repository runs the whole pipeline on a Visium slide and is the
quickest way in. The same thing in plain Python:

```python
import codeconv
codeconv.set_seed(42)

spatial_path  = "./Glioblastoma"
output_folder = "."
species       = "hs"                      # "hs" | "mm" | anything in your config

slices   = codeconv.step1_acquisition_and_anchoring(spatial_path)
slices   = codeconv.step2_estimate_cell_density(slices, species=species,
                                                min_umi=900, anchor_mean_factor=0.7)
odg_pack = codeconv.step3_feature_selection(slices, species=species, n_odg=3000)
manifold = codeconv.step4_gene_manifold(slices, odg_pack, n_components=30)

ksweep   = codeconv.step5_ksweep(odg_pack, min_k=3, max_k=15)
print(ksweep.summary())  # per-K perplexity, rare topics, which K are in the low band

model = codeconv.step6_final_deconvolution(slices, odg_pack, manifold,
                                           n_topics=codeconv.recommended_K,  # or your own K
                                           k_neighbors=3)
cells = codeconv.step7_sampling_engine(slices, model, species=species)
codeconv.step8_geometry_and_placement(cells, slices)
codeconv.step9_export_results(cells, slices, output_folder, interaction_range_um=50.0)
```

Each step reports its own timing and peak process RAM as it runs, and writes its diagnostic figure to
`./figures` at 600 dpi. `codeconv.set_figure_output(directory, dpi)` changes both.

### Several slices at once

Pass a dict instead of a string. Topics are aligned across slices, so the reconstructed cell types
are comparable between samples:

```python
spatial_path = {"sample_A": "./A", "sample_B": "./B"}
```

Per-slice parameters (`min_umi`, `anchor_mean_factor`, `low_slice_quality`) take either a scalar,
broadcast to every slice, or a dict keyed by slice name:

```python
slices = codeconv.step2_estimate_cell_density(
    slices, species="hs",
    min_umi={"sample_A": 900, "sample_B": 1200},
    anchor_mean_factor=0.7,
)
```

A list of paths also works, with slice names taken from the folder names.

### Choosing K automatically

The number of topics is chosen by an explicit rule rather than by reading the curve:

```python
ksweep = codeconv.step5_ksweep(odg_pack, min_k=3, max_k=15)   # prints the recommendation
model  = codeconv.step6_final_deconvolution(slices, odg_pack, manifold,
                                            n_topics=codeconv.recommended_K)
```

The default rule keeps every K whose held-out perplexity lands in the best 5% of the sweep by rank
and returns the **largest** of them, which is the highest K still in the low-perplexity regime. That
is deliberate: because CoexpressDeconvolve returns individual cells rather than fractions, a rare
topic is a minor cell population worth recovering, not evidence of over-splitting. Alternatives:

```python
codeconv.recommend_k(ksweep.perplexity['Glioblastoma'], rule="relative_tolerance", tol=0.01)
codeconv.recommend_k(ksweep.perplexity['Glioblastoma'], rule="lowest_perplexity")
```

A recommendation landing on `max_k` means perplexity had not turned back up yet and the sweep is too
narrow; you get a warning. Widen it before accepting the value. Pass `n_topics=` to record a manual
choice, and it is drawn on the plot next to the automatic one.

### Species other than human and mouse

The density calibration and the sampling engine need a species profile: housekeeping genes, a noise
pattern and negative-binomial engine parameters. Dump the built-in ones, edit, and point the run at
the file:

```python
codeconv.write_default_config("codeconv_config.json")   # then edit the new species block

cfg = "codeconv_config.json"
slices   = codeconv.step2_estimate_cell_density(slices, config_path=cfg, species="dr")
odg_pack = codeconv.step3_feature_selection(slices, config_path=cfg, species="dr")
```

`config_path` is accepted by steps 2, 3, 4 and 7, the four that read the species block. An external
file always wins over the built-in profiles.

To derive `hk_profiles` and the engine parameters from data rather than by hand, the CoexpressDeconvolve
Config Maker (`Estimate single cell parameters.ipynb`) computes them from a folder of 10x scRNA-seq
datasets for the target organism. Add organism-specific `qc_markers` and `noise_regex` to the same
block so the gene manifold is annotated.

## Downstream analysis

Each slice exports a `filtered_feature_bc_matrix.h5` and a `spatial/` folder, alongside a
`run_summary.json` recording the parameters and geometry of the run. From there the reconstruction is
an ordinary single-cell spatial object and goes wherever such objects go:

```r
library(Seurat)
seurat_obj <- Load10X_Spatial("./slice_Glioblastoma/deconvolved")
```

```python
import scanpy as sc
adata = sc.read_visium("./slice_Glioblastoma/deconvolved")
```

Clustering, annotation, trajectory inference, cell-cell communication - anything that takes a
single-cell object works, with no adapters. `Seurat Spatial.ipynb` in this repository is a worked
example. Topics are aligned across slices, so cluster comparison between samples is meaningful.

At tens of thousands of reconstructed cells the downstream analysis, not the deconvolution, is the
memory-limiting stage. On the Seurat path, store counts on disk with BPCells and scale variable
features only.

<img width="6944" height="10416" alt="Figure 5" src="https://github.com/user-attachments/assets/9211382e-002c-441c-91a2-971d9f207b48" />

(**a**) UMAP projection of Visium spot-level transcriptomes before deconvolution. (**b**) Spatial mapping of these clusters onto the histological section. (**c**) Dot plot of representative marker genes. (**d**) UMAP projection following computational deconvolution. (**e**) Spatial distribution of deconvolved cells. (**f**) Dot plot of marker genes of the deconvolved cell populations.

## Citation

```bibtex
@article{PerikZavodskaia2026CoexpressDeconvolve,
  title   = {CoexpressDeconvolve enables reference-free single-cell-resolution
             deconvolution from spot-based spatial transcriptomics},
  author  = {Perik-Zavodskaia, Olga and Perik-Zavodskii, Roman and
             Alrhmoun, Saleh and Sennikov, Sergey},
  journal = {iScience},
  volume  = {29},
  pages   = {116824},
  year    = {2026},
  doi     = {10.1016/j.isci.2026.116824}
}
```

## License

MIT, see `LICENSE`.
