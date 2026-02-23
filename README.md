# CoexpressDeconvolve

CoexpressDeconvolve is a computational framework designed to resolve the multiple cell per spot problem inherent in spot-based spatial transcriptomics (such as 10x Genomics Visium). Unlike traditional deconvolution methods that only provide cell-type proportions, this tool reconstructs high-fidelity, whole-transcriptome gene expression profiles for individual cell types directly from mixed spatial signals without requiring an external single-cell RNA-seq reference.

## Key Features

Reference-Free: Operates without the need for matched scRNA-seq atlases.

Expression Reconstruction: Recovers the underlying transcriptional identity of cells within a spot.

Standard Output: Generates feature-barcode matrices (.h5 format) compatible with downstream tools like Seurat.

## Getting Started

1. Requirements

To run the pipeline, ensure you have the following files in your working directory:

`codeconv.py`: The core library.

`codeconv_config.json`: The configuration file containing universal single-cell parameters and housekeeping gene standards.

`filtered_feature_bc_matrix.hp5` and `spatial` folder: Your Data.

2. Installation

Install the necessary Python dependencies:

`pip install h5py tqdm numpy pandas scipy scikit-learn umap-learn`

## Usage 

Run the tool by using the provided Jupyter Notebooks. Choose the version corresponding to your species:

For Human data: Open and run CoexpressDeconvolve Hs.ipynb

For Mouse data: Open and run CoexpressDeconvolve Mm.ipynb

The notebooks guide you through the 9-step pipeline:

1. Data Acquisition: Loading H5 and spatial metadata.

2. Density Estimation: Calculating cell counts via housekeeping genes.

3. Feature Selection: Filtering noise and identifying Highly Variable Genes (HVGs).

4. Manifold Construction: Building the spatial gene co-expression topology.

5. K-Sweep: Optimizing the number of latent topics.

6. Deconvolution: Training the model and projecting topics to the whole transcriptome.

7. Sampling: Generating single-cell-like expression profiles.

8. Spatial: Placing reconstructed single-cell-like expression profiles within their physical spot boundaries.

9. Export: Saving results in a 10x-compatible format.

## Downstream Analysis

The output folder will contain a new filtered_feature_bc_matrix.h5 and "spatial" folder. You can load this directly into Seurat using Load10X_Spatial(), e.g. via the "Seurat Spatial" notebook to perform clustering or cell-cell communication analysis as if you had single-cell resolution.

![Figure 6 copy](https://github.com/user-attachments/assets/7dab151f-4c59-408b-88a0-689b108b5e95)
