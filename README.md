CoexpressDeconvolve
Reference-free reconstruction of cell type-specific gene expression profiles from spot-based spatial transcriptomics.

CoexpressDeconvolve is a computational framework designed to resolve the "mixture problem" inherent in spot-based spatial transcriptomics (such as 10x Genomics Visium). Unlike traditional deconvolution methods that only provide cell-type proportions, this tool reconstructs high-fidelity, whole-transcriptome gene expression profiles for individual cell types directly from mixed spatial signals—without requiring an external single-cell RNA-seq reference.

Key Features
Reference-Free: Operates without the need for matched scRNA-seq atlases.

Expression Reconstruction: Recovers the underlying transcriptional identity of individual cells within a spot.

Absolute Abundance: Uses housekeeping gene calibration to estimate actual cell counts per spot.

Standard Output: Generates feature-barcode matrices (H5 format) compatible with downstream tools like Seurat, Scanpy, and CellChat.

Getting Started
1. Requirements
To run the pipeline, ensure you have the following files in your working directory:

codeconv.py: The core library containing the sampling engine and manifold projection logic.

codeconv_config.json: The configuration file containing universal single-cell parameters and housekeeping gene standards.

Your Data: Standard 10x SpaceRanger output (specifically the filtered_feature_bc_matrix.h5 and the spatial/ folder).

2. Installation
Install the necessary Python dependencies:

Bash
pip install h5py tqdm numpy pandas scipy scikit-learn umap-learn
3. Usage (Notebooks)
The easiest way to use the tool is by running the provided Jupyter Notebooks. Choose the version corresponding to your species:

For Human data: Open and run CoexpressDeconvolve.ipynb

For Mouse data: Open and run CoexpressDeconvolve_Mm.ipynb

The notebooks guide you through the 9-step pipeline:

Data Acquisition: Loading H5 and spatial metadata.

Density Estimation: Calculating cell counts via housekeeping genes.

Feature Selection: Filtering noise and identifying Highly Variable Genes (HVGs).

Manifold Construction: Building the gene co-expression topology.

K-Sweep: Optimizing the number of latent topics.

Deconvolution: Training the model and projecting topics to the whole transcriptome.

Sampling Engine: Generating discrete in silico single cells.

Geometry: Placing reconstructed cells within the physical spot boundaries.

Export: Saving results in a 10x-compatible format.

Pipeline Overview
Important Note
Do not forget to keep codeconv.py and codeconv_config.json in the same folder as your notebook. These files are essential for the sampling engine to calibrate expression levels correctly.

Downstream Analysis
The output folder (typically /deconvolved) will contain a new filtered_feature_bc_matrix.h5. You can load this directly into Seurat using Load10X_Spatial() to perform clustering, trajectory inference, or cell-cell communication analysis as if you had single-cell resolution.
