import os
import re
import json
import gzip
import shutil
import time
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.decomposition import FastICA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import h5py

try:
    import umap
except ImportError:
    raise ImportError("Step 4 requires 'umap-learn'. Install via: pip install umap-learn")

def step1_acquisition_and_anchoring(data_path):
    """
    Step 1: 
    Loads 10X Visium data (supports both .h5 and .mtx formats) 
    And visualizes the UMI distribution.
    """
    start_time = time.perf_counter()
    spatial_dir = os.path.join(data_path, "spatial")
    
    h5_file = os.path.join(data_path, "filtered_feature_bc_matrix.h5")
    matrix_dir = os.path.join(data_path, "filtered_feature_bc_matrix")

    # 1. Expression Matrix Acquisition
    print(f"Step 1: Checking data format in {data_path}...")

    if os.path.exists(h5_file):
        print(f"   > Found H5 file: {h5_file}. Loading...")
        
        with h5py.File(h5_file, 'r') as f:
            mat_group = f['matrix'] if 'matrix' in f else f
            data = mat_group['data'][:]
            indices = mat_group['indices'][:]
            indptr = mat_group['indptr'][:]
            shape = mat_group['shape'][:]
            counts = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()

            if 'features' in mat_group:
                feat_group = mat_group['features']
                # Берем 'name' (символы генов), если нужно ENSG - поменяйте на 'id'
                gene_names = [x.decode('utf-8') for x in feat_group['name'][:]]
            else:
                # Старый формат 10x (v2)
                gene_names = [x.decode('utf-8') for x in mat_group['genes'][:]]

            raw_barcodes = [x.decode('utf-8') for x in mat_group['barcodes'][:]]
            
    elif os.path.exists(matrix_dir):
        print(f"   > Found Matrix directory: {matrix_dir}. Loading MTX...")
        
        counts = scipy.io.mmread(os.path.join(matrix_dir, "matrix.mtx.gz")).T.tocsr()
        features = pd.read_csv(os.path.join(matrix_dir, "features.tsv.gz"), header=None, sep='\t')
        gene_names = features[1].values.tolist()
        raw_barcodes = pd.read_csv(os.path.join(matrix_dir, "barcodes.tsv.gz"), header=None, sep='\t')[0].values.tolist()
        
    else:
        raise FileNotFoundError(f"Could not find 'filtered_feature_bc_matrix.h5' OR 'filtered_feature_bc_matrix' dir in {data_path}")

    # 2. Spatial Manifest Parsing
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
            'pxl_col_in_fullres': 'pxl_col'
        })
    
    spatial_df = spatial_df.set_index('barcode')
    
    with open(os.path.join(spatial_dir, "scalefactors_json.json"), 'r') as f:
        scale_factors = json.load(f)

    # 3. Synchronization & Filtering
    if not any(b in spatial_df.index for b in raw_barcodes[:10]):
        print("   ! Barcode mismatch detected (checking suffixes)...")
        if raw_barcodes[0].endswith("-1") and not spatial_df.index[0].endswith("-1"):
            print("   > Trimming '-1' from matrix barcodes.")
            raw_barcodes = [b.split('-')[0] for b in raw_barcodes]
        elif not raw_barcodes[0].endswith("-1") and spatial_df.index[0].endswith("-1"):
            print("   > Adding '-1' to matrix barcodes.")
            raw_barcodes = [b + "-1" for b in raw_barcodes]

    valid_barcodes = [b for b in raw_barcodes if b in spatial_df.index]
    valid_indices = [raw_barcodes.index(b) for b in valid_barcodes]
    
    if len(valid_barcodes) == 0:
        raise ValueError("No common barcodes found between Matrix and Spatial data!")

    counts = counts[valid_indices, :]
    coords = spatial_df.loc[valid_barcodes, ['pxl_row', 'pxl_col', 'array_row', 'array_col']].values
    
    # 4. Library Depth Calculation
    total_counts_orig = np.array(counts.sum(axis=1)).flatten()
    
    # QC VISUALIZATION
    print("\nLIBRARY DEPTH QC")
    mean_umi = np.mean(total_counts_orig)
    median_umi = np.median(total_counts_orig)
    p1 = np.percentile(total_counts_orig, 1) 
    
    print(f"Median UMI per Spot: {median_umi:.0f}")
    print(f"Mean UMI per Spot:   {mean_umi:.0f}")
    print(f"1th Percentile:      {p1:.0f}")
    
    plt.figure(figsize=(10, 6), dpi=100)
    
    bins = np.logspace(np.log10(max(1, total_counts_orig.min())), np.log10(total_counts_orig.max()), 50)
    
    plt.hist(total_counts_orig, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
    
    plt.axvline(median_umi, color='green', linestyle='--', label=f'Median: {int(median_umi)}')
    plt.axvline(p1, color='purple', linestyle=':', label=f'1th %: {int(p1)}')
    
    plt.xscale('log') 
    plt.title(f"UMI Counts per Spot (Library Depth)\nN={len(valid_barcodes)} spots")
    plt.xlabel("Total UMI Counts (Log Scale)")
    plt.ylabel("Number of Spots")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 1 took {duration:.6f} seconds")
    return counts, gene_names, valid_barcodes, coords, total_counts_orig, scale_factors

def step2_estimate_cell_density(counts, gene_names, total_counts_orig, coords, config_path, 
                                min_umi=300,
                                anchor_mean_factor=1.0, 
                                colormap='set1'):
    """
    Step 2: Estimation of Cell Density.
    
    - Loads Housekeeping (HK) genes from a reference standard.
    - Calibrates spot signal against the reference (log1p CP10K).
    - estimates number of cells per spot.
    - Filters low-quality spots based on UMI threshold.
    """
    start_time = time.perf_counter()
    # 1. Load Single Cell Standard
    print(f"Step 2: Loading Single Cell Standard from {config_path}...")
    with open(config_path, 'r') as f:
        standard_data = json.load(f)
    
    hk_reference = standard_data['hk_profiles']
    engine_params = standard_data['engine_parameters']
    
    # 2. Align HK Genes
    common_hk = [g for g in hk_reference.keys() if g in gene_names]
    if not common_hk:
        raise ValueError("Step 2 Error: No overlap with Single Cell Standard HK genes!")
        
    print(f"Step 2: Using {len(common_hk)} HK genes for calibration.")
    
    hk_indices = [gene_names.index(g) for g in common_hk]
    ref_values_raw = np.array([hk_reference[g] for g in common_hk])
    
    # 3. Calculate Normalized HK Expression (Visium) -> Log Space
    # Normalize Visium data to the same log1p(CP10K) scale as the standard
    
    safe_total = total_counts_orig.copy()
    safe_total[safe_total == 0] = 1 # Avoid division by zero
    
    hk_counts_raw = counts[:, hk_indices].toarray()
    
    # Formula: log1p( (count / total) * 10000 )
    normalized_hk_log = np.log1p((hk_counts_raw / safe_total[:, np.newaxis]) * 10000)
    
    # 4. standard Reference Preparation
    # Config values are in the format log1p(CP10K).
    ref_values_log = ref_values_raw 

    # 5. Density Calculation (Geometric Mean Logic)
    # Compare Mean Log Expression of the Spot vs the standard
    
    spot_hk_log_means = np.mean(normalized_hk_log, axis=1) # Log-level Visium
    standard_anchor_log_mean = np.mean(ref_values_log)        # Log-level standard
    
    # Apply Factor
    # Log(standard * Factor) = Log(standard) + Log(Factor)
    adjusted_standard_log_mean = standard_anchor_log_mean + np.log(anchor_mean_factor)
    
    # Convert back to Linear Space to calculate ratio (Number of Cells)
    spot_signal_linear = np.expm1(spot_hk_log_means)
    standard_signal_linear = np.expm1(adjusted_standard_log_mean)
    
    # Safety: Avoid division by zero
    if standard_signal_linear < 0.001: standard_signal_linear = 0.001
    
    raw_n_cells = spot_signal_linear / standard_signal_linear
    n_cells = np.round(raw_n_cells).astype(int)
    
    # 6. Filtering & QC
    is_low_quality = total_counts_orig < min_umi
    n_cells[is_low_quality] = 0
    
    # PLOT 1: Calibration Check
    plt.figure(figsize=(8, 5), dpi=100)
    
    valid_signals = spot_signal_linear[total_counts_orig > min_umi]
    max_x = max(np.percentile(valid_signals, 99) if len(valid_signals)>0 else 10, standard_signal_linear * 2)
    bins = np.linspace(0, max_x, 50)
    
    plt.hist(valid_signals, bins=bins, color='purple', alpha=0.6, label='Spot HK Signal (Visium)')
    plt.axvline(standard_signal_linear, color='red', linewidth=2, linestyle='--', 
                label=f'Single Cell Standard ({standard_signal_linear:.2f})')
    
    plt.title(f"Calibration Check: Spots vs Single Cell Standard\n(Factor={anchor_mean_factor})")
    plt.xlabel("Geometric Mean of HK Normalized Expression")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"   > Single Cell Standard is: {standard_signal_linear:.2f}")

    # Stats
    n_zeros = np.sum(n_cells == 0)
    print(f"\nFILTERING REPORT")
    print(f"Total Empty Spots (0 cells): {n_zeros} / {len(n_cells)} ({n_zeros/len(n_cells)*100:.1f}%)")
    print(f"Among them are those with low UMI content: {sum(is_low_quality)}")

    # PLOT 2: Density Histogram
    plt.figure(figsize=(8, 5), dpi=100)
    
    max_val = int(np.max(n_cells)) if len(n_cells) > 0 else 0
    if max_val > 0:
        bins = np.arange(1, max_val + 2) - 0.5 
        plt.hist(n_cells[n_cells > 0], bins=bins, 
                 color='#27ae60', edgecolor='white', alpha=0.9, label='Tissue')
                 
    plt.bar(0, n_zeros, color='#95a5a6', edgecolor='white', width=0.8, label='Background')
    
    plt.title(f"Cell Density Distribution")
    plt.xlabel("Number of Cells")
    plt.xticks(np.arange(0, max(1, max_val) + 1, 1)) # Force integer ticks
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

    # PLOT 3: Spatial Map
    y = coords[:, 0]
    x = coords[:, 1]
    
    plt.figure(figsize=(8, 8), dpi=100)
    
    bg_mask = n_cells == 0
    plt.scatter(x[bg_mask], y[bg_mask], c='grey', s=10, alpha=0.3)
    
    fg_mask = n_cells > 0
    if np.any(fg_mask):
        sc = plt.scatter(x[fg_mask], y[fg_mask], c=n_cells[fg_mask], 
                         cmap=colormap, s=15, linewidth=0)
        cbar = plt.colorbar(sc, label='Cells per Spot', fraction=0.046, pad=0.04)
    
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.title("Spatial Density")
    plt.show()

    print(f"\nThe number of cells on the Visium slide is {sum(n_cells)}.")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 2 took {duration:.6f} seconds")
    return n_cells, engine_params

def step3_feature_selection(counts, gene_names, species='hs', n_hvg=3000):
    """
    Step 3: Biological Cleanup & Highly Variable Gene (HVG) Selection.
    
    Performs quality control by removing non-informative transcripts (MT, Ribo)
    and identifies the most informative features for LDA topic modeling using
    dispersion-based ranking.
    """
    start_time = time.perf_counter()
    print(f"Step 3: Filtering transcriptome (Species: {species})...")
    
    # 1. Define Biological Noise Patterns (Regex)
    # MT: Mitochondrial (stress/dying cells)
    # RP: Ribosomal (technical noise, usually 30-40% of reads)
    # LINC/Gm: Long Non-coding/Pseudogenes (low mapping quality)
    if species == 'hs':
        # Human patterns
        noise_pattern = r'^MT-|^RP[SL][0-9]+|^LINC|^MIR|^AC[0-9]+' 
    elif species == 'mm':
        # Mouse patterns
        noise_pattern = r'^mt-|^Rp[sl][0-9]+|^Gm|^Mir|^Rik'
    else:
        raise ValueError("Species must be 'hs' or 'mm'")
        
    bio_re = re.compile(noise_pattern, re.IGNORECASE)
    
    # 2. Filter Genes
    keep_indices = []
    genes_clean = []
    
    for i, gene in enumerate(gene_names):
        if not bio_re.match(gene):
            keep_indices.append(i)
            genes_clean.append(gene)
            
    # Slicing columns (genes) from the sparse matrix
    counts_clean = counts[:, keep_indices]
    print(f"   - Removed {len(gene_names) - len(genes_clean)} noise genes.")
    print(f"   - Remaining feature space: {len(genes_clean)} genes.")

    # 3. Identify Highly Variable Genes (HVG)
    # Logic: LDA needs genes that distinguish cell types, not housekeeping genes.
    # We implement Seurat-style Log-Variance selection (vst) mathematically.
    
    print(f"Step 3: Selecting top {n_hvg} HVGs for Model Training...")
    
    # A. Calculate Mean & Variance in Sparse Mode (Memory Efficient)
    # Formula: Var(X) = E[X^2] - (E[X])^2
    
    # First, calculate row sums (sequencing depth per spot)
    row_sums = np.array(counts_clean.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1 # Avoid div by zero
    
    # Normalize matrix (CP10K) without densifying
    scaling_factor = 10000.0
    
    # Calculate Mean expression per gene (normalized)
    # E[X] = (Sum_cols / Sum_rows) * Scaling - simplified approximation for selection
    # For robust selection, we use Log(CP10K) statistics:
    
    # Create a normalized copy just for stats calculation
    norm_counts = counts_clean.copy().astype(float)
    # Sparse broadcast division (divide each row by its depth)
    norm_counts.data /= np.repeat(row_sums, np.diff(norm_counts.indptr))
    norm_counts.data *= scaling_factor
    norm_counts.data = np.log1p(norm_counts.data) # Log space
    
    # Calculate Mean and Variance of Log-Normalized data
    mean_expr = np.array(norm_counts.mean(axis=0)).flatten()
    
    # Var = Mean(X^2) - Mean(X)^2
    sq_counts = norm_counts.copy()
    sq_counts.data **= 2
    mean_sq = np.array(sq_counts.mean(axis=0)).flatten()
    var_expr = mean_sq - (mean_expr ** 2)
    
    # B. Standardize Variance (Dispersion)
    # Genes with high mean naturally have high variance. We need genes with
    # high variance relative to their mean.
    # Loess regression simulation: binning means and z-scoring variance
    n_bins = 20
    bins = np.linspace(np.min(mean_expr), np.max(mean_expr), n_bins + 1)
    dispersion_norm = np.zeros_like(var_expr)
    
    for i in range(n_bins):
        idx = np.where((mean_expr >= bins[i]) & (mean_expr < bins[i+1]))[0]
        if len(idx) > 0:
            bin_var = var_expr[idx]
            bin_mean_var = np.mean(bin_var)
            bin_std_var = np.std(bin_var)
            if bin_std_var > 0:
                dispersion_norm[idx] = (bin_var - bin_mean_var) / bin_std_var
                
    # 4. Select Top N Genes
    # Sort by normalized dispersion (descending)
    hvg_indices_local = np.argsort(dispersion_norm)[-n_hvg:][::-1]
    
    # Map back to original clean indices
    hvg_data = counts_clean[:, hvg_indices_local]
    hvg_names = [genes_clean[i] for i in hvg_indices_local]
    
    # 5. Diagnostic Plot (Mean vs Dispersion)
    plt.figure(figsize=(9, 7))
    plt.scatter(mean_expr, dispersion_norm, s=1, color='grey', alpha=0.5, label='Non-HVG')
    plt.scatter(mean_expr[hvg_indices_local], dispersion_norm[hvg_indices_local], s=1, color='red', label='HVG')
    plt.xlabel('Mean Expression (Log)')
    plt.ylabel('Normalized Dispersion')
    plt.title(f'Step 3: Feature Selection ({n_hvg} genes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Step 3 Complete. Training set shape: {hvg_data.shape}")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 3 took {duration:.6f} seconds")
    return counts_clean, genes_clean, hvg_data, hvg_names

def step4_gene_manifold(counts_clean, genes_clean, n_components = 100):
    """
    Step 4: Gene Manifold Learning via ICA and UMAP.
    
    Constructs a topological map of the transcriptome. Unlike PCA, which focuses 
    on variance (often dominated by sequencing depth), FastICA isolates 
    independent biological signals. UMAP then projects these high-dimensional 
    signals into a latent space where cosine similarity reflects co-expression.
    """
    start_time = time.perf_counter()
    print(f"Step 4: Constructing Gene Manifold for {hvg_data.shape[1]} genes...")
    
    # 1. Transpose: We cluster GENES, not SPOTS.
    # Input: (Spots x Genes) -> Transposed: (Genes x Spots)
    # Now, 'samples' are genes, and 'features' are their expression across spots.
    X_genes = hvg_data.T 
    
    # 2. Preprocessing
    # Log-transform and scale to unit variance.
    # Scaling is critical for ICA convergence.
    print("   - Preprocessing: Log-transformation and Scaling...")
    X_dense = np.log1p(X_genes.toarray())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)
    
    # 3. Independent Component Analysis (FastICA)
    # Extracts 30 independent biological signals (latent topics precursor).
    # ICA is superior to PCA for separating mixed signals in Visium spots.
    print(f"   - Decomposition: FastICA ({n_components} components)...")
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, tol=0.005)
    X_ica = ica.fit_transform(X_scaled)
    
    # 4. Manifold Projection (UMAP)
    # Metric='cosine' is crucial for gene expression vectors.
    print("   - Topology: UMAP Projection (Metric: Cosine)...")
    reducer = umap.UMAP(
        n_neighbors=30,      # Larger neighborhood to preserve global structure
        min_dist=0.1,        # Tight packing of co-expressed genes
        n_components=2,      # 2D for visualization
        metric='cosine',
        random_state=42
    )
    embedding = reducer.fit_transform(X_ica)
    
    # 5. Quality Control: Manifold Visualization
    plt.figure(figsize=(8, 7), dpi=100)
        
    # Plot all genes as background
    plt.scatter(embedding[:, 0], embedding[:, 1], s=2, c='lightgrey', alpha=0.4, label='Genes')
        
    # Highlight key markers to validate topology (if present in HVGs)
    # These markers cover major cell lineages
    qc_markers = ['CD3E', 'CD4', 'CD8A', 'MS4A1', 'CD19', 'PTPRC', 
                  'HBB', 'CD14', 'FCGR3A', 'CD34', 'NCAM1', 'JCHAIN', 
                  'EPCAM', 'KRT18', 'COL1A1', 'DCN', 'PECAM1','ERBB2']
        
    colors = plt.cm.hsv(np.linspace(0, 1, len(qc_markers)))
        
    found_any = False
    for idx, gene in enumerate(qc_markers):
        if gene in hvg_names:
            g_idx = hvg_names.index(gene)
            plt.scatter(embedding[g_idx, 0], embedding[g_idx, 1], 
                        s=100, color=colors[idx], edgecolors='black', label=gene, zorder=10)
            # Add text annotation
            plt.text(embedding[g_idx, 0]+0.1, embedding[g_idx, 1]+0.1, gene, fontsize=9)
            found_any = True
        
    plt.title(f"Step 4: Gene Co-Expression Manifold (ICA+UMAP)\nGenes clustered by spatial co-occurrence")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    if found_any:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 4 took {duration:.6f} seconds") 
    return embedding, reducer

def step5_ksweep(hvg_data, min_k=2, max_k=20, step=2, subsample_frac=0.3,
            doc_topic_prior=0.1,
            topic_word_prior=0.01):
    """
    LDA Elbow Plot.
    """
    start_time = time.perf_counter()
    print(f"Step 5: K-Sweep (Range: {min_k}-{max_k}, Sampling: {int(subsample_frac*100)}%)...")
    
    # 1. Subsampling
    n_spots = hvg_data.shape[0]
    if subsample_frac < 1.0 and n_spots > 1000:
        n_sub = int(n_spots * subsample_frac)
        rng = np.random.default_rng(42)
        indices = rng.choice(n_spots, n_sub, replace=False)
        data_for_sweep = hvg_data[indices, :]
        print(f"   > Using subset: {n_sub} spots out of {n_spots}")
    else:
        data_for_sweep = hvg_data

    ks = range(min_k, max_k+1, step)
    perps = []
    
    for k in tqdm(ks, desc="Training LDA models"):
        lda = LatentDirichletAllocation(
            n_components=k, 
            learning_method='online', 
            learning_offset=50., 
            max_iter=5,          
            random_state=42, 
            n_jobs=-1,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior
        )
        lda.fit(data_for_sweep)
        perps.append(lda.perplexity(data_for_sweep))
        
    plt.figure(figsize=(10, 5))
    plt.plot(ks, perps, 'bo-', markerfacecolor='white')
    plt.xlabel("K Topics"); plt.ylabel("Perplexity (Lower is better)")
    plt.title("Elbow Plot")
    plt.grid(True, alpha=0.3)
    plt.show()
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 5 took {duration:.6f} seconds")
    return dict(zip(ks, perps))

def step6_final_deconvolution(hvg_data, counts_clean, genes_clean, hvg_names, n_topics, 
                             embedding, n_iters=30, k_neighbors=3, min_sim=0.1, 
                             doc_topic_prior=0.1, topic_word_prior=0.01):
    """
    Step 6: Final Deconvolution & Manifold-Guided Transcriptome Projection.
    
    This version uses UMAP coordinates to find neighbors for non-HVG genes,
    ensuring that markers like CD14 inherit profiles from their biological 
    cluster rather than random spatial matches.
    """
    start_time = time.perf_counter()
    print(f"Step 6: Training Final LDA Model with chosen K={n_topics}...")
    
    # 1. Train the definitive LDA model on Highly Variable Genes (HVG)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='batch',
        max_iter=n_iters,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior
    )
    
    # Extract Spot-Topic (theta) and Topic-Gene (beta) distributions
    theta = lda.fit_transform(hvg_data)
    beta_hvg = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    
    print("Step 6: Projecting latent topics using UMAP manifold topology...")
    
    # 2. Prepare Manifold-based neighbor search
    # Map HVG gene names to their indices in the full cleaned gene list
    hvg_indices_in_clean = [genes_clean.index(name) for name in hvg_names]
    hvg_coords = embedding[hvg_indices_in_clean] # Coordinates of anchor genes
    all_coords = embedding                       # Coordinates of all genes in the manifold
    
    # Calculate Euclidean distances in the 2D UMAP space
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(all_coords, hvg_coords, metric='euclidean')
    
    # Calculate Cosine Similarity in the expression space for final weighting
    norm_all = normalize(counts_clean.T, axis=1)
    norm_hvg = normalize(hvg_data.T, axis=1)
    similarity = norm_all @ norm_hvg.T
    
    beta_final = np.zeros((n_topics, len(genes_clean)))
    
    # 3. Projection Loop
    for i in range(len(genes_clean)):
        # Find K nearest neighbors specifically in the UMAP 2D latent space
        umap_neighbors_idx = np.argsort(dist_matrix[i])[:k_neighbors]
        
        # Retrieve the expression similarity scores for these geometric neighbors
        sim_row = similarity[i].toarray().flatten()
        weights = sim_row[umap_neighbors_idx]
        if np.max(weights) < min_sim:
            continue
            
        # Reconstruct the gene's topic profile as a weighted sum of its manifold neighbors
        proj = beta_hvg[:, umap_neighbors_idx] @ weights
        if proj.sum() > 0:
            beta_final[:, i] = proj / proj.sum()
            
    # 4. Generate Quality Control (QC) Table
    topic_dict = {}
    for k in range(n_topics):
        # Identify top 15 genes with highest probability in each topic
        top_idx = beta_final[k].argsort()[::-1][:15]
        topic_dict[f"Topic_{k}"] = [genes_clean[i] for i in top_idx]
        
    print(f"Step 6 Complete.")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 6 took {duration:.6f} seconds")
    return theta, beta_final, pd.DataFrame(topic_dict)

def step7_sampling_engine(raw_counts, n_cells_spot, theta, beta, config_params):
    """
    Step 7: The Sampling Engine
    Converts Spot-level expression into Single-Cell-level expression using 
    a two-stage hierarchical Bayesian sampling process.
    """
    start_time = time.perf_counter()
    print("Step 7: initializing Sampling Engine...")
    
    # Parameters for Gamma (Shape/Scale formulation)
    # Mean = shape * scale = mu
    # Variance = shape * scale^2 = phi * mean
    # Solving for shape/scale:
    # shape (k) = 1 / phi
    # scale (theta_gamma) = mu * phi
    gamma_shape = 1.0 / config_params['phi']
    gamma_scale = config_params['mu'] * config_params['phi']
    
    n_spots, n_genes = raw_counts.shape
    n_topics = theta.shape[1]
    
    # Storage for sparse matrix construction
    rows, cols, data = [], [], []
    cell_metadata = []
    global_cell_idx = 0
    
    for s in tqdm(range(n_spots), desc="Generating Single Cells", unit="spot"):
            
        n_total = n_cells_spot[s]
        if n_total == 0: continue
            
        # 1. Determine Cell Composition per Spot
        # Distribute n_total based on theta probabilities
        topic_dist = np.random.multinomial(n_total, theta[s])
        
        # 2. Generate In Silico Cells
        # Structure: cells_in_spot[topic_k] = [weight_c1, weight_c2, ...]
        cells_weights = {}
        cells_indices = {} # Local mapping to global index
        
        current_spot_cell_idx = 0
        for k in range(n_topics):
            n_k = topic_dist[k]
            if n_k > 0:
                # Gamma weights for heterogeneity
                w = np.random.gamma(gamma_shape, gamma_scale, size=n_k)
                cells_weights[k] = w / w.sum() # Normalize within topic group
                
                # Register metadata
                for i in range(n_k):
                    cell_metadata.append({
                        'spot_idx': s,
                        'topic_idx': k,
                        'cell_num': current_spot_cell_idx + i + 1
                    })
                
                # Map local topic-cell index to global matrix index
                cells_indices[k] = [global_cell_idx + i for i in range(n_k)]
                global_cell_idx += n_k
                current_spot_cell_idx += n_k
        
        # 3. UMI Allocation
        # Get genes present in this spot
        spot_vec = raw_counts[s].toarray().flatten()
        genes_present = np.where(spot_vec > 0)[0]
        
        for g in genes_present:
            count = int(spot_vec[g])
            
            # A. Probability of this gene coming from Topic K in Spot S
            # P(k | g, s) ~ beta[k, g] * theta[s, k]
            p_topic = beta[:, g] * theta[s]
            
            if p_topic.sum() == 0:
                # Fallback if gene is unknown to model (should be rare with projection)
                p_topic = np.ones(n_topics) / n_topics
            else:
                p_topic /= p_topic.sum()
            
            # Sample topics for these UMI copies
            umi_per_topic = np.random.multinomial(count, p_topic)
            
            # B. Distribute to Cells within Topic
            for k in range(n_topics):
                u_count = umi_per_topic[k]
                if u_count > 0 and k in cells_weights:
                    # Distribute UMI among cells of this topic based on brightness
                    umi_per_cell = np.random.multinomial(u_count, cells_weights[k])
                    
                    # Record non-zero entries
                    for idx, val in enumerate(umi_per_cell):
                        if val > 0:
                            rows.append(cells_indices[k][idx])
                            cols.append(g)
                            data.append(val)
                            
    print(f"\n   > Generated {global_cell_idx} single cells total.")
    
    # Construct CSR Matrix
    final_matrix = sp.csr_matrix((data, (rows, cols)), shape=(global_cell_idx, n_genes))
    print(f"Step 7 Complete.")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 7 took {duration:.6f} seconds")
    return final_matrix, cell_metadata

def step8_geometry_and_placement(cell_metadata, spot_coords, spot_barcodes, scale_factors, rotation=0, flip=None):
    """
    Step 8: Places single cells around spot centers using Fibonacci patterns.
    """
    start_time = time.perf_counter()
    print("   > Calculating Fibonacci shell positions...")
    diameter = (scale_factors['spot_diameter_fullres'])
    phi = np.pi * (3. - np.sqrt(5.)) # Golden angle
    
    final_barcodes = []
    final_coords = [] # [pxl_row, pxl_col, array_row, array_col]
    spot_counters = {} 
    
    # Pre-calculate total cells per spot
    spot_totals = {}
    for meta in cell_metadata:
        s = meta['spot_idx']
        spot_totals[s] = spot_totals.get(s, 0) + 1
        
    for meta in cell_metadata:
        s = meta['spot_idx']
        t = meta['topic_idx']
        
        # Which cell is this in the sequence (0, 1, 2...)?
        idx = spot_counters.get(s, 0)
        spot_counters[s] = idx + 1
        n_total = spot_totals[s]

        # spot_coords structure is [pxl_row, pxl_col, array_row, array_col]
        cy = spot_coords[s, 0] # pxl_row (Y)
        cx = spot_coords[s, 1] # pxl_col (X)
        
        # Fibonacci Formula
        r = diameter * np.sqrt(idx) / np.sqrt(n_total) if n_total > 1 else 0
        ang = idx * phi
        
        # Cell coordinates
        py = cy + r * np.sin(ang)
        px = cx + r * np.cos(ang)
        
        # Preserve original array coordinates for Seurat graph building
        ay, ax = spot_coords[s, 2], spot_coords[s, 3]
        
        final_coords.append([py, px, ay, ax])
        
        # Construct Barcode: AAACCGG-1-Topic-5-Cell-3
        orig_bc = spot_barcodes[s]
        new_bc = f"{orig_bc}-Topic-{t}-Cell-{meta['cell_num']}"
        final_barcodes.append(new_bc)
        
    print(f"Step 8 Complete.")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 8 took {duration:.6f} seconds")
    return final_barcodes, np.array(final_coords)

def step9_export_results(output_path, final_matrix, genes, barcodes, coords, orig_spatial_path):
    """
    Step 9: Final Export to 10X SpaceRanger HDF5 Format (.h5) + Spatial Folder.
    This format is natively supported by Seurat::Load10X_Spatial().
    """
    start_time = time.perf_counter()
    print(f"Step 9: Exporting data to {output_path}/deconvolved ...")
    
    out_dir = os.path.join(output_path, "deconvolved")
    spa_dir = os.path.join(out_dir, "spatial")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(spa_dir, exist_ok=True)
        
    # Export Feature Matrix to HDF5 (.h5)
    h5_path = os.path.join(out_dir, "filtered_feature_bc_matrix.h5")
    
    print(f"   > Writing HDF5 matrix to {h5_path}...")
    
    with h5py.File(h5_path, "w") as f:
        # Group structure required by CellRanger/Seurat
        grp = f.create_group("matrix")
        
        # Features (Genes)
        grp.create_dataset("features/id", data=np.array(genes, dtype='S'))
        grp.create_dataset("features/name", data=np.array(genes, dtype='S'))
        grp.create_dataset("features/feature_type", data=np.array(["Gene Expression"] * len(genes), dtype='S'))
        grp.create_dataset("features/genome", data=np.array(["Genome"] * len(genes), dtype='S'))
        grp.create_dataset("features/_all_tag_keys", data=np.array([], dtype='S'))
        
        # Barcodes
        grp.create_dataset("barcodes", data=np.array(barcodes, dtype='S'))
        csc = final_matrix.T.tocsc() 
        
        grp.create_dataset("data", data=csc.data)
        grp.create_dataset("indices", data=csc.indices)
        grp.create_dataset("indptr", data=csc.indptr)
        grp.create_dataset("shape", data=np.array(csc.shape, dtype='i4'))

    # Copy images and scale factors
    for f in ['tissue_lowres_image.png', 'tissue_hires_image.png', 'scalefactors_json.json']:
        src = os.path.join(orig_spatial_path, f)
        dst = os.path.join(spa_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    # Save Tissue Positions
    pos_df = pd.DataFrame(coords, columns=['pxl_row', 'pxl_col', 'array_row', 'array_col'])
    pos_df['barcode'] = barcodes
    pos_df['in_tissue'] = 1
    
    # Reorder to standard 10X columns
    pos_df = pos_df[['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']]
    
    pos_df.to_csv(os.path.join(spa_dir, "tissue_positions.csv"), index=False)
    
    print("Step 9 Complete. Output is ready for Seurat::Load10X_Spatial().")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Step 9 took {duration:.6f} seconds")
