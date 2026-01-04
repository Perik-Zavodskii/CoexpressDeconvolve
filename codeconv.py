"""
CoexpressDeconvolve Module
Designed for spatial transcriptomics deconvolution using marker-assisted gene co-expression clustering
Features:
- Dimensionality Reduction: FastICA
- Clustering: Leiden + Marker Watershed
- Spatial: K-Means Niche detection
- HPA Mode: Direct Cell Type Marker Mapping to Humap Protein Atlas Gene Clusters
"""

import os
import gc
import json
import math
import time
import base64
import shutil
import gzip
from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import io, stats, sparse
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2, whiten
import dask
from dask import delayed
import jinja2

# Machine Learning Imports
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import homogeneity_score

# Try importing Leidenalg and Igraph
try:
    import leidenalg
    import igraph as ig
except ImportError:
    raise ImportError("Please install leidenalg and igraph: pip install leidenalg igraph")

try:
    import umap
except ImportError:
    raise ImportError("Please install umap-learn: pip install umap-learn")

# Try importing display for notebook rendering
try:
    from IPython.display import display
except ImportError:
    display = print

# Configuration & Constants

# Deconvolution Strategy
DECONVOLUTION = "immune" # 'immune', 'immune+custom', 'custom'
CUSTOM_MARKERS = None 

MARKERS_HS = {
    'Other T': {
        'pos': {'TRAC','TRBC1','TRBC2','CD3D','CD3E','CD2','PTPRC','CD4','IL7R','LCK','LAT','CD27','BCL11B','TCF7','CCR7','LEF1','MAL','NOSIP','LDHB'}, 
        'neg': {'CD8A','NKG7','GNLY','CD19','MS4A1','CD14','LYZ','EPCAM'}
    },
    'Treg': {
        'pos': {'FOXP3', 'IL2RA', 'CTLA4', 'ICOS', 'CCR8', 'IKZF2', 'TIGIT', 'TNFRSF18', 'TNFRSF4', 'BATF', 'LAYN'},
        'neg': {'CD8A', 'CD19', 'MS4A1', 'CD14', 'EPCAM'}
    },
    'Cytotoxic T/NK': {
        'pos': {'CD8A','CD8B','NKG7','GNLY','PRF1','GZMB','GZMA','GZMH','GZMK','KLRD1','KLRK1','NCAM1','FCGR3A','CST7','CTSW','FGFBP2','HOPX','MATK','ZAP70'},
        'neg': {'CD4','CD19','MS4A1','CD14','LYZ','EPCAM','S100A8'}
    },
    'B': {
        'pos': {'MS4A1','CD19','CD79A','CD79B','PAX5','BLK','CD22','FCRL5','BANK1','CR2','CD72','STAP1','FCRLA','HVCN1','TCL1A','CD24'}, 
        'neg': {'CD3E','CD14','LYZ','EPCAM','GNLY','COL1A1'}
    },
    'Plasma': {
        'pos': {'SDC1','TNFRSF17','MZB1','JCHAIN','IGHG1','IGHG3','IGKC','IGLC2','CD38','PRDM1','XBP1','IRF4','SLAMF7','SSR4'}, 
        'neg': {'CD19','MS4A1','CD3E','CD14','EPCAM','COL1A1'}
    },
    'Mono/Macro': {
        'pos': {'CD14','LYZ','S100A8','S100A9','CD68','CD163','FCGR1A','CSF1R','MRC1','C1QA','C1QB','C1QC','TYROBP','FCN1','VCAN','LST1','AIF1','MSR1'}, 
        'neg': {'CD3E','CD19','MS4A1','EPCAM','TRAC','GNLY','CD8A','COL1A1'}
    },
    'APCs': {
        'pos': {'HLA-DRA', 'HLA-DRB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DMA', 'HLA-DMB', 'CD74', 'CTSS'},
        'neg': {'COL1A1', 'EPCAM'}
    },
    'Mast': {
        'pos': {'TPSAB1', 'TPSB2', 'CPA3', 'MS4A2', 'KIT', 'SIGLEC6', 'HDC'},
        'neg': {'CD3E', 'CD19', 'EPCAM', 'COL1A1'}
    },
    'Erythroid': {
        'pos': {'HBA1','HBA2','HBZ','HBM','HBE1','HBG1','HBG2','HBD','HBB','ALAS2','GYPA','AHSP','HEMGN','SLC4A1','CA1','BPGM'}, 
        'neg': {'EPCAM','CD3E','CD19','CD14','COL1A1'}
    },
    'Cycling': {
        'pos': {'MKI67', 'TOP2A', 'PCNA', 'CDK1', 'CCNA2', 'CCNB1', 'MCM6', 'MCM2', 'CDC20', 'TYMS'},
        'neg': set()
    },
    'Endothelial': {
        'pos': {'PECAM1','VWF','CDH5','CLDN5','PLVAP','KDR','FLT1','ENG','MCAM','MMRN1','ECSCR','GNG11','RAMP2','HSPG2','NOSTRIN'}, 
        'neg': {'COL1A1','EPCAM','PTPRC','CD14','CD19','ACTA2'}
    },
    'Fibroblast': {  
        'pos': {'COL1A1','COL1A2','COL3A1','DCN','LUM','PDGFRA','PDGFRB','POSTN','FAP','MMP2','VIM','FN1','SPARC','C1R','C1S'}, 
        'neg': {'PECAM1','EPCAM','PTPRC','CD14','CD19','CD8A','MYH11', 'ACTA1'}
    },
    'Smooth Muscle': {
        'pos': {'ACTA2', 'TAGLN', 'MYH11', 'CNN1', 'PLN', 'CALD1', 'TPM2', 'MYLK'},
        'neg': {'PTPRC', 'EPCAM', 'CD14', 'VWF', 'ACTA1'}
    },
    'Skeletal Muscle': {
        'pos': {'ACTA1', 'TTN', 'MYH1', 'MYH2', 'MYH7', 'TNNT3', 'TNNC2', 'CKM', 'DES', 'MYL1', 'NEB'},
        'neg': {'EPCAM','KRT14','KRT5','PTPRC','CD14','PECAM1'}
    },
    'Adipocytes': {
        'pos': {'PLIN1', 'ADIPOQ', 'FABP4', 'LPL', 'GPAM', 'THRSP'},
        'neg': {'EPCAM', 'PTPRC', 'COL1A1'}
    },
    'Epithelial': {
        'pos': {'EPCAM','KRT8','KRT18','KRT19','CDH1','KRT7','ELF3','MUC1','CLDN3','CLDN4','CLDN7','TACSTD2','S100P','AGR2','PERP'}, 
        'neg': {'PTPRC','VIM','COL1A1','CD14','CD19','PECAM1'}
    },
    'Tumor': {
        'pos': {'ERBB2','MAGEA3','CEACAM5','CEACAM6','TFF1','TFF3','AGR3','GATA3','FOXA1','MLPH','DSC2','DSG2'}, 
        'neg': {'PTPRC','VIM','COL1A1','CD14','CD19','LYZ'}
    }
}

MARKERS_MM = {
    'Other T': {
        'pos': {'Cd4','Trac','Trbc1','Trbc2','Cd3d','Cd3e','Cd3g','Il7r','Lck','Lat','Cd27','Bcl11b','Tcf7','Ccr7','Lef1','Mal','Nosip','Ldhb'}, 
        'neg': {'Cd8a','Nkg7','Gnly','Cd19','Ms4a1','Cd14','Lyz2','Epcam'}
    },
    'Treg': {
        'pos': {'Foxp3', 'Il2ra', 'Ctla4', 'Icos', 'Ccr8', 'Ikzf2', 'Tigit', 'Tnfrsf18', 'Tnfrsf4', 'Batf', 'Layn'},
        'neg': {'Cd8a', 'Cd19', 'Ms4a1', 'Cd14', 'Epcam'}
    },
    'Cytotoxic T/NK': {
        'pos': {'Cd8a','Cd8b1','Nkg7','Gnly','Prf1','Gzmb','Gzma','Gzmh','Gzmk','Klrd1','Klrk1','Ncam1','Fcgr4','Cst7','Ctsw','Fgfbp2','Hopx','Matk','Zap70'},
        'neg': {'Cd4','Cd19','Ms4a1','Cd14','Lyz2','Epcam','S100a8'}
    },
    'B': {
        'pos': {'Ms4a1','Cd19','Cd79a','Cd79b','Pax5','Blk','Cd22','Fcrl5','Bank1','Cr2','Cd72','Stap1','Fcrla','Hvcn1','Tcl1','Cd24a'}, 
        'neg': {'Cd3e','Cd14','Lyz2','Epcam','Gnly','Col1a1'}
    },
    'Plasma': {
        'pos': {'Sdc1','Tnfrsf17','Mzb1','Jchain','Ighg1','Ighg3','Igkc','Iglc2','Cd38','Prdm1','Xbp1','Irf4','Slamf7','Ssr4'}, 
        'neg': {'Cd19','Ms4a1','Cd3e','Cd14','Epcam','Col1a1'}
    },
    'Mono/Macro': {
        'pos': {'Cd14','Lyz2','S100a8','S100a9','Cd68','Cd163','Fcgr1','Csf1r','Mrc1','C1qa','C1qb','C1qc','Tyrobp','Fcn1','Vcan','Lst1','Aif1','Msr1'}, 
        'neg': {'Cd3e','Cd19','Ms4a1','Epcam','Trac','Gnly','Cd8a','Col1a1'}
    },
    'APCs': {
        'pos': {'H2-Aa', 'H2-Ab1', 'H2-Eb1', 'H2-Eb2', 'H2-DMb1', 'Cd74', 'Ctss'},
        'neg': {'Col1a1', 'Epcam'}
    },
    'Mast': {
        'pos': {'Cpa3', 'Mcpt4', 'Mcpt1', 'Kit', 'Ms4a2', 'Tpsb2'},
        'neg': {'Cd3e', 'Cd19', 'Epcam', 'Col1a1'}
    },
    'Erythroid': {
        'pos': {'Hbb-bs','Hbb-bt','Hba-a1','Hba-a2','Alas2','Gypa','Ahsp','Hemgn','Slc4a1','Car1','Bpgm'}, 
        'neg': {'Epcam','Cd3e','Cd19','Cd14','Col1a1'}
    },
    'Cycling': {
        'pos': {'Mki67', 'Top2a', 'Pcna', 'Cdk1', 'Ccna2', 'Ccnb1', 'Mcm6', 'Mcm2', 'Cdc20', 'Tyms'},
        'neg': set()
    },
    'Endothelial': {
        'pos': {'Pecam1','Vwf','Cdh5','Cldn5','Plvap','Kdr','Flt1','Eng','Mcam','Mmrn1','Ecscr','Gng11','Ramp2','Hspg2','Nostrin'}, 
        'neg': {'Col1a1','Epcam','Ptprc','Cd14','Cd19','Acta2'}
    },
    'Fibroblast': {
        'pos': {'Col1a1','Col1a2','Col3a1','Dcn','Lum','Pdgfra','Pdgfrb','Postn','Fap','Mmp2','Vim','Fn1','Sparc','C1r','C1s'}, 
        'neg': {'Pecam1','Epcam','Ptprc','Cd14','Cd19','Cd8a', 'Myh11', 'Acta1'}
    },
    'Smooth Muscle': {
        'pos': {'Acta2', 'Tagln', 'Myh11', 'Cnn1', 'Pln', 'Cald1', 'Tpm2', 'Mylk'},
        'neg': {'Ptprc', 'Epcam', 'Cd14', 'Vwf', 'Acta1'}
    },
    'Skeletal Muscle': {
        'pos': {'Acta1', 'Ttn', 'Myh1', 'Myh2', 'Myh7', 'Tnnt3', 'Tnnc2', 'Ckm', 'Des', 'Myl1', 'Neb'},
        'neg': {'Epcam','Krt14','Krt5','Ptprc','Cd14','Pecam1'}
    },
    'Adipocytes': {
        'pos': {'Plin1', 'Adipoq', 'Fabp4', 'Lpl', 'Gpam', 'Thrsp'},
        'neg': {'Epcam', 'Ptprc', 'Col1a1'}
    },
    'Epithelial': {
        'pos': {'Epcam','Krt8','Krt18','Krt19','Cdh1','Krt7','Elf3','Muc1','Cldn3','Cldn4','Cldn7','Tacstd2','S100p','Agr2','Perp'}, 
        'neg': {'Ptprc','Vim','Col1a1','Cd14','Cd19','Pecam1'}
    },
    'Tumor': {
        'pos': {'Erbb2','Magea3','Ceacam5','Ceacam6','Tff1','Tff3','Agr3','Gata3','Foxa1','Mlph','Dsc2','Dsg2'}, 
        'neg': {'Ptprc','Vim','Col1a1','Cd14','Cd19','Lyz2'}
    }
}

# Helper Functions 

def _save_and_encode_fig(fig, path):
    """Saves figure to path with 500 DPI and returns base64 string."""
    fig.savefig(path, format='png', bbox_inches='tight', dpi=500, facecolor=fig.get_facecolor())
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def _upper_panels(panels):
    return {k: {'pos': {g.upper() for g in v['pos']}, 'neg': {g.upper() for g in v['neg']}} for k, v in panels.items()}

def _annotate_cluster(genes_in_cluster, markers_dict):
    if markers_dict is None: return {}
    gset = {str(g).upper() for g in genes_in_cluster}
    notes = {}
    for name, panel in markers_dict.items():
        pos_set = gset & panel['pos']
        neg_set = gset & panel['neg'] if panel['neg'] else set()
        pos_n, neg_n = len(pos_set), len(neg_set)
        
        if pos_n < 3:
            pos_n = 0
            verdict = 'no markers'
        else:
            if pos_n > 0 and neg_n == 0: verdict = 'pure'
            elif pos_n > 0 and neg_n > 0: verdict = 'mixed'
            else: verdict = 'no markers'
            
        txt = f"{verdict} (pos:{pos_n}"
        if pos_n > 0: txt += f" ({', '.join(sorted(pos_set))})"
        txt += f", neg:{neg_n}"
        if neg_n > 0: txt += f" ({', '.join(sorted(neg_set))})"
        txt += ")"
        notes[name] = txt
    return notes

def _get_short_annotation(genes_in_cluster, markers_dict):
    if markers_dict is None: return "Unknown"
    gset = {str(g).upper() for g in genes_in_cluster}
    candidates = []
    for name, panel in markers_dict.items():
        pos_set = gset & panel['pos']
        neg_set = gset & panel['neg'] if panel['neg'] else set()
        if len(pos_set) >= 3 and len(neg_set) == 0:
            candidates.append((name, len(pos_set)))
    if not candidates: return "Unassigned"
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def _first_peak_center(x, bins=80, sigma=1.5, rel_height=0.05):
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0: return None, None, None, None
    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(counts.astype(float), sigma=sigma)
    peaks, _ = find_peaks(smooth)
    if len(smooth) > 1 and smooth[0] > smooth[1] and smooth[0] > smooth[2]:
        if len(peaks) == 0 or peaks[0] != 0: peaks = np.insert(peaks, 0, 0)
    if len(peaks) == 0: return None, None, centers, smooth
    max_h = smooth[peaks].max()
    strong = [p for p in peaks if smooth[p] >= rel_height * max_h]
    if not strong: return None, None, centers, smooth
    p1 = strong[0]
    return float(centers[p1]), int(p1), centers, smooth

def gaussian_noise_threshold_1peak(x, bins=80, smooth_sigma=1.5, k=2.0, rel_height=0.05, max_quantile=0.990):
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    if x.size == 0: return 0.0
    fallback = float(np.quantile(x, 0.975)) if len(x) > 0 else 0.0
    m1, p1, centers, smooth = _first_peak_center(x, bins, smooth_sigma, rel_height)
    if m1 is None or centers is None: return fallback
    bin_width = centers[1] - centers[0] if len(centers) > 1 else 1.0
    is_zero_peak = (m1 <= 1.5 * bin_width) or (p1 == 0)
    if is_zero_peak:
        non_zeros = x[x > 0.1]
        if len(non_zeros) < 10: return fallback
        limit_idx = int(len(non_zeros) * 0.5)
        lower_tail = np.sort(non_zeros)[:limit_idx]
        if len(lower_tail) == 0: return fallback
        mu, sd = 0, np.std(lower_tail)
        thr = mu + (k + 1.0) * sd 
        thr = max(thr, np.percentile(non_zeros, 10)) 
    else:
        thr_gauss = 2.0 * m1
        left_noise = x[x <= m1]
        if left_noise.size >= 5 and left_noise.std(ddof=1) > 0:
            thr = max(thr_gauss, left_noise.mean() + k * left_noise.std(ddof=1))
        else:
            thr = thr_gauss
    return float(min(max(0.0, thr), np.quantile(x, max_quantile)))

def _generate_hex_offsets(max_points, radius_scale=1.0):
    indices = np.arange(0, max_points) + 0.5
    r = np.sqrt(indices / max_points) * radius_scale
    theta = np.pi * (1 + 5**0.5) * indices
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return list(zip(x, y))

# Main CoDeconv Class

class CoDeconvWorkflow:
    def __init__(self, data_path: str, species: str = 'hs', min_umi: int = 20,
                 image_rotate: int = 0, image_flip: str = 'none',
                 custom_markers: dict = None, deconvolution_mode: str = 'immune',
                 use_hpa: bool = False):
        
        self.data_path = data_path
        self.species = species
        self.min_umi = min_umi
        self.image_rotate = image_rotate
        self.image_flip = image_flip
        self.custom_markers = custom_markers
        self.deconvolution_mode = deconvolution_mode
        self.use_hpa = use_hpa # New Flag
        
        self.visium = None
        self.spatial = None
        self.scales = None
        self.filtered_visium = None
        self.df_clusters = None
        self.mapped_clusters = None
        self.hpa_name_map = {} # Store explicit HPA names
        self.module_counts_df = None
        self.thresholds = None
        self.cluster_divisors = None
        self.putative_meta = None
        self.markers_used = None
        self.cell_counts_summary = None
        self.cluster_labels = {}
        
        self.figures = {}
        self.stats = {}
        
        self.figures_dir = os.path.join(self.data_path, "Figures-QC")
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    import pandas as pd
    import numpy as np
    import scipy.io as io
    import json
    import time
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import cdist
    import gc

    def step1_load_and_filter(self):
        """
        Executes the primary data ingestion pipeline for 10x Genomics Visium data.
        
        Methodology:
        1. Ingests raw sparse matrices and feature definitions.
        2. Heuristically determines feature column indices (Ensembl ID vs. Gene Symbol).
        3. Loads spatial tissue coordinates and aligns them via affine transformations 
           (flip/rotate) to match histological image orientation.
        4. Performs Quality Control (QC) by filtering mitochondrial/ribosomal artifacts 
           and low-coverage genes based on UMI counts.
        """
        print("Step 1: Loading & Filtering")
        start = time.time()
        try:
            # Data Ingestion & Schema Validation
            # Extract barcodes and feature definitions from compressed TSV files.
            barcodes = pd.read_csv(f"{self.data_path}/filtered_feature_bc_matrix/barcodes.tsv.gz", header=None, sep='\t').iloc[:, 0]
            features_path = f"{self.data_path}/filtered_feature_bc_matrix/features.tsv.gz"
            features_df = pd.read_csv(features_path, header=None, sep='\t')
            
            # Heuristic detection of gene identifier columns. 
            # Differentiates between Ensembl IDs (ENSG...) and HGNC symbols based on string patterns.
            col_idx = 0
            if features_df.shape[1] >= 2:
                sample_col0 = features_df.iloc[:50, 0].astype(str).str.upper()
                sample_col1 = features_df.iloc[:50, 1].astype(str)
                is_ensembl = sample_col0.str.startswith("ENS").mean() > 0.8
                unique_types = set(sample_col1.unique())
                # specific check for multi-modal data (e.g., Gene Expression vs Antibody Capture)
                is_feature_type = "Gene Expression" in unique_types or len(unique_types) < 3
                if is_ensembl and not is_feature_type: col_idx = 1
            features = features_df.iloc[:, col_idx]
            
            # Matrix instantiation: Convert Market Matrix (MTX) format to dense array.
            # Note: High memory overhead operation.
            matrix = io.mmread(f"{self.data_path}/filtered_feature_bc_matrix/matrix.mtx.gz")
            print("Converting to dense matrix...")
            dense_matrix = matrix.toarray()
            
            self.visium = pd.DataFrame(dense_matrix, index=features, columns=barcodes)
            self.visium.index.name = 'Gene'

            # Spatial Coordinate Mapping
            # Load tissue positions and scaling factors for mapping spots to the high-res image.
            self.spatial = pd.read_csv(f"{self.data_path}/spatial/tissue_positions.csv").set_index("barcode")
            with open(f"{self.data_path}/spatial/scalefactors_json.json", "r") as f: self.scales = json.load(f)
            
            # Geometric Transformation
            # Apply affine transformations to align spatial coordinates with the image orientation.
            # Transforms global coordinates based on user-defined flip/rotate parameters.
            if self.image_flip == 'horizontal': self.spatial['pxl_col_in_fullres'] = self.spatial['pxl_col_in_fullres'].max() - self.spatial['pxl_col_in_fullres']
            elif self.image_flip == 'vertical': self.spatial['pxl_row_in_fullres'] = self.spatial['pxl_row_in_fullres'].max() - self.spatial['pxl_row_in_fullres']
            if self.image_rotate == 90:
                old_x = self.spatial['pxl_col_in_fullres'].copy()
                self.spatial['pxl_col_in_fullres'] = self.spatial['pxl_row_in_fullres']
                self.spatial['pxl_row_in_fullres'] = old_x.max() - old_x
            elif self.image_rotate == 180:
                self.spatial['pxl_col_in_fullres'] = self.spatial['pxl_col_in_fullres'].max() - self.spatial['pxl_col_in_fullres']
                self.spatial['pxl_row_in_fullres'] = self.spatial['pxl_row_in_fullres'].max() - self.spatial['pxl_row_in_fullres']
            elif self.image_rotate == 270:
                old_x = self.spatial['pxl_col_in_fullres'].copy()
                self.spatial['pxl_col_in_fullres'] = self.spatial['pxl_row_in_fullres'].max() - self.spatial['pxl_row_in_fullres']
                self.spatial['pxl_row_in_fullres'] = old_x.max() - old_x
        
            # Data Intersection: Ensure consistency between expression matrix and spatial metadata.
            common = self.visium.columns.intersection(self.spatial.index)
            self.visium = self.visium[common]
            self.spatial = self.spatial.loc[common]

        except Exception as e: raise RuntimeError(f"Data load failed: {e}")

        # Quality Control & Filtering
        # Calculate total Unique Molecular Identifier (UMI) counts per gene.
        umi_sum = self.visium.sum(axis=1)
        
        # Artifact exclusion: Create boolean mask for mitochondrial (MT), ribosomal (RP), 
        # and non-coding RNA (LINC/MIR) genes which often represent technical noise or lysis artifacts.
        mask_noise = self.visium.index.str.startswith(('MT-','RP','LINC','MIR','mt-','Rp','Linc','Mir'))
        
        # Selection logic: Retain genes satisfying minimum UMI threshold and passing artifact filter.
        keep_genes = (umi_sum >= self.min_umi) & (~mask_noise)
        self.filtered_visium = self.visium.loc[keep_genes]
        
        # Distribution Visualization
        # Generate histogram of Log-transformed UMI counts to verify power-law distribution.
        with plt.style.context('dark_background'):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(np.log1p(umi_sum[keep_genes]), bins=50, ax=ax, color='#088F8F', edgecolor='white')
            ax.set_xlabel("Log1p(Total UMI per Gene)")
            ax.set_title(f"Gene Expression Distribution\nRetained Genes: {keep_genes.sum()}")
            self.figures['gene_qc'] = _save_and_encode_fig(fig, os.path.join(self.figures_dir, "1_Gene_QC.png"))
        
        # Garbage collection to release memory allocated to the dense raw matrix.
        del self.visium
        gc.collect()
        print(f"Genes retained: {self.filtered_visium.shape[0]}")
        print(f"Step 1 Done in {(time.time()-start):.2f}s")
        return fig

    def _apply_watershed_split(self, labels, embedding, gene_names):
        """
        Refines clustering results by resolving mixed-phenotype clusters using a 
        marker-guided geometric splitting approach (analogous to Watershed segmentation).
        
        Algorithm:
        1. Identifies "mixed" clusters containing markers for distinct biological entities.
        2. Calculates robust centroids for each marker sub-population within the embedding space.
        3. Re-assigns genes to the nearest sub-population centroid via Euclidean distance minimization.
        
        Args:
            labels (array): Current cluster assignments.
            embedding (array): Coordinates of genes in the manifold (e.g., UMAP/PCA).
            gene_names (array): Array of gene identifiers corresponding to embedding rows.
            
        Returns:
            new_labels (array): Refined cluster assignments.
        """
        print("Refining clusters (Watershed split for mixed markers)...")
        new_labels = labels.copy().astype(object)
        
        unique_clusters = sorted(list(set(labels)))
        
        for cid in unique_clusters:
            # Boolean masking to isolate current cluster members
            cluster_mask = (labels == cid)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0: continue
            
            # Marker gene identification
            # Scan cluster members against pre-defined marker panels (self.markers_used).
            present_markers = {} 
            for i in cluster_indices:
                gene = gene_names[i]
                for m_type, panel in self.markers_used.items():
                    if gene.upper() in panel['pos']:
                        present_markers.setdefault(m_type, []).append(i)
                        break
            
            # Validation: Require a minimum cardinality (n>=3) to define a valid sub-population.
            valid_marker_types = {m_type: indices for m_type, indices in present_markers.items() if len(indices) >= 3}
            
            # Geometric Splitting Logic
            # Trigger split only if multiple distinct marker types co-exist in the cluster.
            if len(valid_marker_types) > 1:
                print(f"  Cluster {cid} is mixed: {list(valid_marker_types.keys())}. Splitting...")
                
                centroids = []
                type_labels = []
                
                for m_type, indices in valid_marker_types.items():
                    coords = embedding[indices]
                    
                    # Robust Centroid Calculation:
                    # 1. Compute geometric median.
                    # 2. Compute Median Absolute Deviation (MAD) of distances to median.
                    # 3. Filter outliers (> 2.5 MAD) to prevent skewed centroids.
                    median = np.median(coords, axis=0)
                    dists = np.linalg.norm(coords - median, axis=1)
                    mad = np.median(np.abs(dists - np.median(dists)))
                    if mad == 0: mad = 1e-6
                    
                    clean_mask = dists < (np.median(dists) + 2.5 * mad)
                    if np.sum(clean_mask) == 0: clean_mask = np.ones(len(dists), dtype=bool)
                    
                    centroid = np.mean(coords[clean_mask], axis=0)
                    centroids.append(centroid)
                    type_labels.append(m_type)
                
                centroids = np.array(centroids)
                
                # Nearest Neighbor Re-assignment
                # Assign every gene in the original cluster to the nearest new centroid.
                # Metric: Euclidean distance in embedding space.
                cluster_gene_coords = embedding[cluster_indices]
                dists = cdist(cluster_gene_coords, centroids, metric='euclidean')
                nearest_centroid_idx = np.argmin(dists, axis=1)
                
                # Update labels with composite ID (OriginalCluster_SubtypeIndex).
                for local_i, type_idx in enumerate(nearest_centroid_idx):
                    global_i = cluster_indices[local_i]
                    new_labels[global_i] = f"{cid}_{type_idx}"
                    
        return new_labels

    def step2_optimize_clustering(self, n_components: int = 50,
                                  resolution: float = None,
                                  manual_params: dict = None,
                                  leiden_resolution_range: tuple = (0.1, 3.0), 
                                  manual_merge: dict = None):
        """
        Step 2: Co-Expression Clustering & Dimensionality Reduction
        
        This stage identifies groups of co-expressed genes (gene modules) which likely 
        represent distinct cell types or biological states.
        
        Workflow:
        1. Dimensionality Reduction: Uses FastICA to decompose the high-dimensional 
           expression matrix into independent components.
        2. Manifold Learning: Constructs a UMAP embedding to approximate the 
           topological structure of the gene expression space.
        3. Community Detection: Applies the Leiden algorithm to a k-Nearest Neighbors (kNN) 
           graph, sweeping resolutions to optimize cluster stability and biological purity.

        Optional HPA Mode: Direct mapping of genes to Human Protein Atlas (HPA) 
           consensus clusters, bypassing de novo learning.
        """
        print(f"Step 2: Clustering Optimization (HPA Mode: {self.use_hpa})")
        if self.filtered_visium is None: raise ValueError("Run Step 1 first.")
        start = time.time()
        
        # Marker Panel Initialization
        # Loads species-specific marker sets (Human/Mouse) or custom user-defined panels
        # to guide the semi-supervised clustering evaluation.
        builtins = _upper_panels(MARKERS_HS) if self.species == 'hs' else _upper_panels(MARKERS_MM)
        custom_norm = _upper_panels(self.custom_markers) if self.custom_markers else None
        
        if self.deconvolution_mode == 'immune': self.markers_used = dict(builtins)
        elif self.deconvolution_mode == 'custom': self.markers_used = custom_norm
        elif self.deconvolution_mode == 'immune+custom':
            self.markers_used = dict(builtins)
            if custom_norm:
                for k, v in custom_norm.items():
                    if k in self.markers_used:
                        self.markers_used[k]['pos'] |= v['pos']
                        self.markers_used[k]['neg'] |= v['neg']
                    else: self.markers_used[k] = v
        
        gene_names = self.filtered_visium.index
        
        # HPA MODE: Knowledge-Based Mapping
        # If enabled, skips statistical clustering and maps genes directly to 
        # biologically validated clusters from the Human Protein Atlas.
        if self.use_hpa and self.species == 'hs':
            print("HPA Mode Active: Mapping genes directly to Human Protein Atlas clusters...")
            try:
                # 1. Try to import the Tumor-Specific Dictionary first
                import hpa_markers_tumor
                HPA_SOURCE = hpa_markers_tumor.MARKERS_HS
                print("Success: Loaded tumor-specific markers from hpa_markers_tumor.py")

            except ImportError:
                try:
                    # 2. Fallback to the standard HPA dictionary
                    import hpa_markers
                    HPA_SOURCE = hpa_markers.MARKERS_HS
                    print("Warning: hpa_markers_tumor.py not found. Loaded standard markers from hpa_markers.py")
                    
                except ImportError:
                    # 3. Final Fallback (e.g. if running in an environment without external files)
                    print("Warning: No external marker files found. Using empty/internal fallback.")
                    HPA_SOURCE = {} # Or your local MARKERS_HS if defined in this script
            
            self.mapped_clusters = {}
            self.hpa_name_map = {}
            valid_gene_set = set(self.filtered_visium.index)
            
            c_idx = 1
            # Sort keys to ensure deterministic ordering (e.g. alphanumeric)
            for cluster_name in sorted(HPA_SOURCE.keys()):
                hpa_genes = set(HPA_SOURCE[cluster_name]['pos'])
                # Intersection: HPA genes present in Visium data
                overlap = list(hpa_genes.intersection(valid_gene_set))
                
                if len(overlap) > 0:
                    self.mapped_clusters[c_idx] = sorted(overlap)
                    self.hpa_name_map[c_idx] = cluster_name
                    c_idx += 1
            
            print(f"Mapped {len(self.mapped_clusters)} HPA clusters containing genes found in dataset.")
            
            # Create Dummy Figures for Step 5 Compatibility (Pipeline integrity)
            with plt.style.context('dark_background'):
                fig_dum, ax_dum = plt.subplots(figsize=(6, 4))
                ax_dum.text(0.5, 0.5, "Skipped: HPA Mode Active\n(Pre-defined Clusters Used)", 
                           ha='center', va='center', color='white')
                self.figures['knee_plot'] = _save_and_encode_fig(fig_dum, os.path.join(self.figures_dir, "2a_Resolution_Sweep.png"))
                
                fig_dum2, ax_dum2 = plt.subplots(figsize=(6, 4))
                ax_dum2.text(0.5, 0.5, "Skipped: HPA Mode Active\n(No Dimensionality Reduction)", 
                           ha='center', va='center', color='white')
                self.figures['gene_umap'] = _save_and_encode_fig(fig_dum2, os.path.join(self.figures_dir, "2b_Gene_UMAP.png"))
            
            # Generate Heatmap (Visualization of mapped clusters)
            heatmap_data = []
            cluster_ids = sorted(self.mapped_clusters.keys())
            
            print("Generating HPA Cluster Heatmap...")
            # Map HPA cluster content back to standard immune markers for cross-verification.
            panel_names = sorted(self.markers_used.keys())
            for cid in cluster_ids:
                genes_upper = {g.upper() for g in self.mapped_clusters[cid]}
                row = []
                for panel in panel_names:
                    target_pos = self.markers_used[panel]['pos']
                    intersect = len(genes_upper.intersection(target_pos))
                    denom = max(len(target_pos), 1)
                    row.append((intersect/denom)*100)
                heatmap_data.append(row)
            
            df_heatmap = pd.DataFrame(heatmap_data, index=[self.hpa_name_map[c] for c in cluster_ids], columns=panel_names)
            
            with plt.style.context('dark_background'):
                if df_heatmap.size > 4 and not (df_heatmap.values == 0).all():
                    cg = sns.clustermap(df_heatmap, cmap='viridis', figsize=(12, 10), row_cluster=False, col_cluster=False)
                    self.figures['marker_heatmap'] = _save_and_encode_fig(cg.fig, os.path.join(self.figures_dir, "2c_Marker_Heatmap.png"))
                else:
                    fig_h, ax_h = plt.subplots()
                    self.figures['marker_heatmap'] = _save_and_encode_fig(fig_h, os.path.join(self.figures_dir, "2c_Marker_Heatmap.png"))

            # Save HPA Mapping to Excel
            df_rows = []
            for cid in sorted(self.mapped_clusters.keys()):
                df_rows.append({'Cluster_ID': cid, 'HPA_Name': self.hpa_name_map.get(cid, "Unknown"), 'Genes': self.mapped_clusters[cid]})
            self.df_clusters = pd.DataFrame(df_rows).set_index('Cluster_ID')
            self.df_clusters.to_excel(f"{self.data_path}/CoExpression_Clusters.xlsx")

            print(f"Step 2 (HPA) Done in {(time.time()-start):.2f}s")
            return None

        # ML MODE: De Novo Clustering

        # 1. FastICA (Independent Component Analysis)
        # Unlike PCA, which maximizes variance, ICA maximizes non-Gaussianity to separate 
        # independent sources (biological signals) from mixed observations.
        print(f"Preprocessing & FastICA (n={n_components})...")
        
        
        model = FastICA(n_components=n_components, random_state=42, max_iter=500)
        X = self.filtered_visium.values
        X_log = np.log1p(X)
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X_log)
        X_reduced = model.fit_transform(X_scaled)
        
        # 2. UMAP & k-Nearest Neighbors Graph
        # UMAP constructs a high-dimensional fuzzy simplicial set and optimizes a low-dimensional 
        # layout (embedding) to preserve local connectivity.
        print(f"Running UMAP & Building Graph...")
        

        u_neighbors = 30
        u_dist = 0.1
        if manual_params:
            u_neighbors = manual_params.get('n_neighbors', 30)
            u_dist = manual_params.get('min_dist', 0.1)
            
        umap_model = umap.UMAP(n_neighbors=u_neighbors, min_dist=u_dist, metric='cosine', random_state=42)
        embedding = umap_model.fit_transform(X_reduced)
        
        # Construct the adjacency matrix (A) for community detection
        A = kneighbors_graph(embedding, u_neighbors, mode='connectivity', include_self=True)
        sources, targets = A.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        g = ig.Graph(edgelist)
        
        # 3. Leiden Clustering Optimization
        # Sweeps through resolution parameters to detect communities.
        # Higher resolution = smaller, more granular clusters.
        # Lower resolution = larger, broader clusters.
        best_labels = None
        best_param = 0.0
        
        # Optimization Metrics Storage
        res_list = []
        n_clust_list = []
        score_list = []
        
        if resolution is not None:
            # Fixed resolution (Manual Override)
            print(f"Running Leiden (Fixed resolution={resolution})...")
            partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution, seed=42)
            best_labels = np.array(partition.membership)
            best_param = resolution
            
        else:
            # Auto Sweep: Optimize modularity
            print(f"Sweeping Leiden resolution {leiden_resolution_range}...")
            

            best_score = -np.inf
            res_values = np.linspace(leiden_resolution_range[0], leiden_resolution_range[1], 15)
            
            # Map genes to marker indices for guided scoring
            marker_indices = []
            marker_true_labels = []
            
            for m_name, panel in self.markers_used.items():
                for gene_name in panel['pos']:
                    if gene_name in gene_names:
                        idx = gene_names.get_loc(gene_name)
                        marker_indices.append(idx)
                        marker_true_labels.append(m_name)
            
            if not marker_indices:
                print("Warning: No marker genes found. Falling back to default resolution.")
                res_values = [0.5]
            
            for res in res_values:
                partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=42)
                labels = np.array(partition.membership)
                n_clusters = len(set(labels))
                
                # Filter trivial solutions (too few or too many clusters)
                if n_clusters < 3 or n_clusters > 200: 
                    score = -100
                else:
                    # Guided Scoring Function:
                    # Maximizes homogeneity (purity of markers in clusters)
                    # Minimizes fragmentation (penalizes excessive cluster counts)
                    marker_pred_labels = labels[marker_indices]
                    h_score = homogeneity_score(marker_true_labels, marker_pred_labels)
                    fragmentation_penalty = n_clusters * 0.005
                    score = h_score - fragmentation_penalty
                
                res_list.append(res)
                n_clust_list.append(n_clusters)
                score_list.append(score)

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_param = res
            
            print(f"Best Clustering: resolution={best_param:.2f} -> {len(set(best_labels))} Clusters")

        # 4. Refinement (Watershed Split)
        # Resolves mixed clusters by geometric splitting in the UMAP space.
        final_labels = self._apply_watershed_split(best_labels, embedding, gene_names)

        # 5. Finalizing Mapping & Renumbering
        self.mapped_clusters = {}
        unique_final_labels = sorted(list(set(final_labels)), key=lambda x: str(x))
        label_map = {l: i+1 for i, l in enumerate(unique_final_labels)}
        
        for idx, lbl in enumerate(final_labels):
            new_id = label_map[lbl]
            gene = gene_names[idx]
            self.mapped_clusters.setdefault(new_id, []).append(gene)

        # Manual Merge Logic (User Intervention)
        if manual_merge:
            print("Applying manual cluster merges...")
            for target_id, source_ids in manual_merge.items():
                if target_id not in self.mapped_clusters: continue
                if not isinstance(source_ids, list): source_ids = [source_ids]
                for src_id in source_ids:
                    if src_id not in self.mapped_clusters: continue
                    if src_id == target_id: continue
                    self.mapped_clusters[target_id].extend(self.mapped_clusters[src_id])
                    self.mapped_clusters[target_id] = list(set(self.mapped_clusters[target_id]))
                    del self.mapped_clusters[src_id]
                    print(f"Merged C{src_id} into C{target_id}.")

        # Re-index clusters sequentially after merges
        final_mapping = {}
        for new_i, (old_i, genes) in enumerate(sorted(self.mapped_clusters.items()), 1):
            final_mapping[new_i] = genes
        self.mapped_clusters = final_mapping

        # Reporting & Visualization
        
        # Save Gene Modules to Excel
        df_rows = []
        for cid in sorted(self.mapped_clusters.keys()):
            df_rows.append({'Cluster': cid, 'Genes': self.mapped_clusters[cid]})
        self.df_clusters = pd.DataFrame(df_rows).set_index('Cluster')
        self.df_clusters.to_excel(f"{self.data_path}/CoExpression_Clusters.xlsx")
        
        # Annotate clusters with likely cell identities based on marker overlap
        purity_rows = []
        panel_order = list(self.markers_used.keys())
        for cid in self.df_clusters.index:
            genes = self.df_clusters.loc[cid, 'Genes']
            notes = _annotate_cluster(genes, self.markers_used)
            row = {'Cluster': cid}
            for p in panel_order: row[p] = notes.get(p, "")
            purity_rows.append(row)
        pd.DataFrame(purity_rows).set_index('Cluster').to_excel(f"{self.data_path}/Cluster_Purity_Report.xlsx")

        # PLOTTING
        with plt.style.context('dark_background'):
            # 1. Knee Plot (Resolution Sweep)
            # Visualizes the trade-off between cluster number and optimization score.
            if resolution is None and len(res_list) > 0:
                fig_knee, ax1 = plt.subplots(figsize=(10, 6))
                ax1.set_xlabel('Resolution')
                ax1.set_ylabel('N Clusters', color='#00CCFF')
                ax1.plot(res_list, n_clust_list, color='#00CCFF', linewidth=2, marker='o', label='N Clusters')
                ax1.tick_params(axis='y', labelcolor='#00CCFF')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel('Score (Homogeneity - Penalty)', color='#FF9900')
                ax2.plot(res_list, score_list, color='#FF9900', linewidth=2, linestyle='--', label='Optimization Score')
                ax2.tick_params(axis='y', labelcolor='#FF9900')
                
                ax1.axvline(best_param, color='white', linestyle=':', label=f'Selected res={best_param:.2f}')
                plt.title("Leiden Resolution Optimization")
                self.figures['knee_plot'] = _save_and_encode_fig(fig_knee, os.path.join(self.figures_dir, "2a_Resolution_Sweep.png"))
                display(fig_knee)
            else:
                # Create placeholder if fixed resolution
                fig_dummy, ax_d = plt.subplots()
                ax_d.text(0.5, 0.5, f"Manual Resolution {resolution} Used. No Sweep.", ha='center', color='white')
                self.figures['knee_plot'] = _save_and_encode_fig(fig_dummy, os.path.join(self.figures_dir, "2a_Resolution_Sweep.png"))

            # 2. UMAP Visualization
            # Projects the graph embedding into 2D space.
            plot_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=gene_names)
            gene_to_cluster = {}
            for cid, g_list in self.mapped_clusters.items():
                for g in g_list: gene_to_cluster[g] = str(cid)
            plot_df['Cluster'] = plot_df.index.map(gene_to_cluster).fillna('Noise')
            
            plot_df['Marker_Type'] = 'None'
            for m_name, panel in self.markers_used.items():
                for g in plot_df.index:
                    if g.upper() in panel['pos']:
                        plot_df.at[g, 'Marker_Type'] = m_name

            # Standard Landscape Figure Size
            fig_umap, ax_u = plt.subplots(figsize=(14, 10)) 
            
            unique_clusters = sorted(plot_df['Cluster'].unique(), key=lambda x: int(x) if x != 'Noise' else -1)
            cluster_palette = sns.color_palette("husl", len(unique_clusters))
            cluster_color_map = {cls: cluster_palette[i] for i, cls in enumerate(unique_clusters)}
            if 'Noise' in cluster_color_map: cluster_color_map['Noise'] = '#333333'

            # Background points (Unmarked genes)
            sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='Cluster', 
                            palette=cluster_color_map, s=15, alpha=0.3, linewidth=0, 
                            legend=False, ax=ax_u)

            markers_only = plot_df[plot_df['Marker_Type'] != 'None'].copy()
            
            if not markers_only.empty:
                # Plot markers on top with specific shapes
                sns.scatterplot(data=markers_only, x='UMAP1', y='UMAP2', 
                                hue='Cluster', style='Marker_Type',
                                palette=cluster_color_map, 
                                s=50, edgecolor='white', linewidth=1.0, ax=ax_u, legend=False)
                
                # Standard Legend (Markers Only) - Reset to defaults but anchored outside
                # We let seaborn generate the style handles for us from a dummy plot
                dummy_fig, dummy_ax = plt.subplots()
                sns.scatterplot(data=markers_only, x='UMAP1', y='UMAP2', style='Marker_Type', s=50, ax=dummy_ax, color='white')
                h, l = dummy_ax.get_legend_handles_labels()
                plt.close(dummy_fig)
                
                # Standard Legend position outside right, default font size
                ax_u.legend(handles=h, labels=l, title="Cell Types", 
                           bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14, title_fontsize=16)

            else:
                ax_u.text(0.5, 0.5, "No marker genes found.", ha='center', color='white')
            
            # Reset aspect ratio to auto (not forced square) to avoid squashing
            ax_u.set_aspect('auto') 
            ax_u.set_title(f"Gene UMAP (Leiden res={best_param:.2f})", fontsize=18)
            plt.tight_layout()
            
            self.figures['gene_umap'] = _save_and_encode_fig(fig_umap, os.path.join(self.figures_dir, "2b_Gene_UMAP.png"))
            display(fig_umap)
            
            # 3. Expression Heatmap
            # Hierarchical clustering of the identified modules vs marker panels
            all_expressed_upper = {g.upper() for g in gene_names}
            heatmap_data = []
            cluster_ids = sorted(self.mapped_clusters.keys())
            panel_names = sorted(self.markers_used.keys())
            for cid in cluster_ids:
                genes = set(self.mapped_clusters[cid])
                genes_upper = {g.upper() for g in genes}
                row = []
                for panel in panel_names:
                    target_pos = self.markers_used[panel]['pos']
                    valid_markers_in_slide = target_pos.intersection(all_expressed_upper)
                    denom = max(len(valid_markers_in_slide), 5)
                    match_count = len(genes_upper.intersection(target_pos))
                    if match_count < 3:
                        pct = 0.0
                    else:
                        pct = (match_count / denom) * 100
                    row.append(pct)
                heatmap_data.append(row)
            df_heatmap = pd.DataFrame(heatmap_data, index=cluster_ids, columns=panel_names)
            
            if df_heatmap.size > 4 and not (df_heatmap.values == 0).all():
                cg = sns.clustermap(df_heatmap, cmap='viridis', metric='euclidean', method='ward',
                                    figsize=(12, 10), col_cluster=True, row_cluster=True)
                cg.fig.suptitle("", y=1.02, color='white') 
                cg.fig.patch.set_facecolor('black')
                plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), color='white')
                plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), color='white')
                self.figures['marker_heatmap'] = _save_and_encode_fig(cg.fig, os.path.join(self.figures_dir, "2c_Marker_Heatmap.png"))
                plt.close(cg.fig)
            else:
                fig_heat, ax_heat = plt.subplots(figsize=(12, 8))
                sns.heatmap(df_heatmap, annot=False, cmap='viridis', ax=ax_heat)
                self.figures['marker_heatmap'] = _save_and_encode_fig(fig_heat, os.path.join(self.figures_dir, "2c_Marker_Heatmap.png"))
                plt.close(fig_heat)

        del X, X_log, embedding
        gc.collect()
        print(f"Step 2 Done in {(time.time()-start):.2f}s")
        return None

    def step3_auto_thresholding(self, manual_override: bool = False, custom_thresholds: dict = None, 
                                plot_y_log2: bool = True,
                                mad_factor: float = 2.0,
                                signal_percentile: int = 90,
                                min_peak_umi: int = 50):
        """
        Step 3: Gene Module Auto-Thresholding
        
        Logic Overview:
        This method determines the active expression threshold for each gene module (cell type) 
        using a two-stage quality control process followed by Gaussian mixture modeling.
        
        1. GLOBAL NOISE CHARACTERIZATION (Statistical Baseline):
           - Aggregates density (UMI/Gene) distributions from all non-zero spots across the slide.
           - Calculates a robust 'Ambient Noise Floor' using the Median Absolute Deviation (MAD):
             Noise_Floor = Median_Global + (mad_factor * Sigma_Estimate).
             
        2. CLUSTER QUALITY CONTROL (Dual-Gating):
           - Gate 1 (Statistical Significance): The 90th percentile of the cluster's signal 
             distribution must exceed the Global Ambient Noise Floor.
           - Gate 2 (Biological Relevance): The 99th percentile (Peak Signal) must exceed 
             a minimum raw UMI count (min_peak_umi, e.g., 50), filtering out weak 'ghost' signals.
             
        3. ADAPTIVE THRESHOLDING (Gaussian Fit):
           - For clusters passing both gates, a Gaussian noise model is fitted to the specific 
             density distribution of that cluster to determine the precise signal-to-noise boundary.
           - This threshold is 'unshackled', meaning it is not artificially forced above the 
             global floor, preserving sensitivity for low-abundance but specific cell types.
        """
        if self.mapped_clusters is None: raise ValueError("Run Step 2 first.")
        start = time.time()
        
        print(f"1. Characterizing Global Ambient Noise (MAD={mad_factor})...")
        
        # 1. Global Statistics Aggregation
        sums = {}
        gene_counts = {}
        all_densities = []
        
        # Calculate raw densities per cluster per spot
        for cid, genes in self.mapped_clusters.items():
            valid_genes = [g for g in genes if g in self.filtered_visium.index]
            if valid_genes:
                # Sum raw UMIs for the module
                module_sum = self.filtered_visium.loc[valid_genes].sum(axis=0)
                sums[cid] = module_sum
                gene_counts[cid] = len(valid_genes)
                
                # Calculate density (UMI / Gene Count)
                # We collect all non-zero densities to model the background distribution
                density = module_sum.values / len(valid_genes)
                nonzero_density = density[density > 0]
                if len(nonzero_density) > 0:
                    all_densities.append(nonzero_density)
            else:
                gene_counts[cid] = 0
        
        self.module_counts_df = pd.DataFrame(sums).T
        self.module_counts_df.columns = self.module_counts_df.columns.astype(str)
        
        # 2. Robust Noise Floor Calculation
        if not all_densities: raise ValueError("No non-zero data found in dataset.")
        
        # Flatten all observations to form the global background distribution
        flat_background = np.concatenate(all_densities)
        
        # Robust Statistics: Median and MAD
        bg_median = np.median(flat_background)
        bg_mad = np.median(np.abs(flat_background - bg_median))
        
        # Estimate Sigma (Standard Deviation) from MAD for normal distribution consistency
        # Factor 1.4826 makes MAD consistent with SD for Gaussian data
        sigma_est = 1.4826 * bg_mad
        
        # Define the Ambient Noise Floor
        ambient_noise_floor = bg_median + (mad_factor * sigma_est)
        
        print(f" -> Global Background Statistics:")
        print(f"    - Median Density: {bg_median:.4f} UMI/Gene")
        print(f"    - Noise Floor (Median + {mad_factor}*MAD): {ambient_noise_floor:.4f} UMI/Gene")
        print(f"    - Minimum Biological Peak: {min_peak_umi} UMI (Raw)")
        
        # Visualization: Global Density Distribution
        with plt.style.context('dark_background'):
            fig_hist, ax_h = plt.subplots(figsize=(12, 4))
            # Filter extreme outliers (>P99.9) for clearer visualization of the noise tail
            viz_data = flat_background[flat_background < np.percentile(flat_background, 99.9)]
            
            sns.histplot(viz_data, bins=100, element="step", color="gray", ax=ax_h, label='Global Density Distribution')
            ax_h.axvline(ambient_noise_floor, color='#00FF00', linestyle='--', linewidth=2, label='Ambient Noise Floor')
            ax_h.axvline(bg_median, color='yellow', linestyle=':', label='Median')
            
            ax_h.set_title(f"Global Ambient Noise Characterization")
            ax_h.set_xlabel("Module Density (UMI / Gene)")
            ax_h.legend()
            display(fig_hist)
            self.figures['global_noise_dist'] = _save_and_encode_fig(fig_hist, os.path.join(self.figures_dir, "3a_Global_Noise_Dist.png"))

        # 3. Cluster-Specific Processing (Dual-Gating)
        print(f"3. Analyzing Clusters (QC Gating & Gaussian Fitting)...")
        
        tasks = {}
        self.thresholds = {}
        self.norm_thresholds = {}
        self.cluster_quality = {}
        self.cluster_meta = {}
        
        for cid in self.module_counts_df.index:
            n_genes = gene_counts[cid]
            if n_genes == 0: 
                self.cluster_quality[cid] = "Empty"
                continue
            
            # Extract Data
            raw_data = self.module_counts_df.loc[cid].values.astype(float)
            norm_data = raw_data / n_genes
            
            # GATE 1: STATISTICAL SIGNIFICANCE
            # Check if the 90th percentile of the signal exceeds the ambient noise floor.
            signal_p90 = np.percentile(norm_data, signal_percentile)
            
            if signal_p90 <= ambient_noise_floor:
                self.cluster_quality[cid] = "Noise (Ambient)"
                self.thresholds[cid] = 999999 # Effectively filters out all cells
                self.norm_thresholds[cid] = ambient_noise_floor # For visualization reference
                self.cluster_meta[cid] = {'n_genes': n_genes, 'p90': signal_p90, 'raw_p99': 0}
                continue 

            # GATE 2: BIOLOGICAL RELEVANCE
            # Check if the peak expression (P99) reaches a minimal biological count.
            # Prevents statistically distinct but biologically negligible "ghost" clusters.
            raw_p99 = np.percentile(raw_data, 99)
            
            if raw_p99 < min_peak_umi:
                self.cluster_quality[cid] = "Noise (Too Weak)"
                self.thresholds[cid] = 999999
                # Visualize the rejection threshold
                self.norm_thresholds[cid] = min_peak_umi / n_genes 
                self.cluster_meta[cid] = {'n_genes': n_genes, 'p90': signal_p90, 'raw_p99': raw_p99}
                continue

            # SURVIVORS: GAUSSIAN FITTING
            # If a cluster passes both gates, we fit a Gaussian model to find its specific noise boundary.
            tasks[cid] = delayed(gaussian_noise_threshold_1peak)(norm_data)
            self.cluster_quality[cid] = "Valid"
            self.cluster_meta[cid] = {'n_genes': n_genes, 'p90': signal_p90, 'raw_p99': raw_p99}

        # Execute parallel computation for Gaussian fits
        results = dask.compute(tasks)[0] if tasks else {}
        
        for cid, calculated_gauss_th in results.items():
            n_genes = gene_counts[cid]
            
            # Adaptive Thresholding:
            # We trust the Gaussian fit for Valid clusters. We do NOT force it to be above the global floor.
            # This preserves sensitivity for active but low-expressing modules.
            final_norm_th = calculated_gauss_th
            
            # Convert Density Threshold back to Raw Integer Threshold for filtering
            raw_th = int(np.ceil(final_norm_th * n_genes))
            
            # Safety Floor: Ensure we require at least 1 read to count a cell
            self.thresholds[cid] = max(raw_th, 1)
            self.norm_thresholds[cid] = final_norm_th

        # 4. Manual Overrides
        if manual_override and custom_thresholds:
            print("Applying manual threshold overrides...")
            for cid, val in custom_thresholds.items():
                target_val = int(val)
                key = cid if cid in self.thresholds else int(cid) if str(cid).isdigit() else None
                if key in self.thresholds:
                    self.thresholds[key] = target_val
                    # Recalculate normalized threshold for consistent plotting
                    self.norm_thresholds[key] = target_val / gene_counts.get(key, 1)
                    self.cluster_quality[key] = "Manual Override"

        # 5. Visualization (Knee Plots)
        print("Generating Diagnostic Knee Plots...")
        self.cluster_divisors = {}
        self.cluster_labels = {} 
        
        # Pre-populate labels dictionary (Critical for Step 4 stability)
        for cid in self.mapped_clusters:
            genes = self.mapped_clusters[cid]
            if self.hpa_name_map and cid in self.hpa_name_map:
                 label = f"C{cid} ({self.hpa_name_map[cid]})"
            else:
                 ann = _get_short_annotation(genes, self.markers_used)
                 label = f"C{cid} ({ann})" if ann != "Unassigned" else f"C{cid}"
            self.cluster_labels[cid] = label

        # 6. Putative Cell Count Estimation
        from scipy.stats import gaussian_kde

        for cid, th in self.thresholds.items():
            if cid not in self.module_counts_df.index: continue
            
            # Get raw counts for this module
            vals = self.module_counts_df.loc[cid].values
            
            # Filter: Analyze only the valid signal (above the noise threshold)
            signal = vals[vals > th]
            
            # CASE 0: No signal
            if len(signal) == 0:
                self.cluster_divisors[cid] = th + 1
                continue

            # CASE 1: Sparse data fallback (Not enough points for KDE)
            # If we have fewer than 10 spots, statistics are unreliable -> use Median
            if len(signal) < 10:
                self.cluster_divisors[cid] = np.median(signal)
                continue

            # CASE 2: Identical values
            # If variance is 0 (all spots have exactly same count), KDE fails -> use value
            if np.std(signal) == 0:
                self.cluster_divisors[cid] = signal[0]
                continue
                
            # CASE 3: KDE Mode Estimation
            try:
                kernel = gaussian_kde(signal)
                
                # Define search grid: from min to P99 (ignore extreme outliers)
                # We use 200 points for sufficient resolution
                grid_min = np.min(signal)
                grid_max = np.percentile(signal, 99)
                x_grid = np.linspace(grid_min, grid_max, 200)
                
                # Find the peak of the density function
                pdf = kernel(x_grid)
                mode_val = x_grid[np.argmax(pdf)]
                
                # SAFETY CHECK:
                # Sometimes KDE can peak at the very left edge if the distribution is cut off.
                # If the mode is suspiciously low (< 50% of median), likely an edge artifact.
                if mode_val < (np.median(signal) * 0.5):
                    self.cluster_divisors[cid] = np.median(signal)
                else:
                    self.cluster_divisors[cid] = mode_val
                    
            except Exception as e:
                # Fallback for any linear algebra errors in KDE
                print(f" -> Warning: KDE failed for Cluster {cid}, using Median. (Err: {e})")
                self.cluster_divisors[cid] = np.median(signal)

        # Generate Plots
        n_clusters = len(self.module_counts_df)
        if n_clusters > 0:
            cols = 8 
            rows = math.ceil(n_clusters / cols)
            plot_height = rows * 3.2
            
            with plt.style.context('dark_background'):
                fig_knee, axes = plt.subplots(rows, cols, figsize=(24, plot_height), constrained_layout=True)
                axes = axes.flatten() if n_clusters > 1 else [axes]
                
                for i, cid in enumerate(self.module_counts_df.index):
                    ax = axes[i]
                    quality = self.cluster_quality.get(cid, "Invalid")
                    
                    if quality == "Empty": 
                        ax.text(0.5, 0.5, "Empty Module", ha='center', color='gray')
                        continue

                    # Data Preparation
                    n_genes = gene_counts[cid]
                    norm_data = self.module_counts_df.loc[cid].values / n_genes
                    norm_th = self.norm_thresholds.get(cid, 0)
                    
                    sorted_data = np.sort(norm_data)[::-1]
                    sorted_data = sorted_data[sorted_data > 0]
                    
                    if len(sorted_data) == 0:
                        ax.text(0.5, 0.5, "No Signal", ha='center', color='gray')
                        continue

                    ranks = np.arange(1, len(sorted_data) + 1)
                    
                    # Color Logic
                    is_noise = "Noise" in quality
                    line_color = '#555555' if is_noise else 'orange'
                    title_color = 'red' if is_noise else 'white'
                    
                    # Plot Signal
                    mask_sig = sorted_data > norm_th
                    if np.any(~mask_sig): 
                        ax.plot(ranks[~mask_sig], sorted_data[~mask_sig], color='#555555', lw=2)
                    if np.any(mask_sig): 
                        ax.plot(ranks[mask_sig], sorted_data[mask_sig], color=line_color, lw=2)
                    
                    # Plot Threshold Lines
                    ax.axhline(norm_th, color='red' if is_noise else '#5D3FD3', linestyle='--', lw=1.5, label='Threshold')
                    
                    if not is_noise:
                        # Show Ambient Floor for context (Gray dotted line)
                        ax.axhline(ambient_noise_floor, color='gray', linestyle=':', lw=1, alpha=0.5, label='Ambient')

                    # Axis Scaling
                    ax.set_xscale('log')
                    if plot_y_log2:
                        ax.set_yscale('log', base=2)
                        y_min = sorted_data.min()
                        ax.set_ylim(bottom=max(0.01, y_min * 0.8))
                        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
                    else:
                        ax.set_yscale('linear')
                    
                    ax.set_xlim(left=0.7, right=len(sorted_data) * 1.5)
                    ax.grid(True, which="both", ls="-", alpha=0.2)
                    
                    # Metadata Title
                    hpa_label = self.cluster_labels.get(cid, f"C{cid}")
                    meta = self.cluster_meta.get(cid, {})
                    p90 = meta.get('p90', 0)
                    raw_p99 = meta.get('raw_p99', 0)
                    
                    stats_str = f"n(g):{n_genes} | P90:{p90:.2f}"
                    if quality == "Noise (Ambient)": 
                        stats_str += "\nRejection: Ambient RNA"
                    elif quality == "Noise (Too Weak)": 
                        stats_str += f"\nRejection: Weak Signal {raw_p99:.0f}<{min_peak_umi}"
                    else: 
                        stats_str += f" | Th:{norm_th:.2f}"
                    
                    ax.set_title(f"{hpa_label}\n{stats_str}", fontsize=9, color=title_color)
                    
                    # Show Cell Count
                    if not is_noise:
                        n_cells = np.sum(self.module_counts_df.loc[cid].values > self.thresholds[cid])
                        ax.text(0.05, 0.05, f"Cnt: {n_cells}", transform=ax.transAxes, 
                                color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

                for j in range(i+1, len(axes)): axes[j].axis('off')
                self.figures['thresholds'] = _save_and_encode_fig(fig_knee, os.path.join(self.figures_dir, "3_Knee_Plots.png"))

        # 6. Summary Report
        counts_data = []
        for cid, th in self.thresholds.items():
            if "Noise" in self.cluster_quality.get(cid, ""): continue
            
            mod_vals = self.module_counts_df.loc[cid]
            div = self.cluster_divisors.get(cid, 1.0)
            count = (mod_vals[mod_vals >= th] / div).fillna(0).apply(np.ceil).sum()
            label = self.cluster_labels.get(cid, f"C{cid}")
            
            if count > 0: counts_data.append({'Label': label, 'Count': count})
            
        counts_df = pd.DataFrame(counts_data).sort_values('Count', ascending=False) if counts_data else pd.DataFrame(columns=['Label', 'Count'])
        self.cell_counts_summary = counts_df
        
        with plt.style.context('dark_background'):
            fig_bar, ax_bar = plt.subplots(figsize=(16, 9))
            if not counts_df.empty:
                sns.barplot(data=counts_df, x='Label', y='Count', palette='viridis', ax=ax_bar)
                plt.xticks(rotation=45, ha='right')
            ax_bar.set_title(f"Predicted Cell Counts (Dual-Gate: MAD-{mad_factor}, MinPeak-{min_peak_umi})")
            self.figures['cell_counts'] = _save_and_encode_fig(fig_bar, os.path.join(self.figures_dir, "4_Predicted_Cell_Counts.png"))
            
        display(fig_knee)
        display(fig_bar)
        
        print(f"Step 3 Done in {(time.time()-start):.2f}s")
        return None

    def step4_deconvolve_and_plot(self):
        """
        Step 4: Spatial Allocation & Niche Discovery
        
        This module transforms spot-level gene expression into single-cell resolution estimates 
        and identifies higher-order tissue structures (Niches).

        Methodology:
        1.  Deconvolution: Converts UMI counts of gene modules into estimated cell counts 
            per spot using pre-defined thresholds and divisor constants (average transcripts/cell).
        2.  Geometric Packing: Uses a phyllotaxis (Golden Angle) spiral algorithm to 
            heuristically distribute estimated cells within the circular spot area. 
        3.  Niche Detection: Performs a neighborhood analysis (k-NN) to compute the 
            cellular composition surrounding each cell.
        4.  Community Clustering: Applies K-Means to these composition vectors to find 
            "Spatial Niches" — regions defined by recurring cellular neighborhoods rather 
            than just cell types.
            
        """
        print("Step 4: Allocation & Spatial Plot")
        if self.thresholds is None: raise ValueError("Run Step 3 first.")
        start = time.time()
        
        # 1. Deconvolution (UMI -> Cell Counts)
        # Filters module counts based on signal-to-noise thresholds derived in Step 3.
        deconv_filt = self.module_counts_df.copy()
        for cid, th in self.thresholds.items():
            deconv_filt.loc[cid] = deconv_filt.loc[cid].apply(lambda x: x if x >= th else 0)
        
        # Estimate cell cardinality by dividing total signal by expected signal per cell.
        # np.ceil ensures integer cell counts (cannot have 0.5 cells).
        cluster_mins = pd.Series(self.cluster_divisors)
        cell_counts = deconv_filt.div(cluster_mins, axis=0).apply(np.ceil).fillna(0).astype(int)
        
        # 2. Geometric Cell Allocation (Phyllotaxis)
        # Pre-compute spatial lookup table for performance.
        spot_radius = math.ceil(self.scales["spot_diameter_fullres"])
        spatial_map = self.spatial[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_dict('index')
        
        putative_meta = []
        print("Allocating cells spatially...")
        
        for spot_bc in cell_counts.columns:
            if spot_bc not in spatial_map: continue
            
            # Extract non-zero cell types for this spot
            spot_cells = cell_counts[spot_bc]
            spot_cells = spot_cells[spot_cells > 0]
            if spot_cells.empty: continue
            total_cells = spot_cells.sum()
            
            # Retrieve spot centroid
            cx, cy = spatial_map[spot_bc]['pxl_col_in_fullres'], spatial_map[spot_bc]['pxl_row_in_fullres']
            
            # Algorithm: Vogel's Model for Fermat's Spiral (Golden Angle Packing)
            # Distributes points uniformly in a circle to minimize overlap.
            # theta = n * 137.508 degrees (Golden Angle)
            indices = np.arange(0, total_cells) + 0.5
            r = np.sqrt(indices / total_cells) * spot_radius
            theta = np.pi * (1 + 5**0.5) * indices 
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            points = list(zip(cx + dx, cy + dy))

            # Assign identities to the generated points
            pt_idx = 0
            for cid, count in spot_cells.items():
                for k in range(1, count+1):
                    px, py = points[pt_idx]
                    pt_idx += 1
                    putative_meta.append({
                        'barcode': f"{spot_bc}_{cid}_{k}", # Synthetic unique ID
                        'spot': spot_bc,
                        'cluster': cid,
                        'center_x': px, 
                        'center_y': py
                    })
        
        self.putative_meta = pd.DataFrame(putative_meta).set_index('barcode')
        self.putative_meta['predicted.cell.type'] = self.putative_meta['cluster'].map(self.cluster_labels)

        # 3. Spatial Niche Discovery
        print("Detecting spatial communities...")
        
        # Build KD-Tree for efficient Nearest Neighbor lookup (k=200)
        coords = self.putative_meta[['center_x', 'center_y']].values
        tree = cKDTree(coords)
        dist, idx = tree.query(coords, k=200)
        
        unique_clusters = sorted(self.putative_meta['cluster'].unique())
        c_to_i = {c: i for i, c in enumerate(unique_clusters)}
        n_cell_types = len(unique_clusters)
        n_cells = len(self.putative_meta)
        
        # Vectorize neighbor types: Map neighbor indices to their cell types
        cluster_ids = self.putative_meta['cluster'].map(c_to_i).values
        neighbor_types = cluster_ids[idx] 
        
        # Calculate Composition Vectors: Fraction of each cell type in the local neighborhood
        neighbor_composition = np.zeros((n_cells, n_cell_types))
        for i in range(n_cells):
            counts = np.bincount(neighbor_types[i], minlength=n_cell_types)
            neighbor_composition[i] = counts / 200.0
            
        # Feature Smoothing: Average the composition vectors of neighbors to reduce noise
        smoothed_comp = np.zeros_like(neighbor_composition)
        for i in range(n_cells):
            neighbors = idx[i]
            smoothed_comp[i] = np.mean(neighbor_composition[neighbors], axis=0)

        # Clustering: Identify "Niches" based on composition
        features = smoothed_comp
        # Whitening removes correlation between features and normalizes variance
        whitened = whiten(features) 
        n_communities = max(3, int(np.sqrt(n_cell_types)) + 2) 
        
        print(f"Running K-Means for Niches (k={n_communities})...")
        np.random.seed(14)
        centroids, labels = kmeans2(whitened, n_communities, minit='points')
        
        self.putative_meta['niche'] = labels.astype(str)
        
        # 4. Visualization & Reporting
        
        # Calculate Cell Type Composition per Niche for Stacked Bar Plot
        cell_type_counts = self.putative_meta.groupby(['niche', 'predicted.cell.type']).size().unstack(fill_value=0)
        cell_type_pct = cell_type_counts.div(cell_type_counts.sum(axis=1), axis=0) * 100

        with plt.style.context('dark_background'):
            # Plot 1: Spatial Deconvolution (Individual Cells)
            fig, ax = plt.subplots(figsize=(16, 9))
            n_groups = self.putative_meta['predicted.cell.type'].nunique()
            
            # Dynamic palette generation based on complexity
            if n_groups <= 20:
                palette_list = sns.color_palette("tab20", n_groups)
            else:
                extended = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", 20) + sns.color_palette("tab20c", 20)
                if n_groups <= 60:
                    palette_list = extended[:n_groups]
                else:
                    palette_list = sns.color_palette("husl", n_groups)
            
            ax.set_aspect('equal')
            
            sns.scatterplot(
                data=self.putative_meta, 
                x='center_x', y='center_y', 
                hue='predicted.cell.type', 
                palette=palette_list, 
                s=2.5, linewidth=0, ax=ax, legend='auto'
            )
            ax.invert_yaxis() # Match image coordinate system (0,0 at top-left)
            ax.set_title("Spatial Deconvolution (Cell Types)")
            ax.set_xlabel("coord_x"); ax.set_ylabel("coord_y")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1 if n_groups < 20 else 2)
            plt.tight_layout()
            self.figures['spatial'] = _save_and_encode_fig(fig, os.path.join(self.figures_dir, "5_Spatial_Deconvolution.png"))
            
            # Plot 2: Spatial Communities (Niches)
            fig2, ax2 = plt.subplots(figsize=(16, 9))
            n_niches_count = self.putative_meta['niche'].nunique()
            
            if n_niches_count <= 20:
                niche_palette = sns.color_palette("tab20", n_niches_count)
            else:
                extended_niche = sns.color_palette("tab20", 20) + sns.color_palette("tab20b", 20) + sns.color_palette("tab20c", 20)
                if n_niches_count <= 60:
                    niche_palette = extended_niche[:n_niches_count]
                else:
                    niche_palette = sns.color_palette("husl", n_niches_count)

            ax2.set_aspect('equal')
            
            sns.scatterplot(data=self.putative_meta, x='center_x', y='center_y', hue='niche', palette=niche_palette, s=2.5, linewidth=0, ax=ax2, legend='full')
            ax2.invert_yaxis()
            ax2.set_title(f"Spatial Niches (K={n_communities})")
            ax2.set_xlabel("coord_x"); ax2.set_ylabel("coord_y")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            self.figures['communities'] = _save_and_encode_fig(fig2, os.path.join(self.figures_dir, "6_Spatial_Communities.png"))

            # Plot 3: Niche Composition Analysis
            fig3, ax3 = plt.subplots(figsize=(16, 9))
            cell_type_pct.plot(kind='bar', stacked=True, color=palette_list, ax=ax3, width=0.8)
            ax3.set_title("Cell Type Composition per Spatial Niche (%)")
            ax3.set_xlabel("Niche")
            ax3.set_ylabel("Percentage (%)")
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            self.figures['cell_type_stack'] = _save_and_encode_fig(fig3, os.path.join(self.figures_dir, "7_Niche_Composition.png"))
        
        # Summary Statistics
        self.stats = {
            "Species": self.species,
            "Spots": self.spatial.shape[0],
            "Clusters": len(self.thresholds),
            "Cells Deconvolved": len(self.putative_meta),
            "Spatial Communities": n_communities
        }
        print(f"Step 4 Done in {(time.time()-start):.2f}s")
        display(fig)
        display(fig2)
        display(fig3)
        return None

    def step5_export_10x(self, cell_size_alpha: float = 5.0, seed: int = 42):
        """
        Step 5: Synthetic Single-Cell Matrix Generation (Dirichlet-Multinomial)

        Reconstructs single-cell profiles using probabilistic sampling to preserve
        biological variance, while ensuring bit-wise reproducibility via seeding.

        Args:
            cell_size_alpha (float): Controls variance in cell library sizes (lower = higher variance).
            seed (int): Random seed for the Monte Carlo simulation.
        """
        print(f"Step 5: Generating 10X Feature BC Matrix (Dirichlet-Multinomial | Seed={seed})")
        if self.putative_meta is None: raise ValueError("Run Step 4 first.")
        
        np.random.seed(seed)
        start = time.time()
        
        out_10x = os.path.join(self.data_path, "deconvolved_feature_bc_matrix")
        os.makedirs(out_10x, exist_ok=True)
        self.putative_meta.to_csv(f"{out_10x}/metadata.csv")
        
        # 1. Export Barcodes & Features
        cell_barcodes = self.putative_meta.index.tolist()
        with gzip.open(f"{out_10x}/barcodes.tsv.gz", 'wt') as f: f.write('\n'.join(cell_barcodes) + '\n')
        
        genes = self.filtered_visium.index.tolist()
        with gzip.open(f"{out_10x}/features.tsv.gz", 'wt') as f:
            for g in genes: f.write(f"{g}\t{g}\tGene Expression\n")
                
        print("Building sparse matrix with biological downsampling...")
        
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        cell_to_idx = {c: i for i, c in enumerate(cell_barcodes)}
        
        row_ind, col_ind, data = [], [], []
        spot_cluster_counts = self.putative_meta.groupby(['spot', 'cluster']).size().to_dict()
        
        raw_matrix_check = self.filtered_visium.iloc[0,0]
        is_float = isinstance(raw_matrix_check, float) or np.issubdtype(self.filtered_visium.dtypes[0], np.floating)
        
        # Random numbers must be consumed in the exact same sequence every run.
        sorted_clusters = sorted(self.mapped_clusters.items(), key=lambda x: x[0])

        for cid, gene_list in sorted_clusters:
            valid_genes = [g for g in gene_list if g in self.filtered_visium.index]
            if not valid_genes: continue
            
            gene_subset = self.filtered_visium.loc[valid_genes] 
            cluster_cells = self.putative_meta[self.putative_meta['cluster'] == cid]
            
            spots_involved = sorted(cluster_cells['spot'].unique())
            valid_spots = [s for s in spots_involved if s in gene_subset.columns]
            
            for spot in valid_spots:
                cells_in_spot = cluster_cells[cluster_cells['spot'] == spot].index
                n_cells = len(cells_in_spot)
                if n_cells == 0: continue

                spot_counts = gene_subset[spot].values
                if is_float: spot_counts = np.round(spot_counts).astype(int)
                
                nz_mask = spot_counts > 0
                if not np.any(nz_mask): continue
                
                active_counts = spot_counts[nz_mask]
                active_genes = gene_subset.index.values[nz_mask]
                g_indices = np.array([gene_to_idx[g] for g in active_genes])
                c_indices = [cell_to_idx[c] for c in cells_in_spot]

                # Reproducible Statistical Allocation
                if n_cells == 1:
                    row_ind.extend(g_indices)
                    col_ind.extend([c_indices[0]] * len(g_indices))
                    data.extend(active_counts)
                else:
                    # 1. Sample library size factors (Dirichlet)
                    # "How much RNA does each cell capture relative to its neighbors?"
                    cell_probs = np.random.dirichlet(np.ones(n_cells) * cell_size_alpha)
                    
                    # 2. Sample transcripts (Multinomial)
                    # "Which cell caught which specific transcript?"
                    distributed_counts = np.array([np.random.multinomial(n, cell_probs) for n in active_counts])
                    assert np.array_equal(distributed_counts.sum(axis=1), active_counts), "Mass conservation violation!"
                    
                    for i, c_idx in enumerate(c_indices):
                        cell_gene_counts = distributed_counts[:, i]
                        nz_in_cell = cell_gene_counts > 0
                        if np.any(nz_in_cell):
                            row_ind.extend(g_indices[nz_in_cell])
                            col_ind.extend([c_idx] * nz_in_cell.sum())
                            data.extend(cell_gene_counts[nz_in_cell])

        # Final Matrix Compilation
        mat = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(len(genes), len(cell_barcodes)))
        io.mmwrite(f"{out_10x}/matrix.mtx", mat)
        
        with open(f"{out_10x}/matrix.mtx", 'rb') as f_in:
            with gzip.open(f"{out_10x}/matrix.mtx.gz", 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        if os.path.exists(f"{out_10x}/matrix.mtx"): os.remove(f"{out_10x}/matrix.mtx")
        
        # Uses Jinja2 templating to embed Base64 encoded images directly into a single HTML file.
        template = """<!DOCTYPE html><html><head><title>CoexpressDeconvolve QC Report</title><style>
        body{font-family:sans-serif;margin:20px; background-color:black; color:white;} 
        h3{border-bottom:1px solid #555;} 
        img{max-height: 85vh; width: auto; max-width: 100%; border:1px solid #444; margin:10px auto; display:block;} 
        ul{background:#222; padding:15px; color:white;}
        </style></head><body>
        <h1>CoexpressDeconvolve QC Report</h1>
        <h3>Deconvolution Statistics</h3><ul>{% for k,v in stats.items() %}<li><b>{{k}}:</b> {{v}}</li>{% endfor %}</ul>
        
        <h3>1. Gene Expression QC</h3>
        <img src="data:image/png;base64,{{ figures.gene_qc }}" />
        
        <h3>2. Co-Expression Topology</h3>
        <img src="data:image/png;base64,{{ figures.knee_plot }}" />
        <img src="data:image/png;base64,{{ figures.gene_umap }}" />
        <img src="data:image/png;base64,{{ figures.marker_heatmap }}" />
        
        <h3>3. Noise Thresholding</h3>
        <img src="data:image/png;base64,{{ figures.thresholds }}" />
        
        <h3>4. Predicted Cell Counts</h3>
        <img src="data:image/png;base64,{{ figures.cell_counts }}" />
        
        <h3>5. Spatial Deconvolution</h3>
        <img src="data:image/png;base64,{{ figures.spatial }}" />
        
        <h3>6. Spatial Communities</h3>
        <img src="data:image/png;base64,{{ figures.communities }}" />
        <img src="data:image/png;base64,{{ figures.cell_type_stack }}" />
        </body></html>"""
        
        with open(f"{self.data_path}/CoexpressDeconvolve_QC_Report.html", "w") as f:
            f.write(jinja2.Template(template).render(figures=self.figures, stats=self.stats))
            
        print(f"Step 5 Done in {(time.time()-start):.2f}s")
        print(f"10X files saved to {out_10x}")