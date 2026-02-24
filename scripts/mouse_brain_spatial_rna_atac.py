#%%
import os
import shutil
import socket
import sys
from pprint import pprint

# Python 3.7 compatibility for muon/mudata (they use typing.Literal in newer versions)
if sys.version_info < (3, 8):
    import typing
    from typing_extensions import Literal

    typing.Literal = Literal

# Ensure this script imports the local repo package, not site-packages.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Make sure env-local binaries (e.g., bedtools) are discoverable when running
# with an explicit python path instead of an activated conda shell.
env_bin = os.path.dirname(sys.executable)
current_path_entries = os.environ.get("PATH", "").split(os.pathsep)
if env_bin and env_bin not in current_path_entries:
    os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import muon as mu
import scanpy as sc
from dotenv import dotenv_values, load_dotenv

import MultiGATE

import warnings
warnings.filterwarnings('ignore')

#%% load env variables from .env file
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")
print("Loaded environment variables from .env or env:", end="\n\n")
pprint(dotenv_values("/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"))
print("Using MultiGATE module:", MultiGATE.__file__)

if os.getenv("DATAPATH") is None:
    raise EnvironmentError(
        "DATAPATH is not set. Export DATAPATH to the base data directory, e.g. "
        "'/home/mcb/users/dmannk/BAKLAVA_base/data'."
    )

if shutil.which("bedtools") is None:
    raise EnvironmentError(
        "bedtools is required for Cal_gene_peak_Net_new. Install bedtools and ensure sortBed is available on PATH."
    )

base_path = os.path.join(os.getenv("DATAPATH"), "Spatial_ATAC_RNA", "mouse", "spatial_omics")

#%% load RNA data
rna_file_name = os.path.join(base_path, "P22_RNA.h5ad")

adata1 = sc.read_h5ad(rna_file_name)
adata1.obsm["spatial"] = adata1.obsm["spatial"][:, [1, 0]] * -1
adata1

#%% load ATAC data
atac_file_name = os.path.join(base_path, "P22_ATAC_lsi.h5ad")

adata2 = sc.read_h5ad(atac_file_name)
adata2.obsm["spatial"] = adata2.obsm["spatial"][:, [1, 0]] * -1
adata2

#%% compute highly variable genes and peaks
if 'P22_RNA.h5ad' not in rna_file_name:
    sc.pp.filter_genes(adata1, min_cells=3)
    sc.pp.normalize_total(adata1, target_sum=1e4)
    sc.pp.log1p(adata1)
    sc.pp.highly_variable_genes(adata1, n_top_genes=2000)

if 'P22_ATAC_lsi.h5ad' not in atac_file_name:
    sc.pp.filter_genes(adata2, min_cells=3)
    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)
    sc.pp.highly_variable_genes(adata2, n_top_genes=10000)

#%% TMP - redo HVG to limit number of features to fit inside GPU memory
if socket.gethostname() != 'ri-muhc-gpu':
    adata1.var['highly_variable'] = False
    adata2.var['highly_variable'] = False

    top_N_genes = 2000
    top_N_peaks = 10000
    adata1.var.loc[adata1.var['highly_variable_rank'].le(top_N_genes - 1), 'highly_variable'] = True
    adata2.var.loc[adata2.var['highly_variable_rank'].le(top_N_peaks - 1), 'highly_variable'] = True

#%% spatial graph
MultiGATE.Cal_Spatial_Net(adata1, rad_cutoff=40)
MultiGATE.Stats_Spatial_Net(adata1)

MultiGATE.Cal_Spatial_Net(adata2, rad_cutoff=40)
MultiGATE.Stats_Spatial_Net(adata2)

adata1 = adata1[:, adata1.var['highly_variable']]
adata2 = adata2[:, adata2.var['highly_variable']]

gtf_path = os.path.join(os.getenv("DATAPATH"), "gene_annotations", "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz")
if not os.path.exists(gtf_path):
    raise FileNotFoundError("GTF annotation file not found: {}".format(gtf_path))

MultiGATE.Cal_gene_peak_Net_new(adata1, adata2, 150000, file=gtf_path)
adata1.uns['gene_peak_Net'] = adata2.uns['gene_peak_Net']

#%% running MultiGATE
num_epochs = int(os.getenv("MULTIGATE_EPOCHS", "3000"))
if num_epochs <= 0:
    raise ValueError("MULTIGATE_EPOCHS must be a positive integer.")

print("Training epochs:", num_epochs)
adata1, adata2 = MultiGATE.train_MultiGATE(adata1, adata2, bp_width=400, n_epochs=num_epochs)

#%% clustering with Muon's WNN clustering
sc.pp.neighbors(adata1)
sc.pp.neighbors(adata2)

mdata = mu.MuData({"rna": adata1, "atac": adata2})
mu.pp.neighbors(mdata)

mu.tl.umap(mdata)
sc.tl.leiden(mdata, resolution=1.5)

# Replicate outputs of wnn_R: propagate cluster labels and UMAP coordinates
# back to the individual AnnData objects so downstream code is unaffected.
for ad in [adata1, adata2]:
    ad.obs['wnn'] = mdata.obs['leiden'].astype(int).astype('category')
    ad.obsm['X_umap'] = mdata.obsm['X_umap']

#%% visualize results
plt.rcParams["figure.figsize"] = (7, 3)
fig, axs = plt.subplots(1, 2)
sc.pl.embedding(adata1, basis="spatial", color="wnn", s=20, show=False, title='MultiGATE Spatial', ax=axs[0], legend_loc='None')
sc.pl.umap(adata1, color="wnn", title='MultiGATE UMAP', ax=axs[1], size=20)
plt.tight_layout()
plt.show()

# %%
