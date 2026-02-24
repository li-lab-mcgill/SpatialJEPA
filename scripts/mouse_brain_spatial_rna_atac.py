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

base_path = os.path.join(os.getenv("DATAPATH"), "aligned_data")

#%% load source data
source_rna = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned.h5ad"))
source_atac = sc.read_h5ad(os.path.join(base_path, "source_atac_aligned.h5ad"))

source_rna.obsm["spatial"] = source_rna.obsm["spatial"][:, [1, 0]] * -1
source_atac.obsm["spatial"] = source_atac.obsm["spatial"][:, [1, 0]] * -1

#%% load target data
target_rna = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned.h5ad"))
target_atac = sc.read_h5ad(os.path.join(base_path, "target_atac_aligned.h5ad"))

#%% TMP - redo HVG to limit number of features to fit inside GPU memory
if socket.gethostname() != 'ri-muhc-gpu':

    source_rna.var['highly_variable'] = False
    source_atac.var['highly_variable'] = False

    target_rna.var['highly_variable'] = False
    target_atac.var['highly_variable'] = False
    
    top_N_genes = 2000
    top_N_peaks = 10000
    source_rna.var.loc[source_rna.var['highly_variable_rank'].le(top_N_genes - 1), 'highly_variable'] = True
    source_atac.var.loc[source_atac.var['highly_variable_rank'].le(top_N_peaks - 1), 'highly_variable'] = True

    target_rna.var.loc[
        target_rna.var_names.isin(source_rna.var_names[source_rna.var['highly_variable']]),
        'highly_variable'] = True
    target_atac.var.loc[
        target_atac.var_names.isin(source_atac.var_names[source_atac.var['highly_variable']]),
        'highly_variable'] = True

#%% spatial graph
MultiGATE.Cal_Spatial_Net(source_rna, rad_cutoff=40)
MultiGATE.Stats_Spatial_Net(source_rna)

MultiGATE.Cal_Spatial_Net(source_atac, rad_cutoff=40)
MultiGATE.Stats_Spatial_Net(source_atac)

source_rna = source_rna[:, source_rna.var['highly_variable']]
source_atac = source_atac[:, source_atac.var['highly_variable']]

gtf_path = os.path.join(os.getenv("DATAPATH"), "gene_annotations", "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz")
if not os.path.exists(gtf_path):
    raise FileNotFoundError("GTF annotation file not found: {}".format(gtf_path))

MultiGATE.Cal_gene_peak_Net_new(source_rna, source_atac, 150000, file=gtf_path)
source_rna.uns['gene_peak_Net'] = source_atac.uns['gene_peak_Net']

#%% running MultiGATE
num_epochs = int(os.getenv("MULTIGATE_EPOCHS", "3000"))
if num_epochs <= 0:
    raise ValueError("MULTIGATE_EPOCHS must be a positive integer.")

print("Training epochs:", num_epochs)
source_rna, source_atac, trainer = MultiGATE.train_MultiGATE(
    source_rna,
    source_atac,
    bp_width=400,
    n_epochs=num_epochs,
    return_trainer=True,
)

#%% clustering with Muon's WNN clustering
sc.pp.neighbors(source_rna)
sc.pp.neighbors(source_atac)

mdata = mu.MuData({"rna": source_rna, "atac": source_atac})
mu.pp.neighbors(mdata)

mu.tl.umap(mdata)
sc.tl.leiden(mdata, resolution=1.5)

# Replicate outputs of wnn_R: propagate cluster labels and UMAP coordinates
# back to the individual AnnData objects so downstream code is unaffected.
for ad in [source_rna, source_atac]:
    ad.obs['wnn'] = mdata.obs['leiden'].astype(int).astype('category')
    ad.obsm['X_umap'] = mdata.obsm['X_umap']

# visualize results
plt.rcParams["figure.figsize"] = (7, 3)
fig, axs = plt.subplots(1, 2)
sc.pl.embedding(source_rna, basis="spatial", color="wnn", s=20, show=False, title='MultiGATE Spatial', ax=axs[0], legend_loc='None')
sc.pl.umap(source_rna, color="wnn", title='MultiGATE UMAP', ax=axs[1], size=20)
plt.tight_layout()
plt.show()

#%% forward pass with target data

import numpy as np
import pandas as pd
import torch
from MultiGATE.MultiGATE import MultiGATE as MultiGATETrainer


def build_knn_graph_as_spatial_net(adata, n_neighbors=15):
    # Build a generic kNN cell graph for non-spatial data and store it in the
    # format expected by MultiGATE.forward_MultiGATE (adata.uns['Spatial_Net']).
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    conn = adata.obsp['connectivities'].tocoo()
    mask = conn.row != conn.col
    adata.uns['Spatial_Net'] = pd.DataFrame(
        {
            'Cell1': adata.obs_names[conn.row[mask]].to_numpy(),
            'Cell2': adata.obs_names[conn.col[mask]].to_numpy(),
            'Distance': np.zeros(int(mask.sum()), dtype=float),
        }
    )


def build_zero_shot_target_trainer(source_trainer, target_spot_num):
    # Rebuild MGATE with target N, load transferable weights, then force the
    # dataset-sized gene-peak gating vectors to zero for prior-only GP attention.
    target_trainer = MultiGATETrainer(
        hidden_dims1=source_trainer.mgate.hidden_dims1,
        hidden_dims2=source_trainer.mgate.hidden_dims2,
        spot_num=target_spot_num,
        temp=float(source_trainer.mgate.logit_scale.detach().cpu().item()),
        n_epochs=1,
        lr=source_trainer.lr,
        gradient_clipping=source_trainer.gradient_clipping,
        nonlinear=source_trainer.mgate.nonlinear,
        weight_decay=source_trainer.mgate.weight_decay,
        verbose=False,
        random_seed=0,
        config={'device': str(source_trainer.device)},
    )

    state_dict = {
        k: v
        for k, v in source_trainer.mgate.state_dict().items()
        if k not in {'vgp0', 'vgp1'}
    }
    target_trainer.mgate.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        target_trainer.mgate.vgp0.zero_()
        target_trainer.mgate.vgp1.zero_()

    return target_trainer


# Subset target to the same HVG feature space as the source model.
# Note: If target is missing any of the source HVGs, forward_MultiGATE will
# raise a dimension mismatch.
target_rna = target_rna[:, target_rna.var['highly_variable']]
target_atac = target_atac[:, target_atac.var['highly_variable']]

# Reuse source gene-peak prior so target forward pass uses the same regulatory graph.
target_rna.uns['gene_peak_Net'] = source_rna.uns['gene_peak_Net']
target_atac.uns['gene_peak_Net'] = source_rna.uns['gene_peak_Net']

# Build a non-spatial kNN graph over target cells and store it as Spatial_Net.
build_knn_graph_as_spatial_net(target_rna, n_neighbors=15)
target_atac.uns['Spatial_Net'] = target_rna.uns['Spatial_Net']
MultiGATE.Stats_Spatial_Net(target_rna)
MultiGATE.Stats_Spatial_Net(target_atac)

# Build a target-compatible trainer and disable data-adaptive GP attention.
trainer_target = build_zero_shot_target_trainer(trainer, target_rna.n_obs)
#%% forward pass
# TMP - subsample target data to match source data
#target_rna = target_rna[:len(source_rna)].copy()
#target_atac = target_atac[:len(source_atac)].copy()

target_rna, target_atac = MultiGATE.forward_MultiGATE(
    target_rna,
    target_atac,
    trainer=trainer_target,
    bp_width=400,
)

print("Target forward pass complete. Embedding shape:", target_rna.obsm["MultiGATE"].shape)

#%% clustering with Muon's WNN clustering

# TMP - filter cells that don't have at least 3 genes expressed
sc.pp.filter_cells(target_rna, min_genes=3)
sc.pp.filter_cells(target_atac, min_genes=3)

sc.pp.neighbors(target_rna, n_neighbors=10)
sc.pp.neighbors(target_atac, n_neighbors=10)

mdata = mu.MuData({"rna": target_rna, "atac": target_atac})
mu.pp.intersect_obs(mdata)

mu.pp.neighbors(mdata, n_neighbors=10)
mu.tl.umap(mdata)
sc.tl.leiden(mdata, resolution=1.5)

# Replicate outputs of wnn_R: propagate cluster labels and UMAP coordinates
# back to the individual AnnData objects so downstream code is unaffected.
for ad in [target_rna, target_atac]:
    ad.obs['wnn'] = mdata.obs['leiden'].astype(int).astype('category')
    ad.obsm['X_umap'] = mdata.obsm['X_umap']

# visualize results
plt.rcParams["figure.figsize"] = (7, 3)
fig, axs = plt.subplots(1, 1)
sc.pl.umap(target_rna, color="wnn", title='MultiGATE UMAP', ax=axs, size=20)
plt.tight_layout()
plt.show()
