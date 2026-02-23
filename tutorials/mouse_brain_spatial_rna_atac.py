#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import scanpy as sc
import muon as mu
import matplotlib.pyplot as plt
import os

import MultiGATE

import warnings
warnings.filterwarnings('ignore')

#%% load env variables from .env file

from dotenv import load_dotenv, dotenv_values
load_dotenv()

from pprint import pprint
print("Loaded environment variables from .env or env:", end="\n\n")
pprint(dotenv_values())

base_path = os.path.join(os.getenv("DATAPATH"), "Spatial_ATAC_RNA", "mouse", "spatial_omics")

#%% load RNA data
#file_name = os.path.join(base_path, "spatial_atac_rna_seq_mouse_brain.h5ad")
rna_file_name = os.path.join(base_path, "P22_RNA.h5ad")

adata1 = sc.read_h5ad(rna_file_name)
adata1.obsm["spatial"] = adata1.obsm["spatial"][:, [1, 0]] * -1
adata1

#%% load ATAC data
#file_name = os.path.join(base_path, "spatial_atac_rna_seq_mouse_brain_atac.h5ad")
atac_file_name = os.path.join(base_path, "P22_ATAC_lsi.h5ad")

adata2 = sc.read_h5ad(atac_file_name)
adata2.obsm["spatial"] = adata2.obsm["spatial"][:, [1, 0]] * -1
adata2

#%% compute highly variable genes and peaks

if 'P22_RNA.h5ad' not in rna_file_name:
    #sc.pp.filter_cells(adata1, min_genes=100)
    sc.pp.filter_genes(adata1, min_cells=3)
    sc.pp.normalize_total(adata1, target_sum=1e4)
    sc.pp.log1p(adata1)
    sc.pp.highly_variable_genes(adata1, n_top_genes=2000)

if 'P22_ATAC_lsi.h5ad' not in atac_file_name:
    #sc.pp.filter_cells(adata2, min_genes=100)
    sc.pp.filter_genes(adata2, min_cells=3)
    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)
    sc.pp.highly_variable_genes(adata2, n_top_genes=10000)

#%% spatial graph
MultiGATE.Cal_Spatial_Net(adata1, rad_cutoff=40)
MultiGATE.Stats_Spatial_Net(adata1)

MultiGATE.Cal_Spatial_Net(adata2, rad_cutoff=40)
MultiGATE.Stats_Spatial_Net(adata2)

adata1 = adata1[:, adata1.var['highly_variable']]
adata2 = adata2[:, adata2.var['highly_variable']]

gtf_path = os.path.join(os.getenv("DATAPATH"), "gene_annotations", "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz")
MultiGATE.Cal_gene_peak_Net_new(adata1, adata2, 150000, file=gtf_path)
adata1.uns['gene_peak_Net'] = adata2.uns['gene_peak_Net']

#%% running MultiGATE

num_epochs = 3000
adata1, adata2 = MultiGATE.train_MultiGATE(adata1, adata2, bp_width=400, n_epochs=num_epochs)

#%% clustering in R (skip, replace by Muon's WNN clustering)
'''
# the location of R (used for the WNN clustering)
os.environ['R_HOME'] = "/lustre/project/Stat/s1155077016/condaenvs/Seurat4/lib/R"
os.environ['R_USER'] = '/users/s1155077016/anaconda3/lib/python3.9/site-packages/rpy2'

size=20
MultiGATE.wnn_R(adata1, adata2, res=2.0)
'''

#%% clustering with Muon's WNN clustering

sc.pp.neighbors(adata1)
sc.pp.neighbors(adata2)

mdata = mu.MuData({"rna": adata1, "atac": adata2})
mu.pp.neighbors(mdata) # WNN neighbors, ~5 minutes

#mdata.write(os.path.join(os.getenv("DATAPATH"), "Spatial_ATAC_RNA", "mouse", "spatial_omics", "mouse_brain_spatial_rna_atac.h5mu"))
#mdata = mu.read_h5mu(os.path.join(os.getenv("DATAPATH"), "Spatial_ATAC_RNA", "mouse", "spatial_omics", "mouse_brain_spatial_rna_atac.h5mu"))

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
