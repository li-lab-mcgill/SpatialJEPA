#%%
import scanpy as sc
from scipy.sparse import issparse
import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import rpy2.robjects as ro
import anndata2ri


#%%

def run_SCTransform(adata, cell_type_key="RNA_clusters"):

    ro.r('library(Seurat)')
    ro.r('library(scater)')
    anndata2ri.activate()

    #sc.pp.filter_genes(adata, min_cells=5)

    if issparse(adata.X):
        if not adata.X.has_sorted_indices:
            adata.X.sort_indices()

    for key in adata.layers:
        if issparse(adata.layers[key]):
            if not adata.layers[key].has_sorted_indices:
                adata.layers[key].sort_indices()

    ro.globalenv["adata"] = adata
    ro.r('seurat_obj = as.Seurat(adata, counts = "X", data = NULL)')

    ro.r(f"""
    DefaultAssay(seurat_obj) <- "originalexp"

    seurat_obj <- PercentageFeatureSet(
        seurat_obj,
        pattern = "^Mt", # for mouse genome
        col.name = "percent.mt"
    )

    res <- SCTransform(
        object = seurat_obj,
        assay = "originalexp",
        vars.to.regress = "percent.mt",
        return.only.var.genes = FALSE,
        do.correct.umi = TRUE,
        verbose = FALSE,
        variable.features.n = 6000,
        min_cells = 0
    )

    sct_genes <- VariableFeatures(res)

    res <- RunPCA(res, verbose = FALSE, features = rownames(res)) # use rownames of res to avoid dropping genes
    res <- RunUMAP(res, dims = 1:30, verbose = FALSE)

    res <- FindNeighbors(res, dims = 1:30, verbose = FALSE)
    res <- FindClusters(res, verbose = FALSE)
    
    p <- DimPlot(res, group.by = "{cell_type_key}", label = TRUE)
    ggsave("/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations/seurat_dimplot_{cell_type_key}.png", plot = p, width = 7, height = 5, dpi = 150)
    """)
    
    # Assign QC metrics and dimensionality reductions to variables
    seurat_obs = ro.r('res@meta.data')
    adata.obs = seurat_obs.copy()
    assert adata.obs['percent.mt'].sum() > 0, "Percent mitochondrial genes is 0, likely due to wrong pattern for mitochondrial genes"

    adata.obsm['X_seurat_pca'] = ro.r('res@reductions$pca@cell.embeddings')
    adata.obsm['X_seurat_umap'] = ro.r('res@reductions$umap@cell.embeddings')
    #adata.varm['X_seurat_pca_loadings'] = ro.r('res@reductions$pca@feature.loadings')
    #adata.obsp['X_seurat_neighbors'] = ro.r('res@graphs$SCT_snn')

    #norm_x = ro.r('res@assays$SCT@scale.data').T
    #sct_genes_full = list(ro.r("rownames(res@assays$SCT)"))
    sct_counts = ro.r('GetAssayData(res, assay = "SCT", layer = "counts")').T
    sct_scale_data = ro.r('GetAssayData(res, assay = "SCT", layer = "scale.data")').T
    adata.X = sct_scale_data
    adata.layers['SCT_counts'] = sct_counts

    sct_genes = list(ro.r("sct_genes"))
    adata.var['SCT_gene'] = adata.var_names.isin(sct_genes)

    # plot correlation between QC metrics and PCA components
    sc.pp.pca(adata, n_comps=50)
    df = adata.obs[
        ['percent.mt', 'nFeature_SCT', 'nCount_SCT', 'nFeature_originalexp', 'nCount_originalexp']
        ].merge(pd.DataFrame(adata.obsm['X_pca'], index=adata.obs_names), left_index=True, right_index=True)
    plt.figure(figsize=[10,2])
    h = sns.heatmap(df.corr().iloc[:5, 5:])
    h.set_xlabel('PCA components')
    h.set_title('Correlation between QC metrics and PCA components derived from SCT features')

    return adata

#%%
if __name__ == "__main__":
    #%%
    base_path = '/home/mcb/users/dmannk/BAKLAVA_base/data/aligned_data'
    output_path = '/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations'
    os.makedirs(output_path, exist_ok=True)

    source_rna  = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned.h5ad"))
    source_adata = source_rna.copy()
    source_adata.layers['pseudocounts'] = source_adata.X.copy()
    source_adata.X = source_adata.layers["counts"].copy()

    target_rna  = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned.h5ad"))
    target_adata = target_rna.copy()
    target_adata.layers['pseudocounts'] = target_adata.X.copy()
    target_adata.X = target_adata.layers["counts"].copy()

    source_adata = run_SCTransform(source_adata, cell_type_key="RNA_clusters")
    target_adata = run_SCTransform(target_adata, cell_type_key="arc_gex_graphclust_Cluster")

    #from IPython.display import Image, display
    #display(Image("/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations/seurat_dimplot_RNA_clusters.png"))
    #display(Image("/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations/seurat_dimplot_arc_gex_graphclust_Cluster.png"))

    #%% save source and target adatas
    source_adata.write(os.path.join(base_path, "source_rna_aligned_SCT.h5ad"))
    target_adata.write(os.path.join(base_path, "target_rna_aligned_SCT.h5ad"))

    print("Done.")

# %%
