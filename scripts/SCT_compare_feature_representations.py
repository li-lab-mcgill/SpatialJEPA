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
        do.correct.umi = FALSE,
        verbose = FALSE
    )

    sct_genes <- VariableFeatures(res)
    sct_scale <- GetAssayData(res, assay = "SCT", layer = "scale.data")
    sct_scale <- sct_scale[sct_genes, , drop = FALSE]

    res <- RunPCA(res, verbose = FALSE)
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

    seurat_pca = ro.r('res@reductions$pca@cell.embeddings')
    seurat_umap = ro.r('res@reductions$umap@cell.embeddings')
    seurat_neighbors = ro.r('res@graphs$SCT_snn')
    seurat_clusters = ro.r('Idents(res)')

    sct_norm_x = np.asarray(ro.r("sct_scale")).T
    #norm_x = ro.r('res@assays$SCT@scale.data').T
    norm_x = ro.r('GetAssayData(res, assay = "SCT", layer = "data")').T
    adata.X = norm_x

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

    return adata, norm_x, sct_norm_x

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

    source_adata, source_norm_x, source_sct_norm_x = run_SCTransform(source_adata, cell_type_key="RNA_clusters")
    target_adata, target_norm_x, target_sct_norm_x = run_SCTransform(target_adata, cell_type_key="arc_gex_graphclust_Cluster")

    from IPython.display import Image, display
    display(Image("/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations/seurat_dimplot_RNA_clusters.png"))
    display(Image("/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations/seurat_dimplot_arc_gex_graphclust_Cluster.png"))

    #%% save source and target adatas
    source_adata.write(os.path.join(base_path, "source_rna_aligned_SCT.h5ad"))
    target_adata.write(os.path.join(base_path, "target_rna_aligned_SCT.h5ad"))
