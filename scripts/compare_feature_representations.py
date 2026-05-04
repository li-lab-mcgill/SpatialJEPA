#%%
import scanpy as sc
from scipy.sparse import issparse
import numpy as np
import os

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
        pattern = "^MT-",
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

    sct_norm_x = np.asarray(ro.r("sct_scale")).T
    #norm_x = ro.r('res@assays$SCT@scale.data').T
    norm_x = ro.r('GetAssayData(res, assay = "SCT", layer = "data")').T
    adata.layers["SCT"] = norm_x

    return adata, norm_x, sct_norm_x

#%%
if __name__ == "__main__":
    #%%
    base_path = '/home/mcb/users/dmannk/BAKLAVA_base/data/aligned_data'
    output_path = '/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations'
    os.makedirs(output_path, exist_ok=True)

    source_rna  = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned.h5ad"))
    source_adata = source_rna.copy()
    source_adata.X = source_adata.layers["counts"].copy()

    target_rna  = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned.h5ad"))
    target_adata = target_rna.copy()
    target_adata.X = target_adata.layers["counts"].copy()

    source_adata, source_norm_x, source_sct_norm_x = run_SCTransform(source_adata, cell_type_key="RNA_clusters")
    target_adata, target_norm_x, target_sct_norm_x = run_SCTransform(target_adata, cell_type_key="arc_gex_graphclust_Cluster")

    from IPython.display import Image, display
    display(Image("/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations/seurat_dimplot_RNA_clusters.png"))
    display(Image("/home/mcb/users/dmannk/BAKLAVA_base/outputs/compare_feature_representations/seurat_dimplot_arc_gex_graphclust_Cluster.png"))

    #%% save source and target adatas
    source_adata.write(os.path.join(base_path, "source_rna_aligned_SCT.h5ad"))
    target_adata.write(os.path.join(base_path, "target_rna_aligned_SCT.h5ad"))

    # %%
    seurat_umap = ro.r("res@reductions$umap@cell.embeddings")
    adata.obsm["X_seurat_umap"] = seurat_umap
    sc.pl.embedding(adata, color="RNA_clusters", basis="X_seurat_umap")

    seurat_pca = ro.r("res@reductions$pca@cell.embeddings")
    #seurat_pca = seurat_pca[:, 1:]
    adata.obsm["X_seurat_pca"] = seurat_pca

    sc.pp.neighbors(adata, n_pcs=30, use_rep="X_seurat_pca")
    sc.tl.umap(adata)
    sc.pl.umap(adata, color="RNA_clusters")

    #%%
    sct_genes = list(ro.r("sct_genes"))

    adata_sct = adata[:, sct_genes].copy()
    adata_sct.X = norm_x.copy()

    sc.pp.pca(adata_sct, n_comps=30)
    sc.pp.neighbors(adata_sct, n_pcs=30)
    sc.tl.umap(adata_sct)
    sc.pl.umap(adata_sct, color="RNA_clusters")

    #%%

    sct_genes_bool = np.isin(source_rna.var_names, sct_genes)
    source_rna_sct = source_rna[:, sct_genes].copy()

    sc.pp.pca(source_rna_sct, n_comps=30)
    sc.pp.neighbors(source_rna_sct, n_pcs=30)
    sc.tl.umap(source_rna_sct)
    sc.pl.umap(source_rna_sct, color="RNA_clusters")

    source_rna_sct.X = source_rna.layers["counts"][:, sct_genes_bool].copy()

    sc.pp.pca(source_rna_sct, n_comps=30)
    sc.pp.neighbors(source_rna_sct, n_pcs=30)
    sc.tl.umap(source_rna_sct)
    sc.pl.umap(source_rna_sct, color="RNA_clusters")