from __future__ import annotations

import os
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

__all__ = [
    "multimodal_latents_adata",
    "load_spatial_atac_rna_mouse_brain_source",
    "load_10x_mouse_brain_ad_data",
    "basic_feature_processing_for_alignment",
    "annotate_genes_and_peaks_for_alignment",
    "filter_spatially_variable_features",
    "align_source_target_multimodal_by_overlap",
    "finalize_target_after_alignment",
    "add_pseudocount_layers",
    "fix_var_for_h5ad",
]


def _read_cellranger_csv_with_barcodes(csv_path: str) -> pd.DataFrame:
    """Read a Cell Ranger CSV and set barcode-like first column as index."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    first_col = df.columns[0]
    first_values = df[first_col].astype(str)
    barcode_colnames = {"barcode", "barcodes", "cell", "cell_id"}
    if first_col.lower() in barcode_colnames or first_values.str.contains("-").any():
        df = df.set_index(first_col)
    df.index = df.index.astype(str)
    return df


def _resolve_analysis_dir(data_dir: str, base_name: str) -> Optional[str]:
    """Best-effort resolution of Cell Ranger ARC analysis directory."""
    candidates = [
        os.path.join(data_dir, "analysis"),
        os.path.join(data_dir, f"{base_name}_analysis"),
        os.path.join(data_dir, f"{base_name}_analysis", "analysis"),
        os.path.join(data_dir, "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote_analysis"),
        os.path.join(data_dir, "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote_analysis", "analysis"),
    ]
    for analysis_dir in candidates:
        if os.path.isdir(analysis_dir) and (
            os.path.isdir(os.path.join(analysis_dir, "clustering"))
            or os.path.isdir(os.path.join(analysis_dir, "dimensionality_reduction"))
        ):
            return analysis_dir
    return None


def _merge_obs_table(adata: ad.AnnData, table: pd.DataFrame, col_prefix: str) -> None:
    """Merge a barcode-indexed table into adata.obs with a prefix."""
    if table.empty:
        return
    table = table.copy()
    table.index = table.index.astype(str)
    table.columns = [f"{col_prefix}{c}" for c in table.columns]
    adata.obs = adata.obs.merge(table, left_index=True, right_index=True, how="left")


def _add_cellranger_arc_analysis_metadata(
    adata_rna: ad.AnnData,
    adata_atac: ad.AnnData,
    analysis_dir: str,
) -> None:
    """Attach key ARC secondary analysis outputs to RNA and ATAC AnnData objects."""

    # 1) Dimensionality reduction projections
    dr_specs = [
        ("gex", "pca_projection.csv", adata_rna, "X_arc_gex_pca"),
        ("gex", "umap_projection.csv", adata_rna, "X_arc_gex_umap"),
        ("gex", "tsne_projection.csv", adata_rna, "X_arc_gex_tsne"),
        ("atac", "lsa_projection.csv", adata_atac, "X_arc_atac_lsa"),
        ("atac", "umap_projection.csv", adata_atac, "X_arc_atac_umap"),
        ("atac", "tsne_projection.csv", adata_atac, "X_arc_atac_tsne"),
    ]
    for modality, fname, adata, obsm_key in dr_specs:
        csv_path = os.path.join(analysis_dir, "dimensionality_reduction", modality, fname)
        if not os.path.isfile(csv_path):
            continue
        proj = _read_cellranger_csv_with_barcodes(csv_path)
        if proj.empty:
            continue
        proj = proj.apply(pd.to_numeric, errors="coerce")
        proj = proj.reindex(adata.obs_names)
        adata.obsm[obsm_key] = proj.to_numpy(dtype=np.float32, copy=False)

    # 2) Clustering outputs
    cluster_specs = [("gex", adata_rna, "arc_gex_"), ("atac", adata_atac, "arc_atac_")]
    cluster_dirs = ["graphclust", "kmeans_2_clusters", "kmeans_3_clusters", "kmeans_4_clusters", "kmeans_5_clusters"]
    for modality, adata, prefix in cluster_specs:
        for cluster_dir in cluster_dirs:
            csv_path = os.path.join(analysis_dir, "clustering", modality, cluster_dir, "clusters.csv")
            if not os.path.isfile(csv_path):
                continue
            clusters = _read_cellranger_csv_with_barcodes(csv_path)
            if clusters.empty:
                continue
            _merge_obs_table(adata, clusters, f"{prefix}{cluster_dir}_")

    # 3) Feature linkage (peak-gene links)
    linkage_bedpe = os.path.join(analysis_dir, "feature_linkage", "feature_linkage.bedpe")
    if os.path.isfile(linkage_bedpe):
        try:
            linkage_df = pd.read_csv(linkage_bedpe, sep="\t", header=None)
            linkage_df.columns = linkage_df.columns.astype(str)
            adata_rna.uns["arc_feature_linkage_bedpe"] = linkage_df
            adata_atac.uns["arc_feature_linkage_bedpe"] = linkage_df
        except Exception:
            pass

    linkage_matrix_h5 = os.path.join(analysis_dir, "feature_linkage", "feature_linkage_matrix.h5")
    if os.path.isfile(linkage_matrix_h5):
        adata_rna.uns["arc_feature_linkage_matrix_h5_path"] = linkage_matrix_h5
        adata_atac.uns["arc_feature_linkage_matrix_h5_path"] = linkage_matrix_h5

    # 4) TF analysis: peak-to-motif mapping
    peak_motif_bed = os.path.join(analysis_dir, "tf_analysis", "peak_motif_mapping.bed")
    if os.path.isfile(peak_motif_bed):
        try:
            peak_motif_df = pd.read_csv(peak_motif_bed, sep="\t", header=None)
            peak_motif_df.columns = peak_motif_df.columns.astype(str)
            adata_atac.uns["arc_peak_motif_mapping_bed"] = peak_motif_df
            adata_rna.uns["arc_peak_motif_mapping_bed"] = peak_motif_df
        except Exception:
            pass

    # 5) TF-by-cell matrix
    tf_matrix_dir = os.path.join(analysis_dir, "tf_analysis", "filtered_tf_bc_matrix")
    barcodes_path = os.path.join(tf_matrix_dir, "barcodes.tsv.gz")
    motifs_path = os.path.join(tf_matrix_dir, "motifs.tsv")
    matrix_path = os.path.join(tf_matrix_dir, "matrix.mtx.gz")
    if os.path.isfile(barcodes_path) and os.path.isfile(motifs_path) and os.path.isfile(matrix_path):
        try:
            from scipy import sparse
            from scipy.io import mmread

            barcodes = pd.read_csv(barcodes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
            motifs = pd.read_csv(motifs_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
            tf_mtx = mmread(matrix_path).tocsr()

            if tf_mtx.shape == (len(motifs), len(barcodes)):
                tf_by_cell = tf_mtx.T.tocsr()
            elif tf_mtx.shape == (len(barcodes), len(motifs)):
                tf_by_cell = tf_mtx.tocsr()
            else:
                tf_by_cell = None

            if tf_by_cell is not None:
                barcode_indexer = pd.Index(barcodes).get_indexer(adata_atac.obs_names.astype(str))
                missing_row = sparse.csr_matrix((1, tf_by_cell.shape[1]), dtype=tf_by_cell.dtype)
                tf_with_missing = sparse.vstack([tf_by_cell, missing_row], format="csr")
                safe_indexer = np.where(barcode_indexer >= 0, barcode_indexer, tf_by_cell.shape[0])
                aligned_tf = tf_with_missing[safe_indexer, :]

                adata_atac.obsm["X_arc_tf_motif"] = aligned_tf
                adata_atac.uns["arc_tf_motif_names"] = np.asarray(motifs, dtype=object)
                adata_rna.uns["arc_tf_motif_names"] = np.asarray(motifs, dtype=object)
        except Exception:
            pass


def multimodal_latents_adata(modality1_dict, modality2_dict, latent_key):
    """Build a single AnnData from two modality-specific latent spaces."""

    adata1 = list(modality1_dict.values())[0]
    adata2 = list(modality2_dict.values())[0]
    modality1_name = list(modality1_dict.keys())[0]
    modality2_name = list(modality2_dict.keys())[0]

    obs = pd.concat(
        [
            adata1.obs.assign(modality=modality1_name),
            adata2.obs.assign(modality=modality2_name),
        ],
        axis=0,
    )

    latents = np.concatenate([adata1.obsm[latent_key], adata2.obsm[latent_key]], axis=0)

    return sc.AnnData(X=latents, obs=obs)


def load_spatial_atac_rna_mouse_brain_source(
    so_data_folder_path: str,
    dataset: str,
    *,
    cell_type_annotations_csv: Optional[str] = None,
    cell_type_key: str = "cell_type",
    cell_type_col_in_csv: str = "predicted.celltype",
    atac_clusters_col_in_csv: str = "ATAC_clusters",
    spatial_key: str = "spatial",
    n_neighbors: int = 4,
    adj_key: str = "spatial_connectivities",
    make_adjacency_symmetric: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData, str, str]:
    """Load the paired source RNA/ATAC spatial mouse brain dataset.

    Returns
    -------
    (adata_rna, adata_atac, source_assembly, source_name)
    """

    adata = sc.read_h5ad(os.path.join(so_data_folder_path, f"{dataset}.h5ad"))
    adata_atac = sc.read_h5ad(os.path.join(so_data_folder_path, f"{dataset}_atac.h5ad"))

    if cell_type_annotations_csv is None:
        cell_type_annotations_csv = os.path.join(
            so_data_folder_path, f"{dataset}_cell_type_annotations.csv"
        )

    if os.path.isfile(cell_type_annotations_csv):
        cell_type_df = pd.read_csv(cell_type_annotations_csv, index_col=0)
        if cell_type_col_in_csv in cell_type_df.columns:
            cell_type_df = cell_type_df.rename({cell_type_col_in_csv: cell_type_key}, axis=1)
        if atac_clusters_col_in_csv in cell_type_df.columns:
            cell_type_df = cell_type_df.drop(atac_clusters_col_in_csv, axis=1)
        adata.obs = adata.obs.merge(cell_type_df, left_index=True, right_index=True, how="left")

    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        spatial_key=spatial_key,
        n_neighs=n_neighbors,
    )

    if make_adjacency_symmetric and (adj_key in adata.obsp):
        adata.obsp[adj_key] = adata.obsp[adj_key].maximum(adata.obsp[adj_key].T)

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    if "counts" not in adata_atac.layers:
        adata_atac.layers["counts"] = adata_atac.X.copy()

    return adata, adata_atac, "mm10", "Spatial_ATAC_RNA"

def load_mousedev_spatial_triomic_data(
    data_dir: str,
    make_adjacency_symmetric: bool = True,
    adj_key: str = "spatial_connectivities",
) -> Tuple[ad.AnnData, ad.AnnData, str, str]:
    """Load the mousedev spatial triomic dataset."""

    rna_datapath = os.path.join(data_dir, "rna_adata.h5ad")
    atac_datapath = os.path.join(data_dir, "MouseDev_Triomic_ATAC.h5ad")
    rna_adata = sc.read_h5ad(rna_datapath, backed="r")
    atac_adata = sc.read_h5ad(atac_datapath, backed="r")

    ## reformat atac obs_names to match rna_adata
    atac_obs_names = atac_adata.obs_names.copy()
    atac_adata.obs_names = atac_obs_names.str.split('_|fragments:|-').str[-2] + '_' + atac_obs_names.str.split('_|fragments:').str[2]

    ## filter to only overlap
    obs_overlap = np.intersect1d(rna_adata.obs_names, atac_adata.obs_names)
    rna_adata = rna_adata[rna_adata.obs_names.isin(obs_overlap)].to_memory()
    atac_adata = atac_adata[atac_adata.obs_names.isin(obs_overlap)].to_memory()

    ## sort cells such that they are in the same order in both adatas
    rna_adata = rna_adata[rna_adata.obs_names.argsort()].copy()
    atac_adata = atac_adata[atac_adata.obs_names.argsort()].copy()
    assert (rna_adata.obs_names == atac_adata.obs_names).all()

    ## load spatial coordinates
    coords_df_list = []
    import tarfile
    with tarfile.open(os.path.join(data_dir, "GSE308623.tar"), "r") as tar:
        for member in tar.getnames():
            if member.endswith("tissue_positions_list.csv.gz"):
                sample_label = member.split("_")[2]
                if sample_label.startswith("P"): # ignores non-developmental samples from LPC experiments
                    coords_df = pd.read_csv(tar.extractfile(member), index_col=0, header=None, compression="gzip")
                    coords_df.columns = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
                    coords_df.index = coords_df.index.astype(str)
                    coords_df.index.name = "barcode"
                    coords_df.index = coords_df.index + "_" + sample_label
                    coords_df_list.append(coords_df)

    coords_df = pd.concat(coords_df_list)

    rna_adata.obs = rna_adata.obs.merge(coords_df, left_index=True, right_index=True, how="left")
    atac_adata.obs = atac_adata.obs.merge(coords_df, left_index=True, right_index=True, how="left")

    in_tissue_value_counts = pd.concat([
        rna_adata.obs["in_tissue"].value_counts(),
        atac_adata.obs["in_tissue"].value_counts()
        ])
    if not in_tissue_value_counts.index.unique().tolist() == [0, 1]:
        print("WARNING: overlapping samples don't contain both in_tissue values (0 and 1)")

    ## add spatial coordinates to adatas
    rna_adata.obsm["spatial"] = rna_adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    atac_adata.obsm["spatial"] = atac_adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)

    ## build spatial neighbors
    sq.gr.spatial_neighbors(
        rna_adata,
        coord_type="generic",
        spatial_key="spatial",
        n_neighs=4,
    )

    if make_adjacency_symmetric and (adj_key in rna_adata.obsp):
        rna_adata.obsp[adj_key] = rna_adata.obsp[adj_key].maximum(rna_adata.obsp[adj_key].T)

    if "counts" not in rna_adata.layers:
        rna_adata.layers["counts"] = rna_adata.X.astype(np.int32).copy()
    if "counts" not in atac_adata.layers:
        atac_adata.layers["counts"] = atac_adata.X.astype(np.int32).copy()

    return rna_adata, atac_adata, "mm10", "mousedev_spatial_triomic"

def load_10x_mouse_brain_ad_data(
    data_dir: Optional[str] = None,
    base_name: str = "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote",
) -> Tuple[ad.AnnData, ad.AnnData, str, str]:
    """Load 10x Multiome (RNA+ATAC) mouse brain AD dataset.

    Expected files in `data_dir`:
    - {base_name}_filtered_feature_bc_matrix.h5  (required)
    - {base_name}_atac_peaks.bed                (optional)
    - {base_name}_atac_peak_annotation.tsv      (optional)

    Returns
    -------
    (target_rna, target_atac, target_assembly, target_name)
    """

    if data_dir is None:
        # data/ folder is one level above the repo folder (BAKLAVA/)
        repo_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(repo_dir, "..", "data", "10x_mouse_brain_AD")
    data_dir = os.path.abspath(data_dir)

    h5_path = os.path.join(data_dir, f"{base_name}_filtered_feature_bc_matrix.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"10x filtered feature-barcode matrix not found: {h5_path}. "
            "Download it from 10x into data_dir."
        )

    adata_full = sc.read_10x_h5(h5_path, gex_only=False)
    ft_col = "feature_types" if "feature_types" in adata_full.var.columns else "feature_type"
    if ft_col not in adata_full.var.columns:
        raise ValueError(
            f"Expected feature type column '{ft_col}' in 10x multiome var. "
            f"Columns: {list(adata_full.var.columns)}"
        )

    is_gex = adata_full.var[ft_col].astype(str).str.strip().str.lower().eq("gene expression")
    is_peaks = adata_full.var[ft_col].astype(str).str.strip().str.lower().eq("peaks")

    adata_rna = adata_full[:, is_gex].copy()
    adata_atac = adata_full[:, is_peaks].copy()

    for a in (adata_rna, adata_atac):
        if ft_col in a.var.columns:
            a.var = a.var.drop(columns=[ft_col])

    adata_rna.layers["counts"] = adata_rna.X.copy()
    adata_atac.layers["counts"] = adata_atac.X.copy()

    peaks_bed_path = os.path.join(data_dir, f"{base_name}_atac_peaks.bed")
    peak_ann_path = os.path.join(data_dir, f"{base_name}_atac_peak_annotation.tsv")

    if os.path.isfile(peaks_bed_path):
        bed = pd.read_csv(
            peaks_bed_path,
            sep="	",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "chromStart", "chromEnd"],
        )
        peak_id_bed = bed["chrom"].astype(str) + "-" + bed["chromStart"].astype(str) + "-" + bed["chromEnd"].astype(str)

        if adata_atac.n_vars == len(peak_id_bed) and (adata_atac.var_names == peak_id_bed.values).all():
            adata_atac.var["chrom"] = bed["chrom"].values
            adata_atac.var["chromStart"] = bed["chromStart"].values
            adata_atac.var["chromEnd"] = bed["chromEnd"].values
        else:
            adata_atac.var["peak"] = adata_atac.var_names
            coords = adata_atac.var["peak"].str.split(":|-", n=2, expand=True)
            adata_atac.var["chrom"] = coords[0]
            adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
            adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""
    else:
        adata_atac.var["peak"] = adata_atac.var_names
        coords = adata_atac.var["peak"].str.split(":|-", n=2, expand=True)
        adata_atac.var["chrom"] = coords[0]
        adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
        adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""

    if os.path.isfile(peak_ann_path):
        ann = pd.read_csv(peak_ann_path, sep="	")
        peak_col = "Peak" if "Peak" in ann.columns else ("peak" if "peak" in ann.columns else None)
        if peak_col is not None:
            ann_indexed = ann.set_index(peak_col)
            extra = [c for c in ann_indexed.columns if c not in adata_atac.var.columns]
            if extra:
                adata_atac.var = adata_atac.var.merge(
                    ann_indexed[extra], left_index=True, right_index=True, how="left"
                )

    # Unpack Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote_analysis.tar.gz in the data_dir if present
    analysis_dir = _resolve_analysis_dir(data_dir, base_name)
    tar_path = os.path.join(data_dir, "Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote_analysis.tar.gz")
    if os.path.isfile(tar_path) and analysis_dir is None:
        import tarfile
        print(f"Found tar archive: {tar_path}. Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction completed.")
        analysis_dir = _resolve_analysis_dir(data_dir, base_name)

    if analysis_dir is not None:
        print(f"Found Cell Ranger ARC analysis directory: {analysis_dir}.")
        _add_cellranger_arc_analysis_metadata(adata_rna, adata_atac, analysis_dir)

    # set 'arc_gex_graphclust_Cluster' as the reference clusters
    reference_clusters_rna_and_atac = adata_rna.obs['arc_gex_graphclust_Cluster']
    reference_clusters_key = 'REF_arc_gex_graphclust_Cluster'

    adata_rna.uns['label_key'] = reference_clusters_key
    adata_atac.uns['label_key'] = reference_clusters_key
    adata_rna.obs[reference_clusters_key] = reference_clusters_rna_and_atac
    adata_atac.obs[reference_clusters_key] = reference_clusters_rna_and_atac

    # convert non-string columns in uns to string representation
    for ad in (adata_rna, adata_atac):
        for k, v in list(ad.uns.items()):
            if isinstance(v, pd.DataFrame) and not all(isinstance(c, str) for c in v.columns):
                v = v.copy()
                v.columns = v.columns.astype(str) 
                ad.uns[k] = v

    return adata_rna, adata_atac, "mm10", "10x_mouse_brain_AD"

def load_10x_mouse_brain_data(
    data_dir: Optional[str] = None,
    base_name: str = "M_Brain_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP",
) -> Tuple[ad.AnnData, ad.AnnData, str, str]:
    """Load 10x Multiome (RNA+ATAC) mouse brain AD dataset.

    Expected files in `data_dir`:
    - {base_name}_filtered_feature_bc_matrix.h5  (required)
    - {base_name}_atac_peaks.bed                (optional)
    - {base_name}_atac_peak_annotation.tsv      (optional)

    Returns
    -------
    (target_rna, target_atac, target_assembly, target_name)
    """

    if data_dir is None:
        # data/ folder is one level above the repo folder (BAKLAVA/)
        repo_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(repo_dir, "..", "data", "10x_mouse_brain")
    data_dir = os.path.abspath(data_dir)

    h5_path = os.path.join(data_dir, f"{base_name}_filtered_feature_bc_matrix.h5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"10x filtered feature-barcode matrix not found: {h5_path}. "
            "Download it from 10x into data_dir."
        )

    adata_full = sc.read_10x_h5(h5_path, gex_only=False)
    ft_col = "feature_types" if "feature_types" in adata_full.var.columns else "feature_type"
    if ft_col not in adata_full.var.columns:
        raise ValueError(
            f"Expected feature type column '{ft_col}' in 10x multiome var. "
            f"Columns: {list(adata_full.var.columns)}"
        )

    is_gex = adata_full.var[ft_col].astype(str).str.strip().str.lower().eq("gene expression")
    is_peaks = adata_full.var[ft_col].astype(str).str.strip().str.lower().eq("peaks")

    adata_rna = adata_full[:, is_gex].copy()
    adata_atac = adata_full[:, is_peaks].copy()

    for a in (adata_rna, adata_atac):
        if ft_col in a.var.columns:
            a.var = a.var.drop(columns=[ft_col])

    adata_rna.layers["counts"] = adata_rna.X.copy()
    adata_atac.layers["counts"] = adata_atac.X.copy()

    peaks_bed_path = os.path.join(data_dir, f"{base_name}_atac_peaks.bed")
    peak_ann_path = os.path.join(data_dir, f"{base_name}_atac_peak_annotation.tsv")

    if os.path.isfile(peaks_bed_path):
        bed = pd.read_csv(
            peaks_bed_path,
            sep="	",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "chromStart", "chromEnd"],
        )
        peak_id_bed = bed["chrom"].astype(str) + "-" + bed["chromStart"].astype(str) + "-" + bed["chromEnd"].astype(str)

        if adata_atac.n_vars == len(peak_id_bed) and (adata_atac.var_names == peak_id_bed.values).all():
            adata_atac.var["chrom"] = bed["chrom"].values
            adata_atac.var["chromStart"] = bed["chromStart"].values
            adata_atac.var["chromEnd"] = bed["chromEnd"].values
        else:
            adata_atac.var["peak"] = adata_atac.var_names
            coords = adata_atac.var["peak"].str.split(":|-", n=2, expand=True)
            adata_atac.var["chrom"] = coords[0]
            adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
            adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""
    else:
        adata_atac.var["peak"] = adata_atac.var_names
        coords = adata_atac.var["peak"].str.split(":|-", n=2, expand=True)
        adata_atac.var["chrom"] = coords[0]
        adata_atac.var["chromStart"] = pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else ""
        adata_atac.var["chromEnd"] = pd.to_numeric(coords[2], errors="coerce") if coords.shape[1] > 2 else ""

    if os.path.isfile(peak_ann_path):
        ann = pd.read_csv(peak_ann_path, sep="	")
        peak_col = "Peak" if "Peak" in ann.columns else ("peak" if "peak" in ann.columns else None)
        if peak_col is not None:
            ann_indexed = ann.set_index(peak_col)
            extra = [c for c in ann_indexed.columns if c not in adata_atac.var.columns]
            if extra:
                adata_atac.var = adata_atac.var.merge(
                    ann_indexed[extra], left_index=True, right_index=True, how="left"
                )

    # Unpack {base_name}_analysis.tar.gz in the data_dir if present
    analysis_dir = _resolve_analysis_dir(data_dir, base_name)
    tar_path = os.path.join(data_dir, f"{base_name}_analysis.tar.gz")
    if os.path.isfile(tar_path) and analysis_dir is None:
        import tarfile
        print(f"Found tar archive: {tar_path}. Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction completed.")
        analysis_dir = _resolve_analysis_dir(data_dir, base_name)

    if analysis_dir is not None:
        print(f"Found Cell Ranger ARC analysis directory: {analysis_dir}.")
        _add_cellranger_arc_analysis_metadata(adata_rna, adata_atac, analysis_dir)

    # set 'arc_gex_graphclust_Cluster' as the reference clusters
    reference_clusters_rna_and_atac = adata_rna.obs['arc_gex_graphclust_Cluster']
    reference_clusters_key = 'REF_arc_gex_graphclust_Cluster'

    adata_rna.uns['label_key'] = reference_clusters_key
    adata_atac.uns['label_key'] = reference_clusters_key
    adata_rna.obs[reference_clusters_key] = reference_clusters_rna_and_atac
    adata_atac.obs[reference_clusters_key] = reference_clusters_rna_and_atac

    # convert non-string columns in uns to string representation
    for ad in (adata_rna, adata_atac):
        for k, v in list(ad.uns.items()):
            if isinstance(v, pd.DataFrame) and not all(isinstance(c, str) for c in v.columns):
                v = v.copy()
                v.columns = v.columns.astype(str) 
                ad.uns[k] = v

    return adata_rna, adata_atac, "mm10", "10x_mouse_brain"


def basic_feature_processing_for_alignment(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    target_min_genes: int = 100,
    target_min_cells: int = 3,
    n_pcs: int = 50,
    target_knn: int = 100,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    """Basic preprocessing used before overlap-based alignment."""

    for obj, name in [
        (adata, "source_rna"),
        (adata_atac, "source_atac"),
        (target_rna, "target_rna"),
        (target_atac, "target_atac"),
    ]:
        if "counts" not in obj.layers:
            raise KeyError(f"Counts layer not found in {name}.layers['counts']")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.normalize_total(adata_atac, target_sum=1e4)
    sc.pp.log1p(adata_atac)

    sc.pp.filter_cells(target_rna, min_genes=target_min_genes)
    sc.pp.filter_genes(target_rna, min_cells=target_min_cells)
    sc.pp.normalize_total(target_rna, target_sum=1e4)
    sc.pp.log1p(target_rna)

    sc.pp.filter_cells(target_atac, min_genes=target_min_genes)
    sc.pp.filter_genes(target_atac, min_cells=target_min_cells)
    sc.pp.normalize_total(target_atac, target_sum=1e4)
    sc.pp.log1p(target_atac)

    sc.pp.pca(target_rna, n_comps=n_pcs)
    sc.pp.neighbors(target_rna, use_rep="X_pca", n_neighbors=target_knn)

    sc.pp.pca(target_atac, n_comps=n_pcs)
    sc.pp.neighbors(target_atac, use_rep="X_pca", n_neighbors=target_knn)

    sc.pp.highly_variable_genes(adata)
    sc.pp.highly_variable_genes(adata_atac)
    sc.pp.highly_variable_genes(target_rna)
    sc.pp.highly_variable_genes(target_atac)

    return adata, adata_atac, target_rna, target_atac


def align_source_target_multimodal_by_overlap(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    source_name: str,
    target_name: str,
    source_assembly: str,
    target_assembly: str,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    """Align source and target by gene and peak overlap using `DataAligner`."""

    from muon import MuData
    from data_aligner import DataAligner

    source_data = MuData({"rna": adata, "atac": adata_atac})
    target_data = MuData({"rna": target_rna, "atac": target_atac})

    data_aligner = DataAligner(
        source_data=source_data,
        target_data=target_data,
        source_name=source_name,
        target_name=target_name,
        source_assembly=source_assembly,
        target_assembly=target_assembly,
    )
    data_aligner.find_gene_overlap()
    data_aligner.find_peak_overlap()
    data_aligner.align_features_by_overlap()

    return (
        data_aligner.source_data["rna"],
        data_aligner.source_data["atac"],
        data_aligner.target_data["rna"],
        data_aligner.target_data["atac"],
    )


def finalize_target_after_alignment(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    adj_type: str = "knn",
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Copy annotations from source to target and set `spatial_connectivities`."""

    from data_aligner import DataAligner

    target_rna, target_atac = DataAligner.copy_annotations_to_target(
        source_rna=adata,
        source_atac=adata_atac,
        target_rna=target_rna,
        target_atac=target_atac,
    )
    target_rna, target_atac = DataAligner.set_target_spatial_connectivities(
        target_rna=target_rna,
        target_atac=target_atac,
        adj_type=adj_type,
    )
    return target_rna, target_atac


def add_pseudocount_layers(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    *,
    pseudocount_key: str = "pseudocounts",
) -> None:
    """Create shallow references to `.X` in `.layers[pseudocount_key]` (in-place)."""

    adata.layers[pseudocount_key] = adata.X
    adata_atac.layers[pseudocount_key] = adata_atac.X
    target_rna.layers[pseudocount_key] = target_rna.X
    target_atac.layers[pseudocount_key] = target_atac.X


def fix_var_for_h5ad(adata_obj: ad.AnnData) -> ad.AnnData:
    """Convert non-string columns in `.var` to string representation (best-effort)."""

    for col in list(adata_obj.var.columns):
        try:
            if adata_obj.var[col].dtype == "object":

                def safe_str_convert(x):
                    if pd.isna(x) or x is None:
                        return ""
                    if isinstance(x, str):
                        return x
                    try:
                        return str(x)
                    except Exception:
                        return ""

                adata_obj.var[col] = adata_obj.var[col].apply(safe_str_convert).astype(str)
            elif hasattr(adata_obj.var[col].dtype, "categories"):
                adata_obj.var[col] = adata_obj.var[col].astype(str)
        except Exception:
            adata_obj.var = adata_obj.var.drop(columns=[col])

    return adata_obj


def annotate_genes_and_peaks_for_alignment(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    *,
    gtf_file_path: str,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Add gene and peak genomic coordinates using a GTF file."""

    from nichecompass.utils import get_gene_annotations

    return get_gene_annotations(adata=adata, adata_atac=adata_atac, gtf_file_path=gtf_file_path)


def filter_spatially_variable_features(
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    *,
    combined_gp_dict: dict,
    filter_genes: bool = True,
    filter_peaks: bool = True,
    n_svg: int = 3000,
    n_svp: int = 15000,
    min_cell_gene_thresh_ratio: float = 0.005,
    min_cell_peak_thresh_ratio: float = 0.005,
    adj_key: str = "spatial_connectivities",
    verbose: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Filter source data to keep only spatially variable genes and peaks.
    
    This exactly mirrors the filtering logic from the tutorial block (section 2.3).
    
    Parameters
    ----------
    adata : ad.AnnData
        Source RNA data with spatial information
    adata_atac : ad.AnnData
        Source ATAC data
    combined_gp_dict : dict
        Combined gene program dictionary
    filter_genes : bool
        Whether to filter genes to spatially variable ones
    filter_peaks : bool
        Whether to filter peaks to spatially variable ones
    n_svg : int
        Number of spatially variable genes to keep
    n_svp : int
        Number of spatially variable peaks to keep
    min_cell_gene_thresh_ratio : float
        Minimum fraction of cells expressing a gene
    min_cell_peak_thresh_ratio : float
        Minimum fraction of cells with peak accessibility
    adj_key : str
        Key for spatial adjacency matrix
    verbose : bool
        Print progress messages
        
    Returns
    -------
    (adata, adata_atac) : Tuple[ad.AnnData, ad.AnnData]
        Filtered source RNA and ATAC data
    """
    
    import squidpy as sq
    from nichecompass.utils import get_unique_genes_from_gp_dict
    
    if filter_genes:
        print("Filtering genes...")
        # Filter genes and only keep ligand, receptor, enzyme, sensor, and
        # the 'n_svg' spatially variable genes
        gp_dict_genes = get_unique_genes_from_gp_dict(
            gp_dict=combined_gp_dict,
            retrieved_gene_entities=["sources", "targets"])
        print(f"Starting with {len(adata.var_names)} genes.")
        min_cells = int(adata.shape[0] * min_cell_gene_thresh_ratio)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        print(f"Keeping {len(adata.var_names)} genes after filtering genes with "
              f"counts in less than {int(adata.shape[0] * min_cell_gene_thresh_ratio)} cells.")
        
        # Identify spatially variable genes
        sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
        svg_genes = adata.uns["moranI"].index[:n_svg].tolist()
        adata.var["spatially_variable"] = adata.var_names.isin(svg_genes)
        adata = adata[:, adata.var["spatially_variable"] == True]
        print(f"Keeping {len(adata.var_names)} spatially variable genes.")
        
    if filter_peaks:
        print("\nFiltering peaks...")
        print(f"Starting with {len(adata_atac.var_names)} peaks.")
        # Filter out peaks that are rarely detected to reduce GPU footprint of model
        min_cells = int(adata_atac.shape[0] * min_cell_peak_thresh_ratio)
        sc.pp.filter_genes(adata_atac, min_cells=min_cells)
        print(f"Keeping {len(adata_atac.var_names)} peaks after filtering peaks with "
              f"counts in less than {int(adata_atac.shape[0] * min_cell_peak_thresh_ratio)} cells.")
        
        # Filter spatially variable peaks
        adata_atac.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"]
        adata_atac.obsp["spatial_distances"] = adata.obsp["spatial_distances"]

        sq.gr.spatial_autocorr(adata_atac,
                            mode="moran",
                            genes=adata_atac.var_names)
        sv_peaks = adata_atac.uns["moranI"].index[:n_svp].tolist()
        adata_atac.var["spatially_variable"] = adata_atac.var_names.isin(sv_peaks)
        adata_atac = adata_atac[:, adata_atac.var["spatially_variable"] == True]
        print(f"Keeping {len(adata_atac.var_names)} peaks after filtering spatially variable "
            f"peaks.")

    print("\n WARNING: genes currently not filtered by GP, 'gp_dict_genes' not used.")
    
    return adata, adata_atac


def build_combined_gp_dict_mouse_brain(
    *,
    species: str,
    omnipath_lr_network_file_path: str,
    nichenet_lr_network_file_path: str,
    nichenet_ligand_target_matrix_file_path: str,
    mebocost_enzyme_sensor_interactions_folder_path: str,
    collectri_tf_network_file_path: str,
    gene_orthologs_mapping_file_path: str,
    nichenet_version: str = "v2",
    keep_target_genes_ratio: float = 1.0,
    max_n_target_genes_per_gp: int = 250,
    verbose: bool = True,
):
    """Build the default combined GP dictionary used in the mouse brain tutorial."""

    from nichecompass.utils import (
        extract_gp_dict_from_collectri_tf_network,
        extract_gp_dict_from_mebocost_ms_interactions,
        extract_gp_dict_from_nichenet_lrt_interactions,
        extract_gp_dict_from_omnipath_lr_interactions,
        filter_and_combine_gp_dict_gps_v2,
    )

    omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
        species=species,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=omnipath_lr_network_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        plot_gp_gene_count_distributions=False,
    )

    # NicheNet supports load-from-disk if the prepared CSV is already present
    save_load_kwargs = {
        "lr_network_file_path": nichenet_lr_network_file_path,
        "ligand_target_matrix_file_path": nichenet_ligand_target_matrix_file_path,
        "gene_orthologs_mapping_file_path": gene_orthologs_mapping_file_path,
    }
    if os.path.exists(nichenet_lr_network_file_path) and os.path.exists(
        nichenet_ligand_target_matrix_file_path
    ):
        save_load_kwargs.update({"load_from_disk": True, "save_to_disk": False})
    else:
        save_load_kwargs.update({"load_from_disk": False, "save_to_disk": True})

    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=species,
        version=nichenet_version,
        keep_target_genes_ratio=keep_target_genes_ratio,
        max_n_target_genes_per_gp=max_n_target_genes_per_gp,
        plot_gp_gene_count_distributions=False,
        **save_load_kwargs,
    )

    mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
        dir_path=mebocost_enzyme_sensor_interactions_folder_path,
        species=species,
        plot_gp_gene_count_distributions=False,
    )

    collectri_gp_dict = extract_gp_dict_from_collectri_tf_network(
        species=species,
        tf_network_file_path=collectri_tf_network_file_path,
        load_from_disk=False,
        save_to_disk=True,
        plot_gp_gene_count_distributions=False,
    )

    gp_dicts = [
        omnipath_gp_dict,
        nichenet_gp_dict,
        mebocost_gp_dict,
        collectri_gp_dict,
    ]
    combined_gp_dict = filter_and_combine_gp_dict_gps_v2(gp_dicts, verbose=verbose)

    if verbose:
        print(f"Number of gene programs after filtering and combining: {len(combined_gp_dict)}")

    return combined_gp_dict


def rebuild_nichecompass_multimodal_masks(
    *,
    adata: ad.AnnData,
    adata_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
    combined_gp_dict: dict,
    gp_targets_mask_key: str,
    gp_targets_categories_mask_key: str,
    gp_sources_mask_key: str,
    gp_sources_categories_mask_key: str,
    gp_names_key: str,
    adj_key: str = "spatial_connectivities",
    filter_peaks_based_on_genes: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData, ad.AnnData, ad.AnnData]:
    """Rebuild GP + chromatin-accessibility masks after alignment.

    This mirrors the previously inlined tutorial block that was commented out.
    """

    from nichecompass.utils import (
        add_gps_from_gp_dict_to_adata,
        add_multimodal_mask_to_adata,
        generate_multimodal_mapping_dict,
    )

    # Defensive cleanup (idempotent)
    for _k in [
        gp_targets_mask_key,
        gp_targets_categories_mask_key,
        gp_sources_mask_key,
        gp_sources_categories_mask_key,
    ]:
        if _k in adata.varm:
            del adata.varm[_k]
    if gp_names_key in adata.uns:
        del adata.uns[gp_names_key]

    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_gp_dict,
        adata=adata,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        gp_names_key=gp_names_key,
        min_genes_per_gp=2,
        min_source_genes_per_gp=0,
        min_target_genes_per_gp=1,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None,
        plot_gp_gene_count_distributions=False,
    )

    gene_peak_mapping_dict = generate_multimodal_mapping_dict(adata=adata, adata_atac=adata_atac)

    adata, adata_atac = add_multimodal_mask_to_adata(
        adata=adata,
        adata_atac=adata_atac,
        gene_peak_mapping_dict=gene_peak_mapping_dict,
        filter_peaks_based_on_genes=filter_peaks_based_on_genes,
    )

    if filter_peaks_based_on_genes:
        remaining_peaks = target_atac.var_names.isin(adata_atac.var_names)
        target_atac = target_atac[:, remaining_peaks]

    # Keep connectivities consistent across modalities
    if adj_key in adata.obsp:
        adata_atac.obsp[adj_key] = adata.obsp[adj_key].copy()

    # Sanity checks: aligned feature spaces must match
    assert adata.var_names.equals(target_rna.var_names), "RNA data must have the same gene names"
    assert adata_atac.var_names.equals(target_atac.var_names), "ATAC data must have the same peak names"

    # Copy masks/indices to target and set target spatial_connectivities
    target_rna, target_atac = finalize_target_after_alignment(
        adata=adata,
        adata_atac=adata_atac,
        target_rna=target_rna,
        target_atac=target_atac,
        adj_type="knn",
    )

    return adata, adata_atac, target_rna, target_atac
