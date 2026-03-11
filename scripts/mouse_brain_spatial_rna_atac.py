#!/usr/bin/env python
#%%
import argparse
import os
import shutil
import socket
import sys
from datetime import datetime
from pprint import pprint

# Python 3.7 compatibility for muon/mudata (they use typing.Literal in newer versions)
if sys.version_info < (3, 8):
    import typing
    from typing_extensions import Literal

    typing.Literal = Literal

#%% load env variables from .env file
from dotenv import dotenv_values, load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")
print("Loaded environment variables from .env or env:", end="\n\n")
pprint(dotenv_values("/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"))

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

# Ensure this script imports the local repo package, not site-packages.
BAKLAVA_BASE_DIR = os.getenv("BAKLAVA_BASE_DIR")
REPO_ROOT = os.path.join(BAKLAVA_BASE_DIR, "MultiGATE")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Make sure env-local binaries (e.g., bedtools) are discoverable when running
# with an explicit python path instead of an activated conda shell.
env_bin = os.path.dirname(sys.executable)
current_path_entries = os.environ.get("PATH", "").split(os.pathsep)
if env_bin and env_bin not in current_path_entries:
    os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import mlflow
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
import tempfile

import MultiGATE
from MultiGATE.MultiGATE import MultiGATE as MultiGATETrainer
print("Using MultiGATE module:", MultiGATE.__file__)

import warnings
warnings.filterwarnings("ignore")


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(description="Train MultiGATE on source and run live zero-shot eval on target.")
    parser.add_argument(
        "--target-subsample-n",
        type=int,
        default=5000,
        help="Maximum number of paired target cells to keep for live evaluation.",
    )
    parser.add_argument(
        "--target-subsample-seed",
        type=int,
        default=0,
        help="Random seed used when subsampling target cells.",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=500,
        help="Number of epochs to train the model for stage 1.",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=10,
        help="Number of teacher-student distillation epochs on target data for stage 2.",
    )
    parser.add_argument(
        "--lambda-kd",
        type=float,
        default=1.0,
        help="Scale factor for KD objectives (stage-1 dual-source KD and stage-2 target KD).",
    )
    parser.add_argument(
        "--stage1-dual-source-kd",
        action="store_true",
        default=False,
        help=(
            "If set, train a stage-1 source teacher (standard losses, spatial graph) and a stage-1 "
            "source student (KD-only, graph from --spatial-graph-type)."
        ),
    )
    parser.add_argument(
        "--log-mudata-umaps",
        action="store_true",
        default=False,
        help="If set, log source/target MuData UMAP artifacts in addition to concat AnnData UMAPs.",
    )
    parser.add_argument(
        "--source-label-key",
        type=str,
        default=None,
        help="Optional source label key in .obs for scib metrics. Falls back to pseudo labels if missing.",
    )
    parser.add_argument(
        "--target-label-key",
        type=str,
        default=None,
        help="Optional target label key in .obs for scib metrics. Falls back to pseudo labels if missing.",
    )
    parser.add_argument(
        "--scib-n-jobs",
        type=int,
        default=1,
        help="Number of jobs for scib-metrics neighbor search.",
    )
    parser.add_argument(
        "--vgp-mode",
        type=str,
        choices=["zero", "feature"],
        default="zero",
        help=(
            "How to initialise GP gating vectors (vgp0/vgp1) when transferring to the target. "
            "'zero': zero out vgp (prior-only GP attention, Option 1). "
            "'feature': transfer trained feature-anchored vgp directly (Option 2)."
        ),
    )
    parser.add_argument(
        "--spatial-graph-type",
        type=str,
        choices=["spatial", "knn", "identity", "tangram"],
        default="identity",
        help="Type of graph to use for MultiGATE.",
    )
    parser.add_argument(
        "--switcharoo",
        action="store_true",
        default=False,
        help="If set, swap source and target.",
    )
    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


def _to_dense_df(adata):
    if isinstance(adata.X, np.ndarray):
        matrix = adata.X
    else:
        matrix = adata.X.toarray()
    return pd.DataFrame(matrix, index=adata.obs.index, columns=adata.var.index)


def prepare_graph_data(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape)


def build_graph_inputs(adata_vars1, adata_vars2, bp_width=450, graph_type="ATAC", protein_value=0.001):

    x1 = _to_dense_df(adata_vars1)
    x2 = _to_dense_df(adata_vars2)

    cells = np.array(x1.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))

    genes = np.array(x1.columns)
    peaks = np.array(x2.columns)
    genes_id_tran = dict(zip(genes, range(genes.shape[0])))
    peaks_id_tran = dict(zip(peaks, range(peaks.shape[0])))

    if "Spatial_Net" not in adata_vars1.uns:
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    spatial_net = adata_vars1.uns["Spatial_Net"]
    graph_df = spatial_net.copy()
    graph_df["Cell1"] = graph_df["Cell1"].map(cells_id_tran)
    graph_df["Cell2"] = graph_df["Cell2"].map(cells_id_tran)
    graph_df = graph_df.dropna(subset=["Cell1", "Cell2"])
    graph_df[["Cell1", "Cell2"]] = graph_df[["Cell1", "Cell2"]].astype(int)

    graph = sp.coo_matrix(
        (np.ones(graph_df.shape[0]), (graph_df["Cell1"], graph_df["Cell2"])),
        shape=(adata_vars1.n_obs, adata_vars1.n_obs),
    )
    graph_tf = prepare_graph_data(graph)

    if "gene_peak_Net" not in adata_vars1.uns:
        raise ValueError("gene_peak_Net is not existed! Run Cal_gene_peak_Net first!")

    gene_peak_net = adata_vars1.uns["gene_peak_Net"]
    if graph_type == "protein":
        gene_peak_net = gene_peak_net.copy()
        gene_peak_net.columns = ["Gene", "Peak"]

    gp_df = gene_peak_net.copy()
    gp_df["Gene"] = gp_df["Gene"].map(genes_id_tran)
    gp_df["Peak"] = gp_df["Peak"].map(peaks_id_tran)
    gp_df = gp_df.dropna(subset=["Gene", "Peak"]).copy()
    if gp_df.empty:
        raise ValueError("gene_peak_Net does not overlap with selected RNA/ATAC features.")

    gp_df["Gene"] = gp_df["Gene"].astype(int)
    gp_df["Peak"] = gp_df["Peak"].astype(int) + adata_vars1.n_vars

    if graph_type in ["ATAC", "ATAC_RNA"]:
        dist = gp_df["Distance"].astype(float)
        gp_bp_width = bp_width if graph_type == "ATAC" else 2000
        weights = np.concatenate(
            (
                ((dist + gp_bp_width) / gp_bp_width) ** (-0.75),
                ((dist + gp_bp_width) / gp_bp_width) ** (-0.75),
            ),
            axis=0,
        )
    else:
        weights = np.ones(gp_df.shape[0] * 2) * protein_value

    gp_graph = sp.coo_matrix(
        (
            weights,
            (
                np.concatenate((gp_df["Gene"], gp_df["Peak"]), axis=0),
                np.concatenate((gp_df["Peak"], gp_df["Gene"]), axis=0),
            ),
        ),
        shape=(adata_vars1.n_vars + adata_vars2.n_vars, adata_vars1.n_vars + adata_vars2.n_vars),
    )
    gp_graph_tf = prepare_graph_data(gp_graph)

    return graph_tf, gp_graph_tf, x1, x2


def set_multigate_embeddings(adata1, adata2, embeddings_rna, embeddings_atac, key_added="MultiGATE"):
    adata1.obsm[key_added] = embeddings_rna
    adata2.obsm[key_added] = embeddings_atac

    norm2 = Normalizer(norm="l2")
    clip_all = (
        norm2.fit_transform(embeddings_rna)
        + norm2.fit_transform(embeddings_atac)
    ) / 2.0
    adata1.obsm[key_added + "_clip_all"] = clip_all
    adata2.obsm[key_added + "_clip_all"] = clip_all


def build_knn_graph_as_spatial_net(adata, n_neighbors=15):
    # Build a generic kNN cell graph for non-spatial data and store it in the
    # format expected by MultiGATE.forward_MultiGATE (adata.uns['Spatial_Net']).
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    conn = adata.obsp["connectivities"].tocoo()
    mask = conn.row != conn.col
    adata.uns["Spatial_Net"] = pd.DataFrame(
        {
            "Cell1": adata.obs_names[conn.row[mask]].to_numpy(),
            "Cell2": adata.obs_names[conn.col[mask]].to_numpy(),
            "Distance": np.zeros(int(mask.sum()), dtype=float),
        }
    )


def build_graph_tf_from_spatial_net(spatial_net, obs_names):
    if not {"Cell1", "Cell2"}.issubset(set(spatial_net.columns)):
        raise ValueError("Spatial_Net must contain columns {'Cell1', 'Cell2'}.")

    cells = np.asarray(obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))

    graph_df = spatial_net.copy()
    graph_df["Cell1"] = graph_df["Cell1"].map(cells_id_tran)
    graph_df["Cell2"] = graph_df["Cell2"].map(cells_id_tran)
    graph_df = graph_df.dropna(subset=["Cell1", "Cell2"])

    if graph_df.empty:
        graph = sp.coo_matrix((len(cells), len(cells)))
        return prepare_graph_data(graph)

    graph_df[["Cell1", "Cell2"]] = graph_df[["Cell1", "Cell2"]].astype(int)
    graph = sp.coo_matrix(
        (np.ones(graph_df.shape[0]), (graph_df["Cell1"], graph_df["Cell2"])),
        shape=(len(cells), len(cells)),
    )
    return prepare_graph_data(graph)


def build_source_student_graph_tf(source_rna, spatial_graph_type, knn_neighbors=15):
    if spatial_graph_type == "spatial":
        if "Spatial_Net" not in source_rna.uns:
            raise ValueError("Source Spatial_Net is missing for stage-1 dual-source KD.")
        return build_graph_tf_from_spatial_net(source_rna.uns["Spatial_Net"], source_rna.obs_names)

    if spatial_graph_type == "knn":
        # Build kNN edges on source cells without mutating the main source AnnData.
        tmp_adata = AnnData(X=source_rna.X, obs=source_rna.obs.copy())
        build_knn_graph_as_spatial_net(tmp_adata, n_neighbors=knn_neighbors)
        return build_graph_tf_from_spatial_net(tmp_adata.uns["Spatial_Net"], source_rna.obs_names)

    if spatial_graph_type == "identity":
        identity_net = pd.DataFrame(columns=["Cell1", "Cell2", "Distance"])
        return build_graph_tf_from_spatial_net(identity_net, source_rna.obs_names)

    raise ValueError(
        "Unsupported --spatial-graph-type '{}' for stage-1 source student graph.".format(spatial_graph_type)
    )


def build_zero_shot_target_trainer(source_trainer, target_spot_num, vgp_mode="zero"):
    # Rebuild MGATE with target N and load transferable weights.
    # vgp_mode="zero":    zero out vgp0/vgp1 (prior-only GP attention, Option 1).
    # vgp_mode="feature": transfer trained feature-anchored vgp directly (Option 2).
    target_trainer = MultiGATETrainer(
        hidden_dims1=source_trainer.mgate.hidden_dims1,
        hidden_dims2=source_trainer.mgate.hidden_dims2,
        spot_num=target_spot_num,
        temp=float(source_trainer.mgate.logit_scale.detach().cpu().item()),
        n_epochs=1,
        lr=source_trainer.lr,
        gradient_clipping=source_trainer.gradient_clipping,
        nonlinear=source_trainer.mgate.nonlinear,
        verbose=False,
        random_seed=0,
        config={"device": str(source_trainer.device)},
    )

    if vgp_mode == "feature":
        # vgp0/vgp1 are now (feat_num, 1) — same shape regardless of cell count, so load directly.
        state_dict = source_trainer.mgate.state_dict()
        target_trainer.mgate.load_state_dict(state_dict, strict=False)

    else:  # vgp_mode == "zero"
        state_dict = {
            key: value
            for key, value in source_trainer.mgate.state_dict().items()
            if key not in {"vgp0", "vgp1"}
        }
        target_trainer.mgate.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            target_trainer.mgate.vgp0.zero_()
            target_trainer.mgate.vgp1.zero_()

    return target_trainer


def pair_and_subsample_target(target_rna, target_atac, subsample_n, seed):
    shared_obs = target_rna.obs_names.intersection(target_atac.obs_names)
    if len(shared_obs) == 0:
        raise ValueError("Target RNA/ATAC share zero cells after preprocessing.")

    target_rna = target_rna[shared_obs].copy()
    target_atac = target_atac[shared_obs].copy()

    if target_rna.n_obs > subsample_n:
        rng = np.random.RandomState(seed)
        selected = np.array(target_rna.obs_names)[
            rng.choice(target_rna.n_obs, size=subsample_n, replace=False)
        ]
        target_rna = target_rna[selected].copy()
        target_atac = target_atac[selected].copy()

    return target_rna, target_atac


_SCIB_BACKEND = None


def require_scib_backend():
    global _SCIB_BACKEND
    if _SCIB_BACKEND is not None:
        return _SCIB_BACKEND

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("JAX_DISABLE_JIT", os.environ.get("BAKLAVA_JAX_DISABLE_JIT", "1"))

    try:
        from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
    except Exception as exc:
        raise ImportError(
            "Failed to import scib-metrics. Install missing dependencies in MultiGATEenv_py310_scib, "
            "e.g. `pip install chex scib-metrics` (or conda equivalents)."
        ) from exc

    _SCIB_BACKEND = {
        "Benchmarker": Benchmarker,
        "BioConservation": BioConservation,
        "BatchCorrection": BatchCorrection,
    }
    return _SCIB_BACKEND


def resolve_scib_labels(rna_adata, atac_adata, concat_adata, label_key, domain_name):
    if label_key is not None:
        if label_key in rna_adata.obs.columns and label_key in atac_adata.obs.columns:
            if label_key not in concat_adata.obs.columns:
                raise KeyError(
                    "Label key '{}' missing in concatenated obs for {} domain.".format(label_key, domain_name)
                )
            return label_key, "provided"
        warnings.warn(
            "Requested label key '{}' for {} not found in both RNA and ATAC obs. "
            "Falling back to pseudo labels.".format(label_key, domain_name)
        )

    sc.pp.neighbors(concat_adata, use_rep="X", n_neighbors=15, key_added="scib_eval")
    sc.tl.leiden(
        concat_adata,
        neighbors_key="scib_eval",
        key_added="scib_pseudo_leiden",
        resolution=1.5,
        random_state=0,
    )
    return "scib_pseudo_leiden", "pseudo_leiden"


def compute_scib_metrics_for_domain(
    rna_adata,
    atac_adata,
    domain_name,
    label_key=None,
    scib_n_jobs=1,
):
    scib_backend = require_scib_backend()
    Benchmarker = scib_backend["Benchmarker"]
    BioConservation = scib_backend["BioConservation"]
    BatchCorrection = scib_backend["BatchCorrection"]

    concat_adata = build_concat_adata_for_umap(rna_adata, atac_adata, embedding_key="MultiGATE")
    effective_label_key, label_mode = resolve_scib_labels(
        rna_adata=rna_adata,
        atac_adata=atac_adata,
        concat_adata=concat_adata,
        label_key=label_key,
        domain_name=domain_name,
    )

    concat_adata.obsm["multigate_latent"] = np.asarray(concat_adata.X)

    benchmarker = Benchmarker(
        adata=concat_adata,
        batch_key="modality",
        label_key=effective_label_key,
        embedding_obsm_keys=["multigate_latent"],
        bio_conservation_metrics=BioConservation(
            isolated_labels=False,
            nmi_ari_cluster_labels_leiden=False,
            nmi_ari_cluster_labels_kmeans=False,
            silhouette_label=True,
            clisi_knn=False,
        ),
        batch_correction_metrics=BatchCorrection(
            bras=False,
            ilisi_knn=True,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=False,
        ),
        pre_integrated_embedding_obsm_key="multigate_latent",
        n_jobs=scib_n_jobs,
        progress_bar=False,
    )
    benchmarker.benchmark()

    results = benchmarker.get_results(min_max_scale=False, clean_names=False)
    if "multigate_latent" not in results.index:
        raise KeyError("scib results missing expected embedding row 'multigate_latent'.")
    row = results.loc["multigate_latent"]

    metrics = {
        "label_mode": label_mode,
        "effective_label_key": effective_label_key,
    }

    if "silhouette_label" in row.index:
        metrics["silhouette_label"] = float(row["silhouette_label"])
    if "ilisi_knn" in row.index:
        metrics["ilisi"] = float(row["ilisi_knn"])
    if "bras" in row.index:
        metrics["bras"] = float(row["bras"])
    if "Bio conservation" in row.index:
        metrics["bio_conservation"] = float(row["Bio conservation"])
    if "Batch correction" in row.index:
        metrics["batch_correction"] = float(row["Batch correction"])
    if "Total" in row.index:
        metrics["total"] = float(row["Total"])

    return metrics


def log_scib_metrics(prefix, metrics, step):
    mapping = {
        "silhouette_label": "{}_scib_silhouette_label".format(prefix),
        "ilisi": "{}_scib_ilisi".format(prefix),
        "bras": "{}_scib_bras".format(prefix),
        "bio_conservation": "{}_scib_bio_conservation".format(prefix),
        "batch_correction": "{}_scib_batch_correction".format(prefix),
        "total": "{}_scib_total".format(prefix),
    }
    for key, metric_name in mapping.items():
        if key in metrics and np.isfinite(metrics[key]):
            mlflow.log_metric(metric_name, float(metrics[key]), step=step)


def log_umap_to_mlflow(mdata, artifact_path, title, color_key="wnn", size=20):
    if mdata.n_obs < 3:
        warnings.warn("Skipping UMAP artifact '{}' because n_obs < 3.".format(artifact_path))
        return
    if "X_umap" not in mdata.obsm:
        warnings.warn("Skipping UMAP artifact '{}' because X_umap is missing.".format(artifact_path))
        return

    umap_fig = None
    try:
        umap_fig, umap_ax = plt.subplots(figsize=(7, 5))
        plot_fn = mu.pl.umap if isinstance(mdata, mu.MuData) else sc.pl.umap
        plot_fn(mdata, color=color_key, title=title, ax=umap_ax, size=size, show=False)
        umap_fig.tight_layout()
        mlflow.log_figure(umap_fig, artifact_path)
    finally:
        if umap_fig is not None:
            plt.close(umap_fig)


def build_concat_adata_for_umap(rna_adata, atac_adata, embedding_key="MultiGATE"):
    if embedding_key not in rna_adata.obsm or embedding_key not in atac_adata.obsm:
        raise KeyError(
            "Missing '{}' in one or both modalities when building concat AnnData.".format(embedding_key)
        )

    rna_obs = rna_adata.obs.copy()
    atac_obs = atac_adata.obs.copy()
    rna_obs["modality"] = "rna"
    atac_obs["modality"] = "atac"
    rna_obs.index = rna_obs.index.astype(str) + "_rna"
    atac_obs.index = atac_obs.index.astype(str) + "_atac"

    concat_adata = AnnData(
        X=np.concatenate(
            [
                rna_adata.obsm[embedding_key],
                atac_adata.obsm[embedding_key],
            ],
            axis=0,
        ),
        obs=pd.concat([rna_obs, atac_obs], axis=0),
    )
    return concat_adata


def compute_concat_umap(concat_adata, n_neighbors=10, resolution=1.5):
    sc.pp.neighbors(concat_adata, n_neighbors=n_neighbors)
    sc.tl.umap(concat_adata)
    sc.tl.leiden(concat_adata, resolution=resolution)


def log_umap_panel_to_mlflow(adata, artifact_path, colors, titles, size=20):
    if adata.n_obs < 3:
        warnings.warn("Skipping UMAP artifact '{}' because n_obs < 3.".format(artifact_path))
        return
    if "X_umap" not in adata.obsm:
        warnings.warn("Skipping UMAP artifact '{}' because X_umap is missing.".format(artifact_path))
        return

    fig = None
    try:
        n_panels = len(colors)
        fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axs = [axs]
        for idx, (color_key, title) in enumerate(zip(colors, titles)):
            sc.pl.umap(
                adata,
                color=color_key,
                title=title,
                ax=axs[idx],
                size=size,
                show=False,
            )
        fig.tight_layout()
        mlflow.log_figure(fig, artifact_path)
    finally:
        if fig is not None:
            plt.close(fig)


def build_mudata_with_umap(rna_adata, atac_adata, embedding_key="MultiGATE", n_neighbors=10, resolution=1.5):
    if embedding_key not in rna_adata.obsm or embedding_key not in atac_adata.obsm:
        raise KeyError(
            "Missing '{}' in one or both modalities when building MuData.".format(embedding_key)
        )

    rna_eval = rna_adata.copy()
    atac_eval = atac_adata.copy()
    sc.pp.neighbors(rna_eval, use_rep=embedding_key, n_neighbors=n_neighbors)
    sc.pp.neighbors(atac_eval, use_rep=embedding_key, n_neighbors=n_neighbors)

    mdata = mu.MuData({"rna": rna_eval, "atac": atac_eval})
    mu.pp.intersect_obs(mdata)
    mu.pp.neighbors(mdata, n_neighbors=n_neighbors)
    mu.tl.umap(mdata)
    sc.tl.leiden(mdata, resolution=resolution)
    mdata.obs["wnn"] = mdata.obs["leiden"].astype(int).astype("category")
    return mdata


def log_stage_umap_artifacts(
    source_rna,
    source_atac,
    target_rna,
    target_atac,
    stage_label,
    log_mudata_umaps=False,
):
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE")
    target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key="MultiGATE")

    compute_concat_umap(source_concat_adata, n_neighbors=10, resolution=1.5)
    compute_concat_umap(target_concat_adata, n_neighbors=10, resolution=1.5)

    source_celltype_key = source_rna.uns['label_key']
    target_celltype_key = target_rna.uns['label_key']

    source_colors = ["modality", "leiden"]
    source_titles = [
        "Source Concat Modality ({})".format(stage_label),
        "Source Concat Leiden ({})".format(stage_label),
    ]
    if source_celltype_key is not None:
        source_colors.append(source_celltype_key)
        source_titles.append("Source Concat Cell Type ({})".format(stage_label))

    log_umap_panel_to_mlflow(
        source_concat_adata,
        artifact_path="umap/{}/source_concat_adata_umap.png".format(stage_label),
        colors=source_colors,
        titles=source_titles,
        size=20,
    )

    target_colors = ["modality", "leiden"]
    target_titles = [
        "Target Concat Modality ({})".format(stage_label),
        "Target Concat Leiden ({})".format(stage_label),
    ]
    if target_celltype_key is not None:
        target_colors.append(target_celltype_key)
        target_titles.append("Target Concat Cell Type ({})".format(stage_label))

    log_umap_panel_to_mlflow(
        target_concat_adata,
        artifact_path="umap/{}/target_concat_adata_umap.png".format(stage_label),
        colors=target_colors,
        titles=target_titles,
        size=20,
    )

    output = {
        "source_concat_adata": source_concat_adata,
        "target_concat_adata": target_concat_adata,
    }

    if log_mudata_umaps:
        source_mdata = build_mudata_with_umap(
            source_rna,
            source_atac,
            embedding_key="MultiGATE",
            n_neighbors=10,
            resolution=1.5,
        )
        target_mdata = build_mudata_with_umap(
            target_rna,
            target_atac,
            embedding_key="MultiGATE",
            n_neighbors=10,
            resolution=1.5,
        )

        log_umap_to_mlflow(
            source_mdata,
            artifact_path="umap/{}/source_mudata_umap.png".format(stage_label),
            title="Source MuData UMAP ({})".format(stage_label),
            color_key="wnn",
            size=20,
        )
        log_umap_to_mlflow(
            target_mdata,
            artifact_path="umap/{}/target_mudata_umap.png".format(stage_label),
            title="Target MuData UMAP ({})".format(stage_label),
            color_key="wnn",
            size=20,
        )

        output["source_mdata"] = source_mdata
        output["target_mdata"] = target_mdata

    return output


def require_ot_backend():
    try:
        from ot import emd
    except Exception as exc:
        raise ImportError(
            "Stage-2 KD requires POT (`ot`). Install it in MultiGATEenv, e.g. "
            "`pip install POT` or `conda install -c conda-forge pot`."
        ) from exc
    return emd


def compute_clip_logits(clip_rna, clip_atac, logit_scale):
    return torch.matmul(clip_rna, clip_atac.transpose(0, 1)) * torch.exp(logit_scale)


def compute_clip_loss_from_logits(logits):
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_rna = F.cross_entropy(logits, labels, reduction="none")
    loss_atac = F.cross_entropy(logits.transpose(0, 1), labels, reduction="none")
    return ((loss_rna + loss_atac) / 2.0).mean()


def compute_kd_kl_loss(student_logits, teacher_logits):
    kd_cols = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.log_softmax(teacher_logits, dim=1),
        reduction="batchmean",
        log_target=True,
    ).clamp(min=0)
    kd_rows = F.kl_div(
        F.log_softmax(student_logits, dim=0),
        F.log_softmax(teacher_logits, dim=0),
        reduction="batchmean",
        log_target=True,
    ).clamp(min=0)

    if not (kd_cols.is_nonzero() and kd_rows.is_nonzero()):
        print("[WARNING] zero-valued KL divergence, temperature too high.")
    return 0.5 * (kd_cols + kd_rows)


def compute_ot_clip_loss(student_logits, teacher_logits, emd):
    one = torch.tensor(1.0, device=teacher_logits.device, dtype=teacher_logits.dtype)
    teacher_cost = 0.5 * (2 - (teacher_logits / torch.exp(1 / one)))

    teacher_cost_np = teacher_cost.detach().cpu().numpy()
    plan = emd(a=[], b=[], M=teacher_cost_np)
    plan_t = emd(a=[], b=[], M=teacher_cost_np.T)

    plan = torch.as_tensor(plan, device=student_logits.device, dtype=student_logits.dtype)
    plan_t = torch.as_tensor(plan_t, device=student_logits.device, dtype=student_logits.dtype)

    labels = torch.argmax(plan, dim=1)
    labels_t = torch.argmax(plan_t, dim=1)

    ot_clip_loss = F.cross_entropy(student_logits, labels, reduction="none")
    ot_clip_loss_t = F.cross_entropy(student_logits.transpose(0, 1), labels_t, reduction="none")
    return 0.5 * (ot_clip_loss.mean() + ot_clip_loss_t.mean())


def run_stage2_distillation(
    source_trainer,
    target_rna,
    target_atac,
    target_graph_tf,
    target_gp_tf,
    target_x1,
    target_x2,
    stage2_epochs,
    lambda_kd,
    target_label_key,
    scib_n_jobs,
    vgp_mode="zero",
):
    if stage2_epochs <= 0:
        return None

    emd = require_ot_backend()

    teacher_trainer = build_zero_shot_target_trainer(source_trainer, target_rna.n_obs, vgp_mode=vgp_mode)
    teacher_model = teacher_trainer.mgate
    teacher_model.eval()
    for teacher_param in teacher_model.parameters():
        teacher_param.requires_grad = False

    student_trainer = MultiGATETrainer(
        hidden_dims1=source_trainer.mgate.hidden_dims1,
        hidden_dims2=source_trainer.mgate.hidden_dims2,
        spot_num=target_rna.n_obs,
        temp=float(source_trainer.mgate.logit_scale.detach().cpu().item()),
        n_epochs=stage2_epochs,
        lr=source_trainer.lr,
        gradient_clipping=source_trainer.gradient_clipping,
        nonlinear=source_trainer.mgate.nonlinear,
        verbose=False,
        random_seed=2021,
        config={"device": str(source_trainer.device)},
    )

    target_a_t, target_prune_t, target_gp_t, target_x1_t, target_x2_t = student_trainer._prepare_inputs(
        target_graph_tf,
        target_graph_tf,
        target_gp_tf,
        target_x1,
        target_x2,
    )

    parent_run = mlflow.active_run()
    parent_run_name = None
    if parent_run is not None:
        parent_run_name = parent_run.data.tags.get("mlflow.runName")
    stage2_run_name = "stage2_distillation" if not parent_run_name else "{}_stage2".format(parent_run_name)

    with mlflow.start_run(run_name=stage2_run_name, nested=True):
        mlflow.set_tag("training_stage", "stage2_distillation")
        mlflow.set_tag("teacher_student_distillation", "true")
        mlflow.log_param("stage2_epochs", stage2_epochs)
        mlflow.log_param("lambda_kd", lambda_kd)
        mlflow.log_param("kd_mix_kl", 0.1)
        mlflow.log_param("kd_mix_ot", 0.9)
        mlflow.log_param("student_init", "random")
        mlflow.log_param("teacher_init", "source_to_target_zero_shot_vgp0")
        mlflow.log_param("stage2_target_label_key", target_label_key if target_label_key is not None else "None")
        target_scib_label_mode_logged = False
        target_scib_effective_label_key_logged = False

        pbar = tqdm(range(1, stage2_epochs + 1), desc="Stage 2 distillation", unit="epoch")
        for epoch in pbar:
            student_trainer.mgate.train()
            student_trainer.optimizer.zero_grad(set_to_none=True)

            student_outputs = student_trainer.mgate(target_a_t, target_prune_t, target_gp_t, target_x1_t, target_x2_t)
            with torch.no_grad():
                teacher_outputs = teacher_model(target_a_t, target_prune_t, target_gp_t, target_x1_t, target_x2_t)

            student_clip_rna, student_clip_atac = student_outputs[5], student_outputs[6]
            teacher_clip_rna, teacher_clip_atac = teacher_outputs[5], teacher_outputs[6]

            student_logits = compute_clip_logits(student_clip_rna, student_clip_atac, student_trainer.mgate.logit_scale)
            teacher_logits = compute_clip_logits(teacher_clip_rna, teacher_clip_atac, teacher_model.logit_scale)

            kd_ot_loss = compute_ot_clip_loss(student_logits, teacher_logits, emd=emd)
            kd_kl_loss = compute_kd_kl_loss(student_logits, teacher_logits)
            kd_kl_loss = kd_kl_loss * 50 # TMP - bring KL loss to same scale as OT loss
            distill_loss = lambda_kd * (0.1 * kd_kl_loss + 0.9 * kd_ot_loss)

            distill_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_trainer.mgate.parameters(), student_trainer.gradient_clipping)
            student_trainer.optimizer.step()

            model_clip_loss = student_outputs[4]
            reconstructed_clip_loss = compute_clip_loss_from_logits(student_logits)
            clip_parity_absdiff = torch.abs(model_clip_loss - reconstructed_clip_loss).detach().cpu().item()

            mlflow.log_metric("stage2_distill_loss", float(distill_loss.detach().cpu().item()), step=epoch)
            mlflow.log_metric("stage2_kd_kl_loss", float(kd_kl_loss.detach().cpu().item()), step=epoch)
            mlflow.log_metric("stage2_kd_ot_clip_loss", float(kd_ot_loss.detach().cpu().item()), step=epoch)
            mlflow.log_metric("stage2_clip_logits_parity_absdiff", float(clip_parity_absdiff), step=epoch)

            loss_val = float(distill_loss.detach().cpu().item())
            pbar.set_postfix({"distill_loss": "{:.4f}".format(loss_val)})

            if (epoch % 100 == 0) or (epoch == stage2_epochs):

                set_multigate_embeddings(
                    target_rna,
                    target_atac,
                    student_clip_rna.detach().cpu().numpy(),
                    student_clip_atac.detach().cpu().numpy(),
                    key_added="MultiGATE",
                )

                # compute and log scib metrics for target data
                target_scib_metrics = compute_scib_metrics_for_domain(
                    rna_adata=target_rna,
                    atac_adata=target_atac,
                    domain_name="target",
                    label_key=target_label_key,
                    scib_n_jobs=scib_n_jobs,
                )
                log_scib_metrics(prefix="stage2_target", metrics=target_scib_metrics, step=epoch)
                if not target_scib_label_mode_logged:
                    mlflow.log_param("stage2_target_scib_label_mode", target_scib_metrics["label_mode"])
                    target_scib_label_mode_logged = True
                if not target_scib_effective_label_key_logged:
                    mlflow.log_param(
                        "stage2_target_scib_effective_label_key",
                        target_scib_metrics["effective_label_key"],
                    )
                    target_scib_effective_label_key_logged = True

            del student_outputs, teacher_outputs
            del student_clip_rna, student_clip_atac, teacher_clip_rna, teacher_clip_atac
            del student_logits, teacher_logits, kd_ot_loss, kd_kl_loss, distill_loss
            del model_clip_loss, reconstructed_clip_loss

    final_target_embeddings = student_trainer.infer(
        target_graph_tf,
        target_graph_tf,
        target_gp_tf,
        target_x1,
        target_x2,
    )
    set_multigate_embeddings(
        target_rna,
        target_atac,
        final_target_embeddings[0],
        final_target_embeddings[1],
        key_added="MultiGATE",
    )

    return student_trainer


def setup_mlflow():
    # Default to a clean tracking location under BAKLAVA_base so we don't
    # accidentally write into other repos' local `mlruns/` directories.
    baklava_base_dir = os.path.dirname(REPO_ROOT)
    default_mlflow_base_dir = os.path.join(baklava_base_dir, "mlflow_tracking", "MultiGATE")

    # If a global MLFLOW_BASE_DIR is set (e.g. by BAKLAVA's `.env`), namespace
    # MultiGATE runs under a dedicated subdirectory to avoid schema/version
    # conflicts with other MLflow usage.
    env_mlflow_base_dir = os.environ.get("MLFLOW_BASE_DIR")
    if env_mlflow_base_dir:
        mlflow_base_dir = os.path.abspath(env_mlflow_base_dir)
        if os.path.basename(mlflow_base_dir.rstrip(os.sep)) != "MultiGATE":
            mlflow_base_dir = os.path.join(mlflow_base_dir, "MultiGATE")
    else:
        mlflow_base_dir = os.path.abspath(default_mlflow_base_dir)
    os.makedirs(mlflow_base_dir, exist_ok=True)

    mlflow_db_path = os.path.join(mlflow_base_dir, "mlflow.db")
    tracking_uri = "sqlite:///{}".format(mlflow_db_path)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    print("MLflow backend-store-uri:", tracking_uri)
    print("MLflow base dir:", mlflow_base_dir)

    experiment_name = "multigate_mouse_brain_live_zeroshot"
    artifact_dir = os.path.join(mlflow_base_dir, "mlflow_artifacts", experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=os.path.abspath(artifact_dir),
        )
        print("Created MLflow experiment: {} (ID: {})".format(experiment_name, experiment_id))
    else:
        experiment_id = experiment.experiment_id
        print("Using MLflow experiment: {} (ID: {})".format(experiment_name, experiment_id))

    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id

def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        else:
            # Other types
            return False
    except Exception:
        return False

#%%
def main():
#%%
    NOTEBOOK = is_notebook()
    args = parse_args(notebook=NOTEBOOK)
    if args.target_subsample_n <= 0:
        raise ValueError("--target-subsample-n must be a positive integer.")
    if args.stage2_epochs < 0:
        raise ValueError("--stage2-epochs must be a non-negative integer.")
    if args.lambda_kd < 0:
        raise ValueError("--lambda-kd must be non-negative.")
    if args.scib_n_jobs <= 0:
        raise ValueError("--scib-n-jobs must be a positive integer.")
    if args.stage1_dual_source_kd and args.spatial_graph_type == "tangram":
        raise ValueError(
            "--stage1-dual-source-kd with --spatial-graph-type tangram is not supported for source training. "
            "Use spatial, knn, or identity."
        )

    # Fail fast if scib-metrics backend is not available.
    require_scib_backend()
    stage1_emd = require_ot_backend() if args.stage1_dual_source_kd else None

    #%% load data

    # SOURCE
    source_rna = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned.h5ad"))
    source_atac = sc.read_h5ad(os.path.join(base_path, "source_atac_aligned.h5ad"))

    source_rna.obsm["spatial"] = source_rna.obsm["spatial"][:, [1, 0]] * -1
    source_atac.obsm["spatial"] = source_atac.obsm["spatial"][:, [1, 0]] * -1

    # TARGET
    target_rna = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned.h5ad"))
    target_atac = sc.read_h5ad(os.path.join(base_path, "target_atac_aligned.h5ad"))
    assert target_rna.obs_names.equals(target_atac.obs_names), "Target RNA and ATAC must have matching obs_names"

    # Set celltype keys
    source_rna.uns['label_key'] = 'RNA_clusters'
    target_rna.uns['label_key'] = 'REF_arc_gex_graphclust_Cluster'

    #%% Scale target data to match source data mean expression levels
    import anndata as ad
    
    def aggregate_cells(adata, groupby, agg="sum", layer=None):
        X = adata.layers[layer] if layer is not None else adata.X

        groups = adata.obs[groupby].astype("category")
        codes = groups.cat.codes.to_numpy()
        cats = groups.cat.categories

        G = sp.csr_matrix(
            (np.ones(adata.n_obs, dtype=np.float32), (codes, np.arange(adata.n_obs))),
            shape=(len(cats), adata.n_obs),
        )

        X_agg = G @ X

        if agg == "mean":
            counts = np.bincount(codes, minlength=len(cats)).astype(np.float32)
            counts[counts == 0] = 1.0
            X_agg = sp.diags(1.0 / counts) @ X_agg if sp.issparse(X_agg) else X_agg / counts[:, None]
        elif agg != "sum":
            raise ValueError("agg must be 'sum' or 'mean'")

        obs = pd.DataFrame(index=cats.astype(str))
        obs[groupby] = cats.astype(str)

        return ad.AnnData(X=X_agg, obs=obs, var=adata.var.copy())

    n = target_rna.n_obs
    group_size = 5

    perm = np.random.permutation(n)
    group_ids = np.arange(n) // group_size
    pseudo_ids = np.empty(n, dtype=int)
    pseudo_ids[perm] = group_ids

    target_rna.obs["pseudo_pixel"] = pseudo_ids.astype(str)
    target_atac.obs["pseudo_pixel"] = pseudo_ids.astype(str)

    target_rna_pb = aggregate_cells(
        target_rna,
        groupby="pseudo_pixel",
        agg="sum",
    )

    target_atac_pb = aggregate_cells(
        target_atac,
        groupby="pseudo_pixel",
        agg="sum",
    )

    target_rna = ad.AnnData(
        X=target_rna_pb.X,
        var=target_rna.var,
        obs = pd.DataFrame(index=target_rna_pb.obs_names),
        uns = target_rna.uns | {'label_key':None},
    )

    target_atac = ad.AnnData(
        X=target_atac_pb.X,
        var=target_atac.var,
        obs = pd.DataFrame(index=target_atac_pb.obs_names),
        uns = target_atac.uns | {'label_key':None},
    )

    #%% TMP - redo HVG to limit number of features to fit inside GPU memory
    if socket.gethostname() != "ri-muhc-gpu":
        source_rna.var["highly_variable"] = False
        source_atac.var["highly_variable"] = False

        target_rna.var["highly_variable"] = False
        target_atac.var["highly_variable"] = False

        top_n_genes = 2000
        top_n_peaks = 10000

        if 'highly_variable_rank' not in source_rna.var.columns:
            order = (-source_rna.var["dispersions_norm"]).argsort()
            rank = np.empty(len(order), dtype=order.dtype)
            rank[order] = np.arange(len(order))
            source_rna.var["highly_variable_rank"] = rank

            order = (-source_atac.var["dispersions_norm"]).argsort()
            rank = np.empty(len(order), dtype=order.dtype)
            rank[order] = np.arange(len(order))
            source_atac.var["highly_variable_rank"] = rank
            
        source_rna.var.loc[source_rna.var["highly_variable_rank"].le(top_n_genes - 1), "highly_variable"] = True
        source_atac.var.loc[source_atac.var["highly_variable_rank"].le(top_n_peaks - 1), "highly_variable"] = True

        target_rna.var.loc[
            target_rna.var_names.isin(source_rna.var_names[source_rna.var["highly_variable"]]),
            "highly_variable",
        ] = True
        target_atac.var.loc[
            target_atac.var_names.isin(source_atac.var_names[source_atac.var["highly_variable"]]),
            "highly_variable",
        ] = True

    #%% source spatial graph
    MultiGATE.Cal_Spatial_Net(source_rna, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_rna)

    MultiGATE.Cal_Spatial_Net(source_atac, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_atac)

    source_rna = source_rna[:, source_rna.var["highly_variable"]].copy()
    source_atac = source_atac[:, source_atac.var["highly_variable"]].copy()

    gtf_path = os.path.join(os.getenv("DATAPATH"), "gene_annotations", "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz")
    if not os.path.exists(gtf_path):
        raise FileNotFoundError("GTF annotation file not found: {}".format(gtf_path))

    MultiGATE.Cal_gene_peak_Net_new(source_rna, source_atac, 150000, file=gtf_path)
    source_rna.uns["gene_peak_Net"] = source_atac.uns["gene_peak_Net"]

    #%% target prep for live zero-shot eval

    if args.spatial_graph_type == "spatial":
        MultiGATE.Cal_Spatial_Net(target_rna, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Cal_Spatial_Net(target_atac, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_atac)
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        MultiGATE.Cal_gene_peak_Net_new(target_rna, target_atac, 150000, file=gtf_path)
        target_rna.uns["gene_peak_Net"] = target_atac.uns["gene_peak_Net"]

    elif args.spatial_graph_type == 'tangram':
        target_rna = target_rna[:, target_rna.var_names.isin(source_rna.var_names)].copy()
        target_atac = target_atac[:, target_atac.var_names.isin(source_atac.var_names)].copy()
        tangram_net = pd.read_csv(os.path.join(os.getenv("OUTPATH"), "tangram", "tangram_spatial_net_affinity.csv"))
        target_rna.uns['Spatial_Net'] = tangram_net.copy()
        target_atac.uns['Spatial_Net'] = tangram_net.copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        target_atac.uns["gene_peak_Net"] = source_atac.uns["gene_peak_Net"].copy()

    elif args.spatial_graph_type == "knn":
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        build_knn_graph_as_spatial_net(target_rna, n_neighbors=15)
        target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Stats_Spatial_Net(target_atac)

    elif args.spatial_graph_type == "identity":
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_rna.uns["Spatial_Net"] = pd.DataFrame(columns=['Cell1', 'Cell2', 'Distance'])
        target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
        #MultiGATE.Stats_Spatial_Net(target_rna)
        #MultiGATE.Stats_Spatial_Net(target_atac)

    print("[INFO] Pairing and subsampling target data...")
    target_rna, target_atac = pair_and_subsample_target(
        target_rna,
        target_atac,
        subsample_n=args.target_subsample_n,
        seed=args.target_subsample_seed,
    )

    #%% swap source and target

    if args.switcharoo:

        source_rna_tmp = source_rna.copy()
        source_atac_tmp = source_atac.copy()

        source_rna = target_rna.copy()
        source_atac = target_atac.copy()

        target_rna = source_rna_tmp.copy()
        target_atac = source_atac_tmp.copy()

    #%% Build reusable graph/data inputs
    bp_width = 400
    graph_type = "ATAC"
    protein_value = 0.001

    source_graph_tf, source_gp_tf, source_x1, source_x2 = build_graph_inputs(
        source_rna,
        source_atac,
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )
    target_graph_tf, target_gp_tf, target_x1, target_x2 = build_graph_inputs(
        target_rna,
        target_atac,
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )

    if target_x1.shape[1] != source_x1.shape[1] or target_x2.shape[1] != source_x2.shape[1]:
        raise ValueError(
            "Target feature dimensions do not match source model dimensions: "
            "RNA {} vs {}, ATAC {} vs {}.".format(
                target_x1.shape[1],
                source_x1.shape[1],
                target_x2.shape[1],
                source_x2.shape[1],
            )
        )

    #%% Build source trainer
    num_epochs = args.stage1_epochs
    if num_epochs <= 0:
        raise ValueError("--stage1-epochs must be a positive integer.")

    hidden_dims = [512, 30]
    trainer = MultiGATETrainer(
        hidden_dims1=[source_x1.shape[1]] + hidden_dims,
        hidden_dims2=[source_x2.shape[1]] + hidden_dims,
        spot_num=source_x1.shape[0],
        temp=1,
        n_epochs=num_epochs,
        lr=0.0001,
        gradient_clipping=5,
        nonlinear=True,
        weight_decay=0.0001,
        verbose=False,
        random_seed=2020,
    )

    source_a_t, source_prune_t, source_gp_t, source_x1_t, source_x2_t = trainer._prepare_inputs(
        source_graph_tf,
        source_graph_tf,
        source_gp_tf,
        source_x1,
        source_x2,
    )

    student_trainer = None
    source_student_graph_tf = None
    source_student_a_t = source_student_prune_t = source_student_gp_t = source_student_x1_t = source_student_x2_t = None

    if args.stage1_dual_source_kd:

        source_student_graph_tf = build_source_student_graph_tf(
            source_rna=source_rna,
            spatial_graph_type=args.spatial_graph_type,
            #knn_neighbors=15,
        )
        student_trainer = MultiGATETrainer(
            hidden_dims1=[source_x1.shape[1]] + hidden_dims,
            hidden_dims2=[source_x2.shape[1]] + hidden_dims,
            spot_num=source_x1.shape[0],
            temp=1,
            n_epochs=num_epochs,
            lr=0.0001,
            gradient_clipping=5,
            nonlinear=True,
            weight_decay=0.0001,
            verbose=False,
            random_seed=2021,
            config={"device": str(trainer.device)},
        )
        source_student_a_t, source_student_prune_t, source_student_gp_t, source_student_x1_t, source_student_x2_t = student_trainer._prepare_inputs(
            source_student_graph_tf,
            source_student_graph_tf,
            source_gp_tf,
            source_x1,
            source_x2,
        )

    stage1_primary_trainer = student_trainer if args.stage1_dual_source_kd else trainer
    stage1_primary_source_graph_tf = source_student_graph_tf if args.stage1_dual_source_kd else source_graph_tf
    stage1_primary_model_name = "student" if args.stage1_dual_source_kd else "teacher"

    #%% MLflow setup
    experiment_id = setup_mlflow()
    eval_every = 1000 # set to -1 for very basic debugging only, since will skip the incorporation of MultiGATE embeddings into the anndatas
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Training epochs for stage 1:", num_epochs)
    print("Target paired cells after subsampling:", target_rna.n_obs)
    if args.stage1_dual_source_kd:
        print(
            "[Stage1 Dual KD] Enabled: teacher graph=spatial, student graph={}".format(args.spatial_graph_type)
        )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("mlflow_experiment_id", experiment_id)
        mlflow.log_param("n_epochs", num_epochs)
        mlflow.log_param("stage2_epochs", args.stage2_epochs)
        mlflow.log_param("lambda_kd", args.lambda_kd)
        mlflow.log_param("bp_width", bp_width)
        mlflow.log_param("target_subsample_n", args.target_subsample_n)
        mlflow.log_param("target_subsample_seed", args.target_subsample_seed)
        mlflow.log_param("log_mudata_umaps", args.log_mudata_umaps)
        mlflow.log_param("source_label_key", args.source_label_key if args.source_label_key is not None else "None")
        mlflow.log_param("target_label_key", args.target_label_key if args.target_label_key is not None else "None")
        mlflow.log_param("target_effective_n", int(target_rna.n_obs))
        mlflow.log_param("eval_every", eval_every)
        mlflow.log_param("source_cells", int(source_rna.n_obs))
        mlflow.log_param("target_cells", int(target_rna.n_obs))
        mlflow.log_param("graph_type", graph_type)
        mlflow.log_param("stage1_dual_source_kd", bool(args.stage1_dual_source_kd))
        mlflow.log_param("stage1_primary_model", stage1_primary_model_name)
        mlflow.log_param("stage1_teacher_graph", "spatial")
        mlflow.log_param("stage1_student_graph", args.spatial_graph_type if args.stage1_dual_source_kd else "NA")
        source_scib_label_mode_logged = False
        target_scib_label_mode_logged = False
        source_scib_effective_label_key_logged = False
        target_scib_effective_label_key_logged = False

        for epoch in tqdm(range(1, num_epochs + 1), desc="Stage 1 training", unit="epoch"):

            teacher_loss = trainer.run_epoch(epoch, source_a_t, source_prune_t, source_gp_t, source_x1_t, source_x2_t)
            mlflow.log_metric("source_train_loss", float(teacher_loss), step=epoch)

            if args.stage1_dual_source_kd:

                trainer.mgate.eval()
                student_trainer.mgate.train()
                student_trainer.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    teacher_outputs = trainer.mgate(
                        source_a_t,
                        source_prune_t,
                        source_gp_t,
                        source_x1_t,
                        source_x2_t,
                    )

                student_outputs = student_trainer.mgate(
                    source_student_a_t,
                    source_student_prune_t,
                    source_student_gp_t,
                    source_student_x1_t,
                    source_student_x2_t,
                )

                student_clip_rna, student_clip_atac = student_outputs[5], student_outputs[6]
                teacher_clip_rna, teacher_clip_atac = teacher_outputs[5], teacher_outputs[6]
                '''
                student_logits = compute_clip_logits(
                    student_clip_rna,
                    student_clip_atac,
                    student_trainer.mgate.logit_scale,
                )
                teacher_logits = compute_clip_logits(
                    teacher_clip_rna,
                    teacher_clip_atac,
                    trainer.mgate.logit_scale,
                )
                '''
                #stage1_kd_kl_loss = compute_kd_kl_loss(student_logits, teacher_logits)
                stage1_distill_loss = F.mse_loss(student_clip_rna, teacher_clip_rna) + F.mse_loss(student_clip_atac, teacher_clip_atac)

                stage1_distill_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_trainer.mgate.parameters(), student_trainer.gradient_clipping)
                student_trainer.optimizer.step()

                mlflow.log_metric(
                    "stage1_student_distill_loss",
                    float(stage1_distill_loss.detach().cpu().item()),
                    step=epoch,
                )
                mlflow.log_metric(
                    "stage1_student_distill_loss",
                    float(stage1_distill_loss.detach().cpu().item()),
                    step=epoch,
                )
                del teacher_outputs, student_outputs
                del teacher_clip_rna, teacher_clip_atac, student_clip_rna, student_clip_atac
                del stage1_distill_loss

            should_eval = \
                ((epoch == 1) or (epoch % eval_every == 0) or (epoch == num_epochs)) and \
                eval_every > 0

            if not should_eval:
                continue

            source_embeddings = stage1_primary_trainer.infer( # if args.stage1_dual_source_kd is True, source_embeddings is the stage 1 student source embeddings. Else, source_embeddings is the teacher source embeddings.
                stage1_primary_source_graph_tf,
                stage1_primary_source_graph_tf,
                source_gp_tf,
                source_x1,
                source_x2,
            )
            set_multigate_embeddings(
                source_rna,
                source_atac,
                source_embeddings[0],
                source_embeddings[1],
                key_added="MultiGATE",
            )

            trainer_target = build_zero_shot_target_trainer(
                stage1_primary_trainer,
                target_rna.n_obs,
                vgp_mode=args.vgp_mode,
            )
            target_embeddings = trainer_target.infer(
                target_graph_tf,
                target_graph_tf,
                target_gp_tf,
                target_x1,
                target_x2,
            )
            set_multigate_embeddings(
                target_rna,
                target_atac,
                target_embeddings[0],
                target_embeddings[1],
                key_added="MultiGATE",
            )
            if epoch != num_epochs:
                del trainer_target, target_embeddings, source_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # compute and log scib metrics for source data
            source_scib_metrics = compute_scib_metrics_for_domain(
                rna_adata=source_rna,
                atac_adata=source_atac,
                domain_name="source",
                label_key=args.source_label_key,
                scib_n_jobs=args.scib_n_jobs,
            )
            log_scib_metrics(prefix="source", metrics=source_scib_metrics, step=epoch)
            if not source_scib_label_mode_logged:
                mlflow.log_param("source_scib_label_mode", source_scib_metrics["label_mode"])
                source_scib_label_mode_logged = True
            if not source_scib_effective_label_key_logged:
                mlflow.log_param(
                    "source_scib_effective_label_key",
                    source_scib_metrics["effective_label_key"],
                )
                source_scib_effective_label_key_logged = True

            # compute and log scib metrics for target data
            target_scib_metrics = compute_scib_metrics_for_domain(
                rna_adata=target_rna,
                atac_adata=target_atac,
                domain_name="target",
                label_key=args.target_label_key,
                scib_n_jobs=args.scib_n_jobs,
            )
            log_scib_metrics(prefix="target", metrics=target_scib_metrics, step=epoch)

            if not target_scib_label_mode_logged:
                mlflow.log_param("target_scib_label_mode", target_scib_metrics["label_mode"])
                target_scib_label_mode_logged = True
            if not target_scib_effective_label_key_logged:
                mlflow.log_param(
                    "target_scib_effective_label_key",
                    target_scib_metrics["effective_label_key"],
                )
                target_scib_effective_label_key_logged = True


        # log stage-1 UMAP artifacts
        log_stage_umap_artifacts(
            source_rna=source_rna,
            source_atac=source_atac,
            target_rna=target_rna,
            target_atac=target_atac,
            stage_label="stage1",
            log_mudata_umaps=args.log_mudata_umaps,
        )

        if args.stage1_dual_source_kd:
            
            trainer.mgate.eval()
            teacher_source_embeddings = trainer.infer(
                source_graph_tf,
                source_graph_tf,
                source_gp_tf,
                source_x1,
                source_x2,
            )
            set_multigate_embeddings(
                source_rna,
                source_atac,
                teacher_source_embeddings[0],
                teacher_source_embeddings[1],
                key_added="MultiGATE",
            )
            log_stage_umap_artifacts(
                source_rna=source_rna,
                source_atac=source_atac,
                target_rna=target_rna,
                target_atac=target_atac,
                stage_label="stage1_teacher",
                log_mudata_umaps=args.log_mudata_umaps,
            )

        # log stage-1 model artifacts and attention matrix
        source_peak_gene_attention = source_embeddings[4][0] #peak_gene_attention = source_rna.uns['MultiGATE_gene_peak_attention'][0]
        model_stage1 = stage1_primary_trainer.mgate.state_dict()
        model_stage1_teacher = trainer.mgate.state_dict()
        model_stage1_student = student_trainer.mgate.state_dict() if args.stage1_dual_source_kd else None

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "source_peak_gene_attention.npz")
            sp.save_npz(local_path, source_peak_gene_attention)
            mlflow.log_artifact(local_path, artifact_path="matrices")

            local_path = os.path.join(tmpdir, "model_stage1.pth")
            torch.save(model_stage1, local_path)
            mlflow.log_artifact(local_path, artifact_path="models")

            if args.stage1_dual_source_kd:
                local_path = os.path.join(tmpdir, "model_stage1_teacher.pth")
                torch.save(model_stage1_teacher, local_path)
                mlflow.log_artifact(local_path, artifact_path="models")

                local_path = os.path.join(tmpdir, "model_stage1_student.pth")
                torch.save(model_stage1_student, local_path)
                mlflow.log_artifact(local_path, artifact_path="models")

        #%% stage 2
        if args.stage2_epochs > 0:
            print(
                "[Stage2 KD] Starting target distillation for {} epochs (lambda_kd={})".format(
                    args.stage2_epochs,
                    args.lambda_kd,
                )
            )
            run_stage2_distillation(
                source_trainer=stage1_primary_trainer,
                target_rna=target_rna,
                target_atac=target_atac,
                target_graph_tf=target_graph_tf,
                target_gp_tf=target_gp_tf,
                target_x1=target_x1,
                target_x2=target_x2,
                stage2_epochs=args.stage2_epochs,
                lambda_kd=args.lambda_kd,
                target_label_key=args.target_label_key,
                scib_n_jobs=args.scib_n_jobs,
                vgp_mode=args.vgp_mode,
            )
            # log stage-2 UMAP artifacts
            log_stage_umap_artifacts(
                source_rna=source_rna,
                source_atac=source_atac,
                target_rna=target_rna,
                target_atac=target_atac,
                stage_label="stage2",
                log_mudata_umaps=args.log_mudata_umaps,
            )

            # log stage-2 model artifacts and attention matrix
            target_peak_gene_attention = target_embeddings[4][0] #target_peak_gene_attention = target_rna.uns['MultiGATE_gene_peak_attention'][0]
            model_stage2 = trainer_target.mgate.state_dict()

            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, "target_peak_gene_attention.npz")
                sp.save_npz(local_path, target_peak_gene_attention)
                mlflow.log_artifact(local_path, artifact_path="matrices")

                local_path = os.path.join(tmpdir, "model_stage2.pth")
                torch.save(model_stage2, local_path)
                mlflow.log_artifact(local_path, artifact_path="models")
        else:
            print("[Stage2 KD] Skipped because --stage2-epochs is 0.")

#%%
if __name__ == "__main__":
    main()
else:
    from gene_peak_attention_utils import (
        add_gene_and_peak_columns,
        assign_regulatory_region,
        compute_gene_peak_distance,
        extract_peak_gene_connections,
        filter_by_attention_threshold,
        get_gmm_attention_threshold,
        merge_with_gene_annotations,
        parse_gtf_file,
        plot_attention_distribution,
        plot_distance_distribution,
        save_attention_outputs,
    )

    def run_gene_peak_attention_tutorial(
        peak_gene_attention=None,
        adata_rna=None,
        adata_atac=None,
        gtf_file=None,
        output_dir=None,
        index_layout="gene_first",
        promoter_window_bp=2000,
        min_attention=0.0,
        save_results=True,
        show_plots=True,
        random_state=0,
    ):
        """Run tutorial-style gene-peak attention analysis using reusable utilities."""
        if peak_gene_attention is None:
            peak_gene_attention = globals().get("source_peak_gene_attention")
        if peak_gene_attention is None:
            peak_gene_attention = globals().get("target_peak_gene_attention")
        if peak_gene_attention is None:
            raise ValueError("No attention matrix found. Pass peak_gene_attention explicitly.")

        if adata_rna is None:
            adata_rna = globals().get("source_rna")
        if adata_rna is None:
            adata_rna = globals().get("target_rna")
        if adata_rna is None:
            raise ValueError("No RNA AnnData found. Pass adata_rna explicitly.")

        if adata_atac is None:
            adata_atac = globals().get("source_atac")
        if adata_atac is None:
            adata_atac = globals().get("target_atac")
        if adata_atac is None:
            raise ValueError("No ATAC AnnData found. Pass adata_atac explicitly.")

        if gtf_file is None:
            gtf_file = globals().get("gtf_path")
        if gtf_file is None and os.getenv("DATAPATH"):
            gtf_file = os.path.join(
                os.getenv("DATAPATH"),
                "gene_annotations",
                "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
            )
        if gtf_file is None or not os.path.exists(gtf_file):
            raise FileNotFoundError(f"GTF annotation file not found: {gtf_file}")

        if output_dir is None:
            OUTPATH = os.getenv("OUTPATH")
            output_dir = os.path.join(OUTPATH, "MultiGATE", "attention_analysis")
            os.makedirs(output_dir, exist_ok=True)

        peak_gene_df = extract_peak_gene_connections(
            peak_gene_attention=peak_gene_attention,
            peaks=adata_atac.var_names,
            genes=adata_rna.var_names,
            index_layout=index_layout,
            min_attention=min_attention,
        )
        peak_gene_df = add_gene_and_peak_columns(peak_gene_df)

        gtf_df = parse_gtf_file(gtf_file)
        merged_df = merge_with_gene_annotations(peak_gene_df, gtf_df)
        merged_df = compute_gene_peak_distance(merged_df)
        merged_df = assign_regulatory_region(merged_df, promoter_window_bp=promoter_window_bp)

        gmm_result = get_gmm_attention_threshold(
            merged_df["Attention"].to_numpy(),
            random_state=random_state,
        )
        threshold = gmm_result.threshold
        merged_df_threshold = filter_by_attention_threshold(merged_df, threshold)
        merged_df_threshold = assign_regulatory_region(
            merged_df_threshold,
            promoter_window_bp=promoter_window_bp,
        )

        if show_plots:
            plot_attention_distribution(
                merged_df["Attention"].to_numpy(),
                threshold=threshold,
                gmm=gmm_result.model,
            )
            plot_distance_distribution(merged_df_threshold["distance"].to_numpy())
            plt.show()

        if save_results:
            save_attention_outputs(
                merged_df=merged_df,
                merged_df_threshold=merged_df_threshold,
                threshold=threshold,
                output_dir=output_dir,
            )

        summary = {
            "output_dir": output_dir,
            "pairs_total": int(merged_df.shape[0]),
            "pairs_above_threshold": int(merged_df_threshold.shape[0]),
            "attention_threshold": float(threshold),
        }

        globals()["peak_gene_df"] = peak_gene_df
        globals()["merged_df"] = merged_df
        globals()["merged_df_threshold"] = merged_df_threshold
        globals()["attention_threshold"] = threshold

        print("Gene-peak attention analysis complete:", summary)
        return summary

    _auto_ready = all(
        name in globals()
        for name in ("source_peak_gene_attention", "source_rna", "source_atac")
    )
    if _auto_ready:
        attention_analysis_summary = run_gene_peak_attention_tutorial(
            peak_gene_attention=source_peak_gene_attention,
            adata_rna=source_rna,
            adata_atac=source_atac,
        )
    else:
        print(
            "Imported gene_peak_attention_utils. "
            "Run run_gene_peak_attention_tutorial(...) after training objects are available."
        )
