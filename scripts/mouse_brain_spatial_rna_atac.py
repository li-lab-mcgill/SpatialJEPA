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
import mlflow
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from dotenv import dotenv_values, load_dotenv
from sklearn.preprocessing import Normalizer

import MultiGATE
from MultiGATE.MultiGATE import MultiGATE as MultiGATETrainer

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
        "--stage2-epochs",
        type=int,
        default=100,
        help="Number of teacher-student distillation epochs on target data (stage 2).",
    )
    parser.add_argument(
        "--lambda-kd",
        type=float,
        default=1.0,
        help="Scale factor for stage-2 KD objective.",
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


def build_graph_inputs(adata1, adata2, bp_width=450, graph_type="ATAC", protein_value=0.001):
    if "highly_variable" in adata1.var.columns and "highly_variable" in adata2.var.columns:
        adata_vars1 = adata1[:, adata1.var["highly_variable"]]
        adata_vars2 = adata2[:, adata2.var["highly_variable"]]
    else:
        adata_vars1 = adata1
        adata_vars2 = adata2

    x1 = _to_dense_df(adata_vars1)
    x2 = _to_dense_df(adata_vars2)

    cells = np.array(x1.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))

    genes = np.array(x1.columns)
    peaks = np.array(x2.columns)
    genes_id_tran = dict(zip(genes, range(genes.shape[0])))
    peaks_id_tran = dict(zip(peaks, range(peaks.shape[0])))

    if "Spatial_Net" not in adata1.uns:
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

    if "gene_peak_Net" not in adata1.uns:
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
        config={"device": str(source_trainer.device)},
    )

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


def compute_morans_i_mean(adata, rep_key="MultiGATE", neighbors_key="eval_graph", n_neighbors=15):
    sc.pp.neighbors(adata, use_rep=rep_key, n_neighbors=n_neighbors, key_added=neighbors_key)
    conn_key = neighbors_key + "_connectivities"
    values = sc.metrics.morans_i(adata.obsp[conn_key].tocsr(), adata.obsm[rep_key].T)
    return float(np.nanmean(values))


def log_umap_to_mlflow(adata, artifact_path, title, color_key="wnn", size=20):
    if adata.n_obs < 3:
        warnings.warn("Skipping UMAP artifact '{}' because n_obs < 3.".format(artifact_path))
        return
    if "X_umap" not in adata.obsm:
        warnings.warn("Skipping UMAP artifact '{}' because X_umap is missing.".format(artifact_path))
        return

    umap_fig = None
    try:
        umap_fig, umap_ax = plt.subplots(figsize=(7, 5))
        sc.pl.umap(adata, color=color_key, title=title, ax=umap_ax, size=size, show=False)
        umap_fig.tight_layout()
        mlflow.log_figure(umap_fig, artifact_path)
    finally:
        if umap_fig is not None:
            plt.close(umap_fig)


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
    )
    kd_rows = F.kl_div(
        F.log_softmax(student_logits, dim=0),
        F.log_softmax(teacher_logits, dim=0),
        reduction="batchmean",
        log_target=True,
    )
    return 0.5 * (kd_cols + kd_rows)


def compute_ot_clip_loss(student_logits, teacher_logits, emd):
    one = torch.tensor(1.0, device=teacher_logits.device, dtype=teacher_logits.dtype)
    teacher_cost = 1 - (teacher_logits / torch.exp(1 / one))

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
):
    if stage2_epochs <= 0:
        return None

    emd = require_ot_backend()

    teacher_trainer = build_zero_shot_target_trainer(source_trainer, target_rna.n_obs)
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
        weight_decay=source_trainer.mgate.weight_decay,
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

        for epoch in range(1, stage2_epochs + 1):
            student_trainer.mgate.train()
            student_trainer.optimizer.zero_grad()

            student_outputs = student_trainer.mgate(target_a_t, target_prune_t, target_gp_t, target_x1_t, target_x2_t)
            with torch.no_grad():
                teacher_outputs = teacher_model(target_a_t, target_prune_t, target_gp_t, target_x1_t, target_x2_t)

            student_clip_rna, student_clip_atac = student_outputs[5], student_outputs[6]
            teacher_clip_rna, teacher_clip_atac = teacher_outputs[5], teacher_outputs[6]

            student_logits = compute_clip_logits(student_clip_rna, student_clip_atac, student_trainer.mgate.logit_scale)
            teacher_logits = compute_clip_logits(teacher_clip_rna, teacher_clip_atac, teacher_model.logit_scale)

            kd_kl_loss = compute_kd_kl_loss(student_logits, teacher_logits)
            kd_ot_loss = compute_ot_clip_loss(student_logits, teacher_logits, emd=emd)
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

            set_multigate_embeddings(
                target_rna,
                target_atac,
                student_clip_rna.detach().cpu().numpy(),
                student_clip_atac.detach().cpu().numpy(),
                key_added="MultiGATE",
            )

            try:
                target_morans = compute_morans_i_mean(
                    target_rna,
                    rep_key="MultiGATE",
                    neighbors_key="target_stage2_eval",
                    n_neighbors=15,
                )
                mlflow.log_metric("stage2_target_morans_i_mean", target_morans, step=epoch)
                mlflow.log_metric("stage2_target_modularity_placeholder", target_morans, step=epoch)
            except Exception as exc:
                warnings.warn("Stage-2 target metric computation failed at epoch {}: {}".format(epoch, exc))

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

    NOTEBOOK = is_notebook()
    args = parse_args(notebook=NOTEBOOK)
    if args.target_subsample_n <= 0:
        raise ValueError("--target-subsample-n must be a positive integer.")
    if args.stage2_epochs < 0:
        raise ValueError("--stage2-epochs must be a non-negative integer.")
    if args.lambda_kd < 0:
        raise ValueError("--lambda-kd must be non-negative.")

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
    if socket.gethostname() != "ri-muhc-gpu":
        source_rna.var["highly_variable"] = False
        source_atac.var["highly_variable"] = False

        target_rna.var["highly_variable"] = False
        target_atac.var["highly_variable"] = False

        top_n_genes = 2000
        top_n_peaks = 10000
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
    target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
    target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()

    target_rna, target_atac = pair_and_subsample_target(
        target_rna,
        target_atac,
        subsample_n=args.target_subsample_n,
        seed=args.target_subsample_seed,
    )

    target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
    target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]

    build_knn_graph_as_spatial_net(target_rna, n_neighbors=15)
    target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
    MultiGATE.Stats_Spatial_Net(target_rna)
    MultiGATE.Stats_Spatial_Net(target_atac)

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
    num_epochs = int(os.getenv("MULTIGATE_EPOCHS", "3000"))
    if num_epochs <= 0:
        raise ValueError("MULTIGATE_EPOCHS must be a positive integer.")

    hidden_dims = [512, 30]
    trainer = MultiGATETrainer(
        hidden_dims1=[source_x1.shape[1]] + hidden_dims,
        hidden_dims2=[source_x2.shape[1]] + hidden_dims,
        spot_num=source_x1.shape[0],
        temp=-10,
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

    #%% MLflow setup
    experiment_id = setup_mlflow()
    eval_every = 100
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Training epochs:", num_epochs)
    print("Target paired cells after subsampling:", target_rna.n_obs)

    parent_run_id = None
    with mlflow.start_run(run_name=run_name):
        parent_run_id = mlflow.active_run().info.run_id
        mlflow.log_param("mlflow_experiment_id", experiment_id)
        mlflow.log_param("n_epochs", num_epochs)
        mlflow.log_param("stage2_epochs", args.stage2_epochs)
        mlflow.log_param("lambda_kd", args.lambda_kd)
        mlflow.log_param("bp_width", bp_width)
        mlflow.log_param("target_subsample_n", args.target_subsample_n)
        mlflow.log_param("target_subsample_seed", args.target_subsample_seed)
        mlflow.log_param("target_effective_n", int(target_rna.n_obs))
        mlflow.log_param("eval_every", eval_every)
        mlflow.log_param("source_cells", int(source_rna.n_obs))
        mlflow.log_param("target_cells", int(target_rna.n_obs))
        mlflow.log_param("graph_type", graph_type)

        for epoch in range(1, num_epochs + 1):
            loss = trainer.run_epoch(epoch, source_a_t, source_prune_t, source_gp_t, source_x1_t, source_x2_t)
            mlflow.log_metric("source_train_loss", float(loss), step=epoch)

            should_eval = (epoch % eval_every == 0) or (epoch == num_epochs)
            if not should_eval:
                continue

            print("[Live eval] Epoch {}/{}".format(epoch, num_epochs))

            source_embeddings = trainer.infer(
                source_graph_tf,
                source_graph_tf,
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

            trainer_target = build_zero_shot_target_trainer(trainer, target_rna.n_obs)
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

            try:
                source_morans = compute_morans_i_mean(
                    source_rna,
                    rep_key="MultiGATE",
                    neighbors_key="source_eval",
                    n_neighbors=15,
                )
                mlflow.log_metric("source_morans_i_mean", source_morans, step=epoch)
                mlflow.log_metric("source_modularity_placeholder", source_morans, step=epoch)
            except Exception as exc:
                warnings.warn("Source metric computation failed at epoch {}: {}".format(epoch, exc))

            try:
                target_morans = compute_morans_i_mean(
                    target_rna,
                    rep_key="MultiGATE",
                    neighbors_key="target_eval",
                    n_neighbors=15,
                )
                mlflow.log_metric("target_morans_i_mean", target_morans, step=epoch)
                mlflow.log_metric("target_modularity_placeholder", target_morans, step=epoch)
            except Exception as exc:
                warnings.warn("Target metric computation failed at epoch {}: {}".format(epoch, exc))

        if args.stage2_epochs > 0:
            print(
                "[Stage2 KD] Starting target distillation for {} epochs (lambda_kd={})".format(
                    args.stage2_epochs,
                    args.lambda_kd,
                )
            )
            run_stage2_distillation(
                source_trainer=trainer,
                target_rna=target_rna,
                target_atac=target_atac,
                target_graph_tf=target_graph_tf,
                target_gp_tf=target_gp_tf,
                target_x1=target_x1,
                target_x2=target_x2,
                stage2_epochs=args.stage2_epochs,
                lambda_kd=args.lambda_kd,
            )
        else:
            print("[Stage2 KD] Skipped because --stage2-epochs is 0.")

    #%% clustering with Muon's WNN clustering (source)
    sc.pp.neighbors(source_rna)
    sc.pp.neighbors(source_atac)

    mdata = mu.MuData({"rna": source_rna, "atac": source_atac})
    mu.pp.neighbors(mdata)

    mu.tl.umap(mdata)
    sc.tl.leiden(mdata, resolution=1.5)

    # Replicate outputs of wnn_R: propagate cluster labels and UMAP coordinates
    # back to the individual AnnData objects so downstream code is unaffected.
    for ad in [source_rna, source_atac]:
        ad.obs["wnn"] = mdata.obs["leiden"].astype(int).astype("category")
        ad.obsm["X_umap"] = mdata.obsm["X_umap"]

    # visualize source results
    plt.rcParams["figure.figsize"] = (7, 3)
    fig, axs = plt.subplots(1, 2)
    sc.pl.embedding(source_rna, basis="spatial", color="wnn", s=20, show=False, title="MultiGATE Spatial", ax=axs[0], legend_loc="None")
    sc.pl.umap(source_rna, color="wnn", title="MultiGATE UMAP", ax=axs[1], size=20)
    plt.tight_layout()
    plt.show()

    print("Target forward pass complete. Embedding shape:", target_rna.obsm["MultiGATE"].shape)

    #%% clustering with Muon's WNN clustering (target)
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
        ad.obs["wnn"] = mdata.obs["leiden"].astype(int).astype("category")
        ad.obsm["X_umap"] = mdata.obsm["X_umap"]

    # visualize target results
    plt.rcParams["figure.figsize"] = (7, 3)
    fig, axs = plt.subplots(1, 1)
    sc.pl.umap(target_rna, color="wnn", title="MultiGATE UMAP", ax=axs, size=20)
    plt.tight_layout()
    plt.show()

    if parent_run_id is not None:
        try:
            with mlflow.start_run(run_id=parent_run_id):
                log_umap_to_mlflow(
                    source_rna,
                    artifact_path="umap/source_data_umap_final.png",
                    title="Source Data UMAP",
                    color_key="wnn",
                    size=20,
                )
                log_umap_to_mlflow(
                    target_rna,
                    artifact_path="umap/target_data_umap_final.png",
                    title="Target Data UMAP",
                    color_key="wnn",
                    size=20,
                )
        except Exception as exc:
            warnings.warn("Failed to generate/log source/target UMAP artifacts to MLflow: {}".format(exc))

#%%
if __name__ == "__main__":
    main()
