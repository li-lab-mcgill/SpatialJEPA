#!/usr/bin/env python
"""
multigate_co_embed.py

Given an MLflow run name produced by mouse_brain_spatial_rna_atac.py (e.g.
"20260314_095952"), this script:
  1. Resolves the run name to a run ID via the MLflow tracking DB.
  2. Queries the run for its training hyperparameters.
  3. Downloads one or more trained model artifacts (state dicts).
  4. Reconstructs the MultiGATE model architecture from the state dict.
  5. Reproduces the same feature-selection pipeline as training.
  6. Co-embeds source and target datasets into the shared latent space.
  7. Optionally saves the annotated AnnData objects as h5ad files.

Model artifacts available under artifacts/models/ of a parent run:
  model_stage1.pth          stage-1 primary  (= student if dual-KD, else teacher)
  model_stage1_teacher.pth  stage-1 spatial source teacher    [dual-KD runs only]
  model_stage1_student.pth  stage-1 non-spatial source student [dual-KD runs only]
  model_stage2.pth          stage-2 target distilled student   [stage2_epochs > 0]

Usage:
  python multigate_co_embed.py --run-name <mlflow-run-name> [options]

  # Co-embed using the default (stage1 primary for source, stage2 for target):
  python multigate_co_embed.py --run-name 20260314_095952 --save-h5ad

  # Use zero-shot target transfer instead of stage2:
  python multigate_co_embed.py --run-name 20260314_095952 --target-model zero_shot --vgp-mode zero

  # Use the spatial teacher for source and zero-shot for target:
  python multigate_co_embed.py --run-name 20260314_095952 --source-model stage1_teacher --target-model zero_shot
"""
#%%
import argparse
import json
import os
import shutil
import sys
import tempfile
from pprint import pprint

#%%

from dotenv import dotenv_values, load_dotenv
load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")
print("Loaded environment variables from .env:", end="\n\n")
pprint(dotenv_values("/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"))

if os.getenv("DATAPATH") is None:
    raise EnvironmentError(
        "DATAPATH is not set. Export DATAPATH to the base data directory."
    )

if shutil.which("bedtools") is None:
    raise EnvironmentError(
        "bedtools is required for Cal_gene_peak_Net_new. "
        "Ensure bedtools is installed and on PATH."
    )

base_path = os.path.join(os.getenv("DATAPATH"), "aligned_data")

# Force the local MultiGATE repo, not any site-packages installation.
BAKLAVA_BASE_DIR = os.getenv("BAKLAVA_BASE_DIR")
REPO_ROOT = os.path.join(BAKLAVA_BASE_DIR, "MultiGATE")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Ensure the active conda env's bin directory is on PATH so bedtools is found
# when the script is invoked without an activated shell.
_env_bin = os.path.dirname(sys.executable)
_path_entries = os.environ.get("PATH", "").split(os.pathsep)
if _env_bin and _env_bin not in _path_entries:
    os.environ["PATH"] = _env_bin + os.pathsep + os.environ.get("PATH", "")

import warnings
warnings.filterwarnings("ignore")

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

import MultiGATE
from MultiGATE.MultiGATE import MultiGATE as MultiGATETrainer
from MultiGATE.model_MultiGATE import MGATE

print("Using MultiGATE module:", MultiGATE.__file__)


#%% ─── Argument parsing ─────────────────────────────────────────────────────────

def is_notebook() -> bool:
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

def parse_args(notebook: bool = False):
    p = argparse.ArgumentParser(
        description=(
            "Co-embed source + target using trained MultiGATE artifacts "
            "from an MLflow run."
        )
    )
    p.add_argument(
        "--run-name",
        required=not notebook,
        default=None,
        help=(
            "MLflow run name to load model artifacts from "
            "(e.g. '20260314_095952', as logged by mouse_brain_spatial_rna_atac.py)."
        ),
    )
    p.add_argument(
        "--source-model",
        choices=["stage1", "stage1_teacher", "stage1_student"],
        default="stage1",
        help=(
            "Which source model artifact to use for source inference. "
            "'stage1' uses model_stage1.pth (recommended primary). "
            "'stage1_teacher' uses model_stage1_teacher.pth (dual-KD runs only). "
            "'stage1_student' uses model_stage1_student.pth (dual-KD runs only)."
        ),
    )
    p.add_argument(
        "--target-model",
        choices=["stage2", "zero_shot"],
        default="stage2",
        help=(
            "Which model to use for target inference. "
            "'stage2' loads model_stage2.pth (requires stage2_epochs > 0). "
            "'zero_shot' zero-shot transfers the chosen source model weights."
        ),
    )
    p.add_argument(
        "--vgp-mode",
        choices=["zero", "feature"],
        default="zero",
        help=(
            "How to initialise vgp0/vgp1 when --target-model zero_shot is used. "
            "'zero': zero out GP gating vectors (prior-only attention). "
            "'feature': copy trained source vgp weights directly."
        ),
    )
    p.add_argument(
        "--spatial-graph-type",
        choices=["spatial", "knn", "identity", "tangram"],
        default=None,
        help=(
            "Cell graph type to use for the target. "
            "If omitted, reads 'stage1_student_graph' from the run params. "
            "The source always uses its spatial graph."
        ),
    )
    p.add_argument(
        "--target-subsample-n",
        type=int,
        default=5000,
        help="Maximum number of paired target cells to keep.",
    )
    p.add_argument(
        "--target-subsample-seed",
        type=int,
        default=0,
        help="Random seed for target subsampling.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory in which to write output h5ad files. "
            "Defaults to $OUTPATH/MultiGATE/co_embed/<run-name>/."
        ),
    )
    p.add_argument(
        "--save-h5ad",
        action="store_true",
        default=False,
        help="Save the annotated source and target AnnData objects as h5ad files.",
    )
    if notebook:
        return p.parse_known_args()[0]
    else:
        return p.parse_args()


# ─── MLflow helpers ───────────────────────────────────────────────────────────

def setup_mlflow_tracking():
    """Point MLflow at the MultiGATE tracking DB, mirroring mouse_brain_spatial_rna_atac.py."""
    env_mlflow_base_dir = os.environ.get("MLFLOW_BASE_DIR")
    if env_mlflow_base_dir:
        mlflow_base_dir = os.path.abspath(env_mlflow_base_dir)
        if os.path.basename(mlflow_base_dir.rstrip(os.sep)) != "MultiGATE":
            mlflow_base_dir = os.path.join(mlflow_base_dir, "MultiGATE")
    else:
        mlflow_base_dir = os.path.join(BAKLAVA_BASE_DIR, "mlflow_tracking", "MultiGATE")

    mlflow_db_path = os.path.join(mlflow_base_dir, "mlflow.db")
    tracking_uri = "sqlite:///{}".format(mlflow_db_path)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    print("MLflow tracking URI:", tracking_uri)
    return mlflow_base_dir


def resolve_run_id_from_name(client, run_name, experiment_name="multigate_mouse_brain_live_zeroshot"):
    """
    Look up the run ID for a given run name within the MultiGATE experiment.
    MLflow stores the run name both in run.info.run_name and as the tag
    'mlflow.runName'; we query via attributes.run_name which matches both.
    Raises ValueError if no run or more than one run is found.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            "MLflow experiment '{}' not found. "
            "Ensure the tracking URI points to the correct database.".format(
                experiment_name
            )
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.run_name = '{}'".format(run_name),
    )
    if len(runs) == 0:
        raise ValueError(
            "No run named '{}' found in experiment '{}'.".format(
                run_name, experiment_name
            )
        )
    if len(runs) > 1:
        ids = [r.info.run_id for r in runs]
        raise ValueError(
            "Multiple runs named '{}' found in experiment '{}': {}. "
            "Run names should be unique (they are timestamped by default).".format(
                run_name, experiment_name, ids
            )
        )
    run_id = runs[0].info.run_id
    print("Resolved run name '{}' → run ID: {}".format(run_name, run_id))
    return run_id


def load_run_params(client, run_id, run_name):
    run = client.get_run(run_id)
    params = run.data.params
    print("\nRun params for '{}' (ID: {}):".format(run_name, run_id))
    pprint(params)
    return params


def artifact_exists(client, run_id, artifact_path):
    """Return True if `artifact_path` exists as an artifact of `run_id`."""
    parent_dir = os.path.dirname(artifact_path)
    try:
        listed = client.list_artifacts(run_id, parent_dir)
        return any(a.path == artifact_path for a in listed)
    except Exception:
        return False


def download_model_artifact(client, run_id, artifact_name, dst_dir):
    """
    Download models/<artifact_name> to dst_dir and return the local path.
    Raises FileNotFoundError with a helpful message if the artifact is absent.
    """
    artifact_path = "models/{}".format(artifact_name)
    if not artifact_exists(client, run_id, artifact_path):
        raise FileNotFoundError(
            "Artifact '{}' not found for run {}. "
            "Check that the training run completed the corresponding stage.".format(
                artifact_path, run_id
            )
        )
    print("  Downloading {} ...".format(artifact_path))
    local_path = client.download_artifacts(run_id, artifact_path, dst_dir)
    return local_path


# ─── Model reconstruction ─────────────────────────────────────────────────────

def hidden_dims_from_state_dict(state_dict, w_prefix):
    """
    Reconstruct a hidden_dims list from a ParameterList weight chain stored in
    a state dict.  Keys are expected to follow the pattern '<w_prefix>.0',
    '<w_prefix>.1', etc.

    Example: W1.0 shape (2000, 512) and W1.1 shape (512, 30)
             → hidden_dims = [2000, 512, 30]
    """
    keys = sorted(
        [k for k in state_dict if k.startswith(w_prefix + ".")],
        key=lambda k: int(k.split(".")[1]),
    )
    if not keys:
        raise ValueError("No '{}' weights found in state dict.".format(w_prefix))
    dims = [state_dict[keys[0]].shape[0]]
    for k in keys:
        dims.append(state_dict[k].shape[1])
    return dims


def mgate_from_state_dict(state_dict, device):
    """
    Infer the MGATE architecture entirely from a state dict, instantiate
    the model, load the weights, and return it in eval mode.
    """
    hidden_dims1 = hidden_dims_from_state_dict(state_dict, "W1")
    hidden_dims2 = hidden_dims_from_state_dict(state_dict, "W2")
    temp = float(state_dict.get("logit_scale", torch.tensor(1.0)).item())

    mgate = MGATE(
        hidden_dims1=hidden_dims1,
        hidden_dims2=hidden_dims2,
        spot_num=1,   # vestigial in the PyTorch MGATE; not used in any parameter shape
        temp=temp,
        nonlinear=True,
    ).to(device)
    mgate.load_state_dict(state_dict, strict=False)
    mgate.eval()
    return mgate, hidden_dims1, hidden_dims2


def load_mgate(client, run_id, artifact_name, device, dst_dir):
    """Download a model .pth artifact and return a reconstructed, eval-mode MGATE."""
    local_path = download_model_artifact(client, run_id, artifact_name, dst_dir)
    state_dict = torch.load(local_path, map_location=device, weights_only=False)
    mgate, hidden_dims1, hidden_dims2 = mgate_from_state_dict(state_dict, device)
    print("    hidden_dims1={}, hidden_dims2={}".format(hidden_dims1, hidden_dims2))
    return mgate, hidden_dims1, hidden_dims2


def build_zero_shot_mgate(source_mgate, vgp_mode, device):
    """
    Build a target MGATE by copying all transferable source weights.
    vgp_mode='zero'    : zero out vgp0/vgp1 (prior-only GP attention).
    vgp_mode='feature' : copy trained feature-anchored vgp vectors directly.
    """
    target_mgate = MGATE(
        hidden_dims1=source_mgate.hidden_dims1,
        hidden_dims2=source_mgate.hidden_dims2,
        spot_num=1,
        temp=float(source_mgate.logit_scale.detach().cpu().item()),
        nonlinear=source_mgate.nonlinear,
    ).to(device)

    if vgp_mode == "feature":
        target_mgate.load_state_dict(source_mgate.state_dict(), strict=False)
    else:  # vgp_mode == "zero"
        state_dict = {
            k: v
            for k, v in source_mgate.state_dict().items()
            if k not in {"vgp0", "vgp1"}
        }
        target_mgate.load_state_dict(state_dict, strict=False)
        with torch.no_grad():
            target_mgate.vgp0.zero_()
            target_mgate.vgp1.zero_()

    target_mgate.eval()
    return target_mgate


# ─── Import shared helpers from mouse_brain_spatial_rna_atac.py ───────────────
#
# The training script's `else` branch (executed when imported as a module rather
# than run directly) calls `from gene_peak_attention_utils import ...`, but that
# module does not exist on disk.  We inject a lightweight stub into sys.modules
# *before* the import so Python does not raise an ImportError.  The stub names
# are only used inside run_gene_peak_attention_tutorial(), which we don't call.

import types as _types

_gpa_stub = _types.ModuleType("gene_peak_attention_utils")
for _fn_name in [
    "add_gene_and_peak_columns",
    "assign_regulatory_region",
    "compute_gene_peak_distance",
    "extract_peak_gene_connections",
    "filter_by_attention_threshold",
    "get_gmm_attention_threshold",
    "merge_with_gene_annotations",
    "parse_gtf_file",
    "plot_attention_distribution",
    "plot_distance_distribution",
    "save_attention_outputs",
]:
    setattr(_gpa_stub, _fn_name, None)
sys.modules.setdefault("gene_peak_attention_utils", _gpa_stub)

# Add the scripts directory so Python can find mouse_brain_spatial_rna_atac.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from mouse_brain_spatial_rna_atac import (  # noqa: E402
    build_graph_inputs,
    build_knn_graph_as_spatial_net,
    set_multigate_embeddings,
    pair_and_subsample_target,
)


# ─── Feature selection (mirrors mouse_brain_spatial_rna_atac.py exactly) ──────

def replicate_feature_selection(
    source_rna, source_atac, target_rna, target_atac,
    gp_net, top_n_genes, top_n_peaks, rank_type="fused",
):
    """
    Reproduce the HVG + gene-peak-network filtering from training.
    Returns updated (source_rna, source_atac, target_rna, target_atac, gp_net).
    All returned adatas have uns['gene_peak_Net'] NOT yet attached; that is done
    after HVG slicing in main().
    """
    # Restrict to features covered by the gene-peak network
    gp_genes = gp_net["Gene"].unique()
    gp_peaks = gp_net["Peak"].unique()
    source_rna  = source_rna[:,  source_rna.var_names.isin(gp_genes)].copy()
    source_atac = source_atac[:, source_atac.var_names.isin(gp_peaks)].copy()
    target_rna  = target_rna[:,  target_rna.var_names.isin(gp_genes)].copy()
    target_atac = target_atac[:, target_atac.var_names.isin(gp_peaks)].copy()

    for adata in (source_rna, source_atac, target_rna, target_atac):
        adata.var["highly_variable"] = False

    def _fused_rank(src, tgt):
        if rank_type == "fused":
            return (
                pd.concat(
                    [src.var["dispersions_norm"], tgt.var["dispersions_norm"]],
                    axis=1,
                )
                .mean(axis=1)
                .rank(ascending=True, method="min")
            )
        if rank_type == "source":
            return src.var["dispersions_norm"].rank(ascending=False)
        if rank_type == "target":
            return tgt.var["dispersions_norm"].rank(ascending=False)
        raise ValueError("Unknown rank_type: {}".format(rank_type))

    # Gene filtering
    rna_rank = _fused_rank(source_rna, target_rna)
    gene_mask = rna_rank.le(top_n_genes)
    source_rna.var.loc[gene_mask, "highly_variable"] = True
    target_rna.var.loc[gene_mask, "highly_variable"] = True

    # Peak filtering: only consider peaks in-cis with kept genes
    peak_candidates = gp_net.loc[
        gp_net["Gene"].isin(gene_mask.loc[gene_mask].index), "Peak"
    ].unique()
    atac_rank = _fused_rank(source_atac, target_atac)
    atac_rank_filt = atac_rank.loc[source_atac.var_names.isin(peak_candidates)]
    atac_rank_filt = atac_rank_filt.rank(ascending=True, method="min")
    peak_mask = atac_rank_filt.le(top_n_peaks)
    peak_mask = peak_mask.loc[peak_mask].index
    source_atac.var.loc[source_atac.var_names.isin(peak_mask), "highly_variable"] = True
    target_atac.var.loc[target_atac.var_names.isin(peak_mask), "highly_variable"] = True

    # Restrict gene-peak network to the surviving features
    gp_net_filtered = gp_net[
        gp_net["Gene"].isin(gene_mask.loc[gene_mask].index)
        & gp_net["Peak"].isin(peak_mask)
    ]
    return source_rna, source_atac, target_rna, target_atac, gp_net_filtered


# ─── Target graph construction ────────────────────────────────────────────────

def prepare_target_graphs(
    target_rna, target_atac, source_rna, spatial_graph_type, gtf_path,
):
    """
    Attach uns['Spatial_Net'] and uns['gene_peak_Net'] to the (already
    HVG-filtered) target adatas, matching the graph strategy used in training.
    Returns (target_rna, target_atac).
    """
    if spatial_graph_type == "spatial":
        MultiGATE.Cal_Spatial_Net(target_rna, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Cal_Spatial_Net(target_atac, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_atac)
        # Recompute gene-peak net on already-filtered target features
        MultiGATE.Cal_gene_peak_Net_new(target_rna, target_atac, 150000, file=gtf_path)
        target_rna.uns["gene_peak_Net"] = target_atac.uns["gene_peak_Net"]

    elif spatial_graph_type == "tangram":
        tangram_csv = os.path.join(
            os.getenv("OUTPATH"), "tangram", "tangram_spatial_net_affinity.csv"
        )
        if not os.path.exists(tangram_csv):
            raise FileNotFoundError(
                "Tangram affinity CSV not found: {}".format(tangram_csv)
            )
        tangram_net = pd.read_csv(tangram_csv)
        target_rna.uns["Spatial_Net"] = tangram_net.copy()
        target_atac.uns["Spatial_Net"] = tangram_net.copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()

    elif spatial_graph_type == "knn":
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        build_knn_graph_as_spatial_net(target_rna, n_neighbors=15)
        target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Stats_Spatial_Net(target_atac)

    elif spatial_graph_type == "identity":
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        empty_net = pd.DataFrame(columns=["Cell1", "Cell2", "Distance"])
        target_rna.uns["Spatial_Net"] = empty_net
        target_atac.uns["Spatial_Net"] = empty_net.copy()

    else:
        raise ValueError("Unknown spatial_graph_type: '{}'".format(spatial_graph_type))

    return target_rna, target_atac


# ─── Inference ────────────────────────────────────────────────────────────────

def _as_sparse_tensor(graph_tf, device):
    """Convert a prepare_graph_data() tuple to a torch sparse tensor."""
    indices, values, shape = graph_tf
    indices = np.asarray(indices)
    if indices.ndim == 2 and indices.shape[1] == 2:
        indices = indices.T
    indices_t = torch.as_tensor(indices, dtype=torch.long, device=device)
    values_t  = torch.as_tensor(values,  dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices_t, values_t, torch.Size(shape), device=device
    ).coalesce()


def run_inference(mgate, graph_tf, gp_tf, x1_df, x2_df, device):
    """
    Run a forward pass through mgate (eval mode, no grad) and return
    (rna_embedding, atac_embedding) as NumPy arrays.
    These are the L2-normalised CLIP projections H1/H2 (outputs[5] and [6]).
    """
    x1_t = torch.as_tensor(
        x1_df.values if hasattr(x1_df, "values") else np.asarray(x1_df),
        dtype=torch.float32, device=device,
    )
    x2_t = torch.as_tensor(
        x2_df.values if hasattr(x2_df, "values") else np.asarray(x2_df),
        dtype=torch.float32, device=device,
    )
    a_t  = _as_sparse_tensor(graph_tf, device)
    gp_t = _as_sparse_tensor(gp_tf,    device)

    mgate.eval()
    with torch.no_grad():
        outputs = mgate(a_t, a_t, gp_t, x1_t, x2_t)

    rna_emb  = outputs[5].detach().cpu().numpy()
    atac_emb = outputs[6].detach().cpu().numpy()
    return rna_emb, atac_emb


#%% ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    #%%
    NOTEBOOK = is_notebook()
    args = parse_args(notebook=NOTEBOOK)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #%% ── MLflow setup ────────────────────────────────────────────────────────
    setup_mlflow_tracking()
    client = MlflowClient()
    run_id = resolve_run_id_from_name(client, args.run_name)
    run_params = load_run_params(client, run_id, args.run_name)

    #%% Resolve graph type for the target (arg takes precedence over run param)
    spatial_graph_type = (
        args.spatial_graph_type
        or run_params.get("stage1_student_graph", "identity")
    )
    print("Target spatial graph type:", spatial_graph_type)

    bp_width       = int(run_params.get("bp_width", 400))
    graph_type     = run_params.get("graph_type", "ATAC")
    dual_source_kd = run_params.get("stage1_dual_source_kd", "False").lower() == "true"
    has_stage2     = int(run_params.get("stage2_epochs", 0)) > 0

    # Validate model selections against available artifacts
    if args.source_model in ("stage1_teacher", "stage1_student") and not dual_source_kd:
        raise ValueError(
            "--source-model {} requires the training run to have used "
            "--stage1-dual-source-kd, but 'stage1_dual_source_kd={}' in run params.".format(
                args.source_model, run_params.get("stage1_dual_source_kd")
            )
        )
    if args.target_model == "stage2" and not has_stage2:
        raise ValueError(
            "--target-model stage2 requires stage2_epochs > 0 in the training run, "
            "but 'stage2_epochs={}'. Use --target-model zero_shot instead.".format(
                run_params.get("stage2_epochs", 0)
            )
        )

    # ── Output directory ────────────────────────────────────────────────────
    if args.output_dir is None:
        outpath = os.getenv("OUTPATH", "/tmp")
        output_dir = os.path.join(outpath, "MultiGATE", "co_embed", args.run_name)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    # ── GTF path ────────────────────────────────────────────────────────────
    gtf_path = os.path.join(
        os.getenv("DATAPATH"), "gene_annotations",
        "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    )
    if not os.path.exists(gtf_path):
        raise FileNotFoundError("GTF annotation file not found: {}".format(gtf_path))

    #%% ── Load raw aligned datasets ────────────────────────────────────────────
    print("\nLoading source and target datasets...")
    source_rna  = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned.h5ad"))
    source_atac = sc.read_h5ad(os.path.join(base_path, "source_atac_aligned.h5ad"))
    target_rna  = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned.h5ad"))
    target_atac = sc.read_h5ad(os.path.join(base_path, "target_atac_aligned.h5ad"))

    # Flip spatial coordinates (same convention as training)
    source_rna.obsm["spatial"]  = source_rna.obsm["spatial"][:,  [1, 0]] * -1
    source_atac.obsm["spatial"] = source_atac.obsm["spatial"][:, [1, 0]] * -1

    # ── Gene-peak network ────────────────────────────────────────────────────
    print("\nComputing gene-peak network on source...")
    MultiGATE.Cal_gene_peak_Net_new(source_rna, source_atac, 150000, file=gtf_path)
    gp_net = source_atac.uns["gene_peak_Net"].copy()
    del source_atac.uns["gene_peak_Net"]

    # ── Feature selection ────────────────────────────────────────────────────
    print(
        "Replicating feature selection "
        "(top_n_genes={}, top_n_peaks={})...".format(run_params.get("n_genes"), run_params.get("n_peaks"))
    )
    source_rna, source_atac, target_rna, target_atac, gp_net = replicate_feature_selection(
        source_rna, source_atac, target_rna, target_atac,
        gp_net,
        top_n_genes=int(run_params.get("n_genes")),
        top_n_peaks=int(run_params.get("n_peaks")),
    )

    # Attach gene-peak net before HVG slicing (needed by build_graph_inputs)
    source_rna.uns["gene_peak_Net"] = gp_net.copy()

    # Apply HVG filter
    source_rna  = source_rna[:,  source_rna.var["highly_variable"]].copy()
    source_atac = source_atac[:, source_atac.var["highly_variable"]].copy()
    target_rna  = target_rna[:,  target_rna.var["highly_variable"]].copy()
    target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()

    # Re-attach gene-peak net after slicing
    source_rna.uns["gene_peak_Net"] = gp_net.copy()

    n_genes = source_rna.n_vars
    n_peaks = source_atac.n_vars
    print("After feature selection: {} genes, {} peaks".format(n_genes, n_peaks))

    #%% ── Download model artifacts ─────────────────────────────────────────────
    source_artifact_map = {
        "stage1":         "model_stage1.pth",
        "stage1_teacher": "model_stage1_teacher.pth",
        "stage1_student": "model_stage1_student.pth",
    }
    source_artifact_name = source_artifact_map[args.source_model]

    print(
        "\nDownloading model artifacts for run '{}' (ID: {})...".format(
            args.run_name, run_id
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:

        print("Source model ({}) :".format(source_artifact_name))
        source_mgate, hidden_dims1, hidden_dims2 = load_mgate(
            client, run_id, source_artifact_name, device, tmpdir
        )

        # Verify feature dimensions match after our feature-selection replication
        expected_n_genes = hidden_dims1[0]
        expected_n_peaks = hidden_dims2[0]
        if n_genes != expected_n_genes or n_peaks != expected_n_peaks:
            raise ValueError(
                "Feature dimension mismatch — the loaded model expects "
                "({} genes, {} peaks) but our feature selection produced "
                "({} genes, {} peaks).\n"
                "Adjust --top-n-genes / --top-n-peaks to match what was "
                "used during training.".format(
                    expected_n_genes, expected_n_peaks, n_genes, n_peaks
                )
            )
        print("  Feature dimensions verified: {} genes, {} peaks".format(n_genes, n_peaks))

        if args.target_model == "stage2":
            print("Target model (model_stage2.pth):")
            target_mgate, _, _ = load_mgate(
                client, run_id, "model_stage2.pth", device, tmpdir
            )
        else:
            # Zero-shot: built after download, outside the tmpdir block
            target_mgate = None

    # Build zero-shot target model outside the tmpdir context (weights already loaded)
    if args.target_model == "zero_shot":
        print(
            "\nBuilding zero-shot target model from source weights "
            "(vgp_mode='{}')...".format(args.vgp_mode)
        )
        target_mgate = build_zero_shot_mgate(source_mgate, args.vgp_mode, device)

    #%% ── Source spatial graph ─────────────────────────────────────────────────
    print("\nBuilding source spatial graph...")
    MultiGATE.Cal_Spatial_Net(source_rna, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_rna)
    MultiGATE.Cal_Spatial_Net(source_atac, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_atac)
    # source_rna already has gene_peak_Net from above
    source_atac.uns["gene_peak_Net"] = gp_net.copy()

    source_graph_tf, source_gp_tf, source_x1, source_x2 = build_graph_inputs(
        source_rna, source_atac, bp_width=bp_width, graph_type=graph_type
    )
    print("  Source graph built: {} cells".format(source_rna.n_obs))

    # ── Target graphs ────────────────────────────────────────────────────────
    print("\nPairing and subsampling target cells...")
    target_rna, target_atac = pair_and_subsample_target(
        target_rna, target_atac,
        subsample_n=args.target_subsample_n,
        seed=args.target_subsample_seed,
    )
    print("  {} cells after pairing/subsampling".format(target_rna.n_obs))

    print("Building target graph (type: '{}')...".format(spatial_graph_type))
    target_rna, target_atac = prepare_target_graphs(
        target_rna, target_atac, source_rna, spatial_graph_type, gtf_path
    )
    target_graph_tf, target_gp_tf, target_x1, target_x2 = build_graph_inputs(
        target_rna, target_atac, bp_width=bp_width, graph_type=graph_type
    )
    print("  Target graph built: {} cells".format(target_rna.n_obs))

    #%% ── Inference ────────────────────────────────────────────────────────────
    print("\nRunning source inference ({})...".format(source_artifact_name))
    source_rna_emb, source_atac_emb = run_inference(
        source_mgate, source_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, source_rna_emb, source_atac_emb)
    print("  Source embeddings: shape {}".format(source_rna_emb.shape))

    target_model_label = (
        "model_stage2.pth" if args.target_model == "stage2"
        else "zero_shot_from_{}".format(source_artifact_name)
    )
    print("Running target inference ({})...".format(target_model_label))
    target_rna_emb, target_atac_emb = run_inference(
        target_mgate, target_graph_tf, target_gp_tf, target_x1, target_x2, device
    )
    set_multigate_embeddings(target_rna, target_atac, target_rna_emb, target_atac_emb)
    print("  Target embeddings: shape {}".format(target_rna_emb.shape))

    #%% Plot source and target embeddings
    multigate_adata = sc.AnnData(
        X=np.concatenate([source_rna_emb, source_atac_emb, target_rna_emb, target_atac_emb], axis=0),
        obs=pd.concat([
            source_rna.obs.assign(modality="rna", source_or_target="source"),
            source_atac.obs.assign(modality="atac", source_or_target="source"),
            target_rna.obs.assign(modality="rna", source_or_target="target"),
            target_atac.obs.assign(modality="atac", source_or_target="target"),
            ], axis=0),
    )
    sc.pp.neighbors(multigate_adata, use_rep='X', n_neighbors=100)
    sc.tl.leiden(multigate_adata, resolution=0.5)
    sc.tl.umap(multigate_adata, min_dist=0.3)
    sc.pl.umap(multigate_adata, color=['modality', 'source_or_target', 'leiden'], ncols=3, wspace=0.1, size=25)

    #%% ── Save outputs ─────────────────────────────────────────────────────────
    if args.save_h5ad:
        print("\nSaving h5ad files to {}...".format(output_dir))
        paths = {
            "source_rna":  (source_rna,  "source_rna_co_embed.h5ad"),
            "source_atac": (source_atac, "source_atac_co_embed.h5ad"),
            "target_rna":  (target_rna,  "target_rna_co_embed.h5ad"),
            "target_atac": (target_atac, "target_atac_co_embed.h5ad"),
        }
        for label, (adata, fname) in paths.items():
            out_path = os.path.join(output_dir, fname)
            adata.write_h5ad(out_path)
            print("  {} ({} cells): {}".format(label, adata.n_obs, out_path))

    print(
        "\nDone. Embeddings are stored in obsm['MultiGATE'] and "
        "obsm['MultiGATE_clip_all'] for each AnnData."
    )
    return source_rna, source_atac, target_rna, target_atac

#%%
if __name__ == "__main__":
    main()
