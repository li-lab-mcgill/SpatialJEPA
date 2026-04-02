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

Model artifacts available under artifacts/models/:
  Parent run:
    model_stage1.pth          stage-1 primary  (= student if dual-KD, else teacher)
    model_stage1_teacher.pth  stage-1 spatial source teacher    [dual-KD runs only]
    model_stage1_student.pth  stage-1 non-spatial source student [dual-KD runs only]
    model_stage2.pth          legacy stage-2 target distilled student fallback
  Stage-2 child run(s):
    model_stage2.pth          stage-2 target distilled student

Usage:
  python multigate_co_embed.py --run-name <mlflow-run-name> [options]

  # Co-embed using the default (stage1 primary for source, stage2 for target):
  # Requires exactly one stage2 child run under --run-name, otherwise pass
  # --stage2-run-name explicitly (or rely on legacy parent fallback when no child exists).
  python multigate_co_embed.py --run-name 20260314_095952 --save-h5ad

  # Use a specific stage2 child run name when multiple stage2 children exist:
  python multigate_co_embed.py --run-name 20260314_095952 --target-model stage2 --stage2-run-name 20260314_095952_stage2_20260323_184501

  # Use zero-shot target transfer instead of stage2:
  python multigate_co_embed.py --run-name 20260314_095952 --target-model zero_shot

  # Use the spatial teacher for source and zero-shot for target:
  python multigate_co_embed.py --run-name 20260314_095952 --source-model stage1_teacher --target-model zero_shot
"""
#%%
import argparse
import os
import random
import shutil
import sys
import tempfile
from pprint import pprint
import matplotlib.pyplot as plt

# Determinism knobs that should be set before importing numpy/torch/scanpy.
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

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
import scanpy as sc
import torch

import MultiGATE
from MultiGATE.model_MultiGATE import MGATE

print("Using MultiGATE module:", MultiGATE.__file__)

from mouse_brain_spatial_rna_atac import (  # noqa: E402
    build_graph_inputs,
    build_source_student_graph_tf,
    set_multigate_embeddings,
    pair_and_subsample_target,
    apply_hvg_and_gp_filtering,
    prepare_target_for_spatial_graph_type,
    build_concat_adata_for_umap,
    compute_concat_umap,
)


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
            "'stage2' loads model_stage2.pth from a resolved stage2 child run "
            "(or legacy parent fallback when no child run exists). "
            "'zero_shot' zero-shot transfers the chosen source model weights."
        ),
    )
    p.add_argument(
        "--stage2-run-name",
        default=None,
        help=(
            "Optional stage2 child run name under --run-name parent. "
            "Valid only with --target-model stage2. If omitted, exactly one stage2 "
            "child run must exist; when zero child runs exist, falls back to "
            "legacy parent model_stage2.pth."
        ),
    )
    p.add_argument(
        "--vgp-anchor-mode",
        choices=["spot", "feature"],
        default=None,
        help=(
            "Override the vgp anchoring mode for zero-shot target construction. "
            "If omitted, uses run param 'vgp_anchor_mode' when available, else "
            "infers from the source checkpoint."
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

def setup_mlflow_tracking(backend_type="auto"):

    valid_backends = ["auto", "sqlite", "postgres"]
    if backend_type not in valid_backends:
        raise ValueError(
            "Invalid MLflow backend type '{}'. Expected one of {}.".format(
                backend_type, valid_backends
            )
        )

    # Keep MultiGATE artifacts/runs namespaced under a dedicated subdirectory.
    env_mlflow_base_dir = os.environ.get("MLFLOW_BASE_DIR")
    if env_mlflow_base_dir:
        mlflow_base_dir = os.path.abspath(env_mlflow_base_dir)
        if os.path.basename(mlflow_base_dir.rstrip(os.sep)) != "MultiGATE":
            mlflow_base_dir = os.path.join(mlflow_base_dir, "MultiGATE")
    else:
        mlflow_base_dir = os.path.join(BAKLAVA_BASE_DIR, "mlflow_tracking", "MultiGATE")

    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if backend_type == "auto":
        # If a non-SQLite tracking URI is present (e.g. http://127.0.0.1:5000),
        # prefer it so this script can read runs logged through an MLflow server
        # backed by PostgreSQL.
        if env_tracking_uri and not env_tracking_uri.startswith("sqlite:///"):
            backend_type = "postgres"
        else:
            backend_type = "sqlite"

    if backend_type == "postgres":
        tracking_uri = env_tracking_uri or "http://127.0.0.1:5000"
    else:
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


def _format_stage2_run_choices(run_infos):
    if not run_infos:
        return "None"
    return ", ".join(
        "{} ({})".format(info["run_name"], info["run_id"])
        for info in run_infos
    )


def list_stage2_child_runs(client, experiment_id, parent_run_id):
    """Return stage2 child runs sorted by most recent start time."""
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.parentRunId = '{}'".format(parent_run_id),
        order_by=["attributes.start_time DESC"],
    )

    child_infos = []
    for run in child_runs:
        run_name = run.data.tags.get("mlflow.runName") or run.info.run_id
        child_infos.append(
            {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "training_stage": run.data.tags.get("training_stage"),
            }
        )

    stage2_tagged = [
        info for info in child_infos
        if info["training_stage"] == "stage2_distillation"
    ]
    if stage2_tagged:
        return stage2_tagged
    return child_infos


STAGE2_MODEL_ARTIFACT = "models/model_stage2.pth"


def resolve_stage2_model_run(client, experiment_id, parent_run_id, stage2_run_name=None):
    """
    Resolve which run ID should provide model_stage2.pth.

    Returns: (resolved_run_id, resolved_run_name, source_kind)
      source_kind is one of {"child", "parent_legacy"}.

    If nested child runs exist but a child's checkpoint was never logged (crashed
    job, older code path), the parent may still hold a legacy model_stage2.pth.
    We only bind to a child run when that run actually lists the artifact.
    """
    stage2_children = list_stage2_child_runs(client, experiment_id, parent_run_id)
    parent_has_legacy = artifact_exists(client, parent_run_id, STAGE2_MODEL_ARTIFACT)

    if stage2_run_name is not None:
        matches = [r for r in stage2_children if r["run_name"] == stage2_run_name]
        if len(matches) == 0:
            raise ValueError(
                "No stage2 child run named '{}' found under parent run {}. "
                "Available stage2 child runs: {}".format(
                    stage2_run_name,
                    parent_run_id,
                    _format_stage2_run_choices(stage2_children),
                )
            )
        if len(matches) > 1:
            raise ValueError(
                "Multiple stage2 child runs named '{}' found under parent run {}: {}. "
                "Use unique child run names.".format(
                    stage2_run_name,
                    parent_run_id,
                    _format_stage2_run_choices(matches),
                )
            )
        selected = matches[0]
        if artifact_exists(client, selected["run_id"], STAGE2_MODEL_ARTIFACT):
            return selected["run_id"], selected["run_name"], "child"
        # Checkpoint may have been logged on the parent (legacy) even though the UI
        # shows a nested stage-2 run row; nested runs do not always receive files.
        if parent_has_legacy:
            return parent_run_id, selected["run_name"], "parent_legacy"
        raise FileNotFoundError(
            "Stage-2 child run '{}' (ID: {}) has no artifact '{}', and parent run {} "
            "has none either. Confirm the Run ID matches and that stage-2 logging finished.".format(
                stage2_run_name,
                selected["run_id"],
                STAGE2_MODEL_ARTIFACT,
                parent_run_id,
            )
        )

    if len(stage2_children) == 1:
        selected = stage2_children[0]
        if artifact_exists(client, selected["run_id"], STAGE2_MODEL_ARTIFACT):
            return selected["run_id"], selected["run_name"], "child"
        if parent_has_legacy:
            return parent_run_id, None, "parent_legacy"
        raise FileNotFoundError(
            "Child run '{}' (ID: {}) exists under parent but has no '{}'; "
            "parent {} also has no legacy checkpoint. "
            "Re-run stage-2 or remove the empty nested run.".format(
                selected["run_name"],
                selected["run_id"],
                STAGE2_MODEL_ARTIFACT,
                parent_run_id,
            )
        )

    if len(stage2_children) > 1:
        raise ValueError(
            "Found multiple stage2 child runs under parent run {}: {}. "
            "Pass --stage2-run-name to select one.".format(
                parent_run_id,
                _format_stage2_run_choices(stage2_children),
            )
        )

    if parent_has_legacy:
        return parent_run_id, None, "parent_legacy"

    raise ValueError(
        "No stage2 child runs found under parent run {} and no legacy "
        "parent artifact {} exists.".format(parent_run_id, STAGE2_MODEL_ARTIFACT)
    )


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
    feat_num = hidden_dims1[0] + hidden_dims2[0]
    if "vgp0" not in state_dict:
        raise KeyError("State dict missing required key 'vgp0'.")
    vgp_len = int(state_dict["vgp0"].shape[0])
    if vgp_len == feat_num:
        vgp_anchor_mode = "feature"
        inferred_spot_num = 1
    else:
        vgp_anchor_mode = "spot"
        inferred_spot_num = vgp_len

    mgate = MGATE(
        hidden_dims1=hidden_dims1,
        hidden_dims2=hidden_dims2,
        spot_num=inferred_spot_num,
        temp=temp,
        nonlinear=True,
        vgp_anchor_mode=vgp_anchor_mode,
    ).to(device)
    mgate.load_state_dict(state_dict, strict=False)
    mgate.eval()
    return mgate, hidden_dims1, hidden_dims2, vgp_anchor_mode


def load_mgate(client, run_id, artifact_name, device, dst_dir):
    """Download a model .pth artifact and return a reconstructed, eval-mode MGATE."""
    local_path = download_model_artifact(client, run_id, artifact_name, dst_dir)
    state_dict = torch.load(local_path, map_location=device, weights_only=False)
    mgate, hidden_dims1, hidden_dims2, vgp_anchor_mode = mgate_from_state_dict(state_dict, device)
    print(
        "    hidden_dims1={}, hidden_dims2={}, vgp_anchor_mode={}".format(
            hidden_dims1, hidden_dims2, vgp_anchor_mode
        )
    )
    return mgate, hidden_dims1, hidden_dims2, vgp_anchor_mode


def build_zero_shot_mgate(source_mgate, target_spot_num, vgp_anchor_mode, device):
    """
    Build a target MGATE by copying transferable source weights.
    If vgp_anchor_mode == 'feature': copy trained vgp vectors directly.
    If vgp_anchor_mode == 'spot': zero out vgp0/vgp1 (prior-only GP attention).
    """
    if vgp_anchor_mode is None:
        vgp_anchor_mode = getattr(source_mgate, "vgp_anchor_mode", "spot")

    target_mgate = MGATE(
        hidden_dims1=source_mgate.hidden_dims1,
        hidden_dims2=source_mgate.hidden_dims2,
        spot_num=target_spot_num,
        temp=float(source_mgate.logit_scale.detach().cpu().item()),
        nonlinear=source_mgate.nonlinear,
        vgp_anchor_mode=vgp_anchor_mode,
    ).to(device)

    if vgp_anchor_mode == "feature":
        target_mgate.load_state_dict(source_mgate.state_dict(), strict=False)
    else:  # vgp_anchor_mode == "spot"
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
    
    if args.stage2_run_name is not None and args.target_model != "stage2":
        raise ValueError("--stage2-run-name is only valid with --target-model stage2.")

    deterministic_seed = 0
    random.seed(deterministic_seed)
    np.random.seed(deterministic_seed)
    sc.settings.n_jobs = 1

    torch.manual_seed(deterministic_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(deterministic_seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        warnings.warn("Could not enable full deterministic torch algorithms: {}".format(exc))
    torch.set_num_threads(1)

    # CPU inference avoids residual GPU sparse-op nondeterminism.
    device = torch.device("cpu")
    print("Device:", device, "(deterministic mode)")

    #%% ── MLflow setup ────────────────────────────────────────────────────────
    #args.run_name = '20260331_105802'
    #args.stage2_run_name = '20260331_105802_stage2_20260331_122453'

    setup_mlflow_tracking()
    client = MlflowClient()
    run_id = resolve_run_id_from_name(client, args.run_name)
    run_params = load_run_params(client, run_id, args.run_name)
    parent_run = client.get_run(run_id)
    experiment_id = parent_run.info.experiment_id

    #%% Resolve graph type for the target (arg takes precedence over run param)
    spatial_graph_type = args.spatial_graph_type or run_params.get("stage1_student_graph", "identity")
    if spatial_graph_type in {"NA", "na", "None", "none", "", None}:
        spatial_graph_type = "identity"
    print("Target spatial graph type:", spatial_graph_type)

    bp_width       = int(run_params.get("bp_width", 400))
    graph_type     = run_params.get("graph_type", "ATAC")
    dual_source_kd = run_params.get("stage1_dual_source_kd", "False").lower() == "true"
    target_subsample_n = int(run_params.get("target_subsample_n", 5000))
    target_subsample_seed = int(run_params.get("target_subsample_seed", 0))
    if target_subsample_n <= 0:
        raise ValueError("--target-subsample-n must be a positive integer.")

    # Validate model selections against available artifacts
    if args.source_model in ("stage1_teacher", "stage1_student") and not dual_source_kd:
        raise ValueError(
            "--source-model {} requires the training run to have used "
            "--stage1-dual-source-kd, but 'stage1_dual_source_kd={}' in run params.".format(
                args.source_model, run_params.get("stage1_dual_source_kd")
            )
        )
    resolved_stage2_run_id = None
    resolved_stage2_run_name = None
    resolved_stage2_source_kind = None
    if args.target_model == "stage2":
        resolved_stage2_run_id, resolved_stage2_run_name, resolved_stage2_source_kind = resolve_stage2_model_run(
            client=client,
            experiment_id=experiment_id,
            parent_run_id=run_id,
            stage2_run_name=args.stage2_run_name,
        )
        if resolved_stage2_source_kind == "child":
            print(
                "Resolved stage2 model run: child '{}' (ID: {})".format(
                    resolved_stage2_run_name,
                    resolved_stage2_run_id,
                )
            )
        else:
            if args.stage2_run_name and resolved_stage2_run_name:
                print(
                    "Resolved stage2 model run: parent '{}' (ID: {}) [legacy {}; "
                    "not found on child '{}' — loading parent checkpoint]".format(
                        args.run_name,
                        run_id,
                        STAGE2_MODEL_ARTIFACT,
                        resolved_stage2_run_name,
                    )
                )
            else:
                print(
                    "Resolved stage2 model run: parent '{}' (ID: {}) [legacy fallback]".format(
                        args.run_name,
                        run_id,
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
    if not target_rna.obs_names.equals(target_atac.obs_names):
        raise AssertionError("Target RNA and ATAC must have matching obs_names")

    # Flip spatial coordinates (same convention as training)
    source_rna.obsm["spatial"]  = source_rna.obsm["spatial"] * -1
    source_atac.obsm["spatial"] = source_atac.obsm["spatial"] * -1

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
    source_rna, source_atac, target_rna, target_atac, gp_net = apply_hvg_and_gp_filtering(
        source_rna=source_rna,
        source_atac=source_atac,
        target_rna=target_rna,
        target_atac=target_atac,
        gp_net=gp_net,
        top_n_genes=int(run_params.get("n_genes")),
        top_n_peaks=int(run_params.get("n_peaks")),
        rank_type="fused",
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
    #teacher_source_artifact_name = source_artifact_map["stage1_teacher"]

    print(
        "\nDownloading model artifacts for run '{}' (ID: {})...".format(
            args.run_name, run_id
        )
    )
    with tempfile.TemporaryDirectory() as tmpdir:

        print("Source model ({}) :".format(source_artifact_name))
        source_mgate, hidden_dims1, hidden_dims2, source_vgp_anchor_mode = load_mgate(
            client, run_id, source_artifact_name, device, tmpdir
        )

        if dual_source_kd:
            teacher_source_mgate, _, _, _ = load_mgate(
                client, run_id, source_artifact_map["stage1_teacher"], device, tmpdir
            )
        else:
            teacher_source_mgate = source_mgate

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
            if resolved_stage2_source_kind == "child":
                print(
                    "Target model (model_stage2.pth) from child run '{}' (ID: {}):".format(
                        resolved_stage2_run_name,
                        resolved_stage2_run_id,
                    )
                )
            else:
                print(
                    "Target model (model_stage2.pth) from parent run '{}' (ID: {}) [legacy fallback]:".format(
                        args.run_name,
                        run_id,
                    )
                )
            target_mgate, _, _, _ = load_mgate(
                client, resolved_stage2_run_id, "model_stage2.pth", device, tmpdir
            )
        else:
            # Zero-shot: built after download, outside the tmpdir block
            target_mgate = None

    #%% ── Subsample source cells to match target cells ────────────────────────────────────────────────────────────
    '''
    print("[TMP] Subsampling source cells to match target cells...")
    source_rna, source_atac = pair_and_subsample_target(
        source_rna, source_atac,
        subsample_n=args.target_subsample_n,
        seed=args.target_subsample_seed,
    )
    print("  {} cells after pairing/subsampling".format(source_rna.n_obs))
    '''
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

    source_student_graph_type = run_params.get("stage1_student_graph", "identity")

    if args.source_model == "stage1_teacher":
        source_infer_graph_tf = source_graph_tf
        source_graph_mode = "spatial_teacher_graph"
    elif args.source_model == "stage1_student":
        source_infer_graph_tf = build_source_student_graph_tf(
            source_rna=source_rna,
            spatial_graph_type=source_student_graph_type,
        )
        source_graph_mode = "student_graph({})".format(source_student_graph_type)
    else:
        if dual_source_kd:
            source_infer_graph_tf = build_source_student_graph_tf(
                source_rna=source_rna,
                spatial_graph_type=source_student_graph_type,
            )
            source_graph_mode = "stage1_primary_student_graph({})".format(source_student_graph_type)
        else:
            source_infer_graph_tf = source_graph_tf
            source_graph_mode = "stage1_primary_teacher_graph(spatial)"
    print("  Source inference graph:", source_graph_mode)

    # ── Target graphs ────────────────────────────────────────────────────────
    print("Building target graph (type: '{}')...".format(spatial_graph_type))
    target_rna, target_atac = prepare_target_for_spatial_graph_type(
        target_rna=target_rna,
        target_atac=target_atac,
        source_rna=source_rna,
        source_atac=source_atac,
        spatial_graph_type=spatial_graph_type,
        gtf_path=gtf_path,
    )
    print("\nPairing and subsampling target cells...")
    target_rna, target_atac = pair_and_subsample_target(
        target_rna,
        target_atac,
        subsample_n=target_subsample_n,
        seed=target_subsample_seed,
    )
    print("  {} cells after pairing/subsampling".format(target_rna.n_obs))
    target_graph_tf, target_gp_tf, target_x1, target_x2 = build_graph_inputs(
        target_rna, target_atac, bp_width=bp_width, graph_type=graph_type
    )
    print("  Target graph built: {} cells".format(target_rna.n_obs))

    # Build zero-shot target model after target pairing/subsampling so spot-anchored
    # vgp has the correct target cell count.
    if args.target_model == "zero_shot":
        run_vgp_anchor_mode = run_params.get("vgp_anchor_mode")
        effective_vgp_anchor_mode = (
            args.vgp_anchor_mode
            or run_vgp_anchor_mode
            or source_vgp_anchor_mode
        )
        print(
            "\nBuilding zero-shot target model from source weights "
            "(vgp_anchor_mode='{}')...".format(
                effective_vgp_anchor_mode
            )
        )
        target_mgate = build_zero_shot_mgate(
            source_mgate=source_mgate,
            target_spot_num=target_rna.n_obs,
            vgp_anchor_mode=effective_vgp_anchor_mode,
            device=device,
        )

    #%% ── Inference, by dataset ────────────────────────────────────────────────────────────

    ## (teacher) source inference
    teacher_source_rna_emb, teacher_source_atac_emb = run_inference(
        teacher_source_mgate, source_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, teacher_source_rna_emb, teacher_source_atac_emb, key_added="MultiGATE_teacher")
    print("  Teacher source embeddings: shape {}".format(teacher_source_rna_emb.shape))
    
    ## (student) source inference
    source_rna_emb, source_atac_emb = run_inference(
        source_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, source_rna_emb, source_atac_emb)
    print("  Source embeddings: shape {}".format(source_rna_emb.shape))

    # Plot source/target concat UMAPs with the same helper as training script.
    teacher_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_teacher")
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE")

    ## compute UMAPs
    compute_concat_umap(
        teacher_source_concat_adata,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        source_concat_adata,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )

    # plot teacher source UMAPs
    teacher_source_umap_colors = ['modality', 'leiden', 'RNA_clusters']
    fig, axs = plt.subplots(1, len(teacher_source_umap_colors), figsize=(18, 5))
    for i, color in enumerate(teacher_source_umap_colors):
        sc.pl.umap(teacher_source_concat_adata, color=color, ncols=3, wspace=0.2, size=25, ax=axs[i], show=False)
    plt.tight_layout(); plt.show()

    # plot source UMAPs
    source_umap_colors = ['modality', 'leiden', 'RNA_clusters']
    fig, axs = plt.subplots(1, len(source_umap_colors), figsize=(18, 5))
    for i, color in enumerate(source_umap_colors):
        sc.pl.umap(source_concat_adata, color=color, ncols=3, wspace=0.2, size=25, ax=axs[i], show=False)
    plt.tight_layout(); plt.show()

    if args.target_model:

        ## student target inference
        target_rna_emb, target_atac_emb = run_inference(
            target_mgate, target_graph_tf, target_gp_tf, target_x1, target_x2, device
        )
        set_multigate_embeddings(target_rna, target_atac, target_rna_emb, target_atac_emb)
        print("  Target embeddings: shape {}".format(target_rna_emb.shape))

        target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key="MultiGATE")

        compute_concat_umap(
            target_concat_adata,
            n_neighbors=10,
            resolution=0.5,
            deterministic=True,
            random_state=deterministic_seed,
        )
        target_concat_adata.obs["arc_gex_kmeans_5_clusters_Cluster"] = target_concat_adata.obs["arc_gex_kmeans_5_clusters_Cluster"].astype("category")

        # plot target UMAPs
        target_umap_colors = ['modality', 'leiden', 'arc_gex_kmeans_5_clusters_Cluster']
        fig, axs = plt.subplots(1, len(target_umap_colors), figsize=(18, 5))
        for i, color in enumerate(target_umap_colors):
            sc.pl.umap(target_concat_adata, color=color, ncols=3, wspace=0.2, size=25, ax=axs[i], show=False)
        plt.tight_layout(); plt.show()

    #%% Inference, all same model

    model = target_mgate

    ## (teacher) source inference
    teacher_source_rna_emb, teacher_source_atac_emb = run_inference(
        model, source_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, teacher_source_rna_emb, teacher_source_atac_emb, key_added="MultiGATE_teacher")
    print("  Teacher source embeddings: shape {}".format(teacher_source_rna_emb.shape))
    
    ## (student) source inference
    source_rna_emb, source_atac_emb = run_inference(
        model, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, source_rna_emb, source_atac_emb)
    print("  Source embeddings: shape {}".format(source_rna_emb.shape))

    ## (target) inference
    target_rna_emb, target_atac_emb = run_inference(
        model, target_graph_tf, target_gp_tf, target_x1, target_x2, device
    )
    set_multigate_embeddings(target_rna, target_atac, target_rna_emb, target_atac_emb)
    print("  Target embeddings: shape {}".format(target_rna_emb.shape))

    teacher_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_teacher")
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE")
    target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key="MultiGATE")

    compute_concat_umap(
        teacher_source_concat_adata,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        source_concat_adata,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        target_concat_adata,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    target_concat_adata.obs["arc_gex_kmeans_5_clusters_Cluster"] = target_concat_adata.obs["arc_gex_kmeans_5_clusters_Cluster"].astype("category")

    sc.pl.umap(teacher_source_concat_adata, color=['modality', 'leiden', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()
    sc.pl.umap(source_concat_adata, color=['modality', 'leiden', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()
    sc.pl.umap(target_concat_adata, color=['modality', 'leiden', 'arc_gex_kmeans_5_clusters_Cluster'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()


    #%% AJIVE analysis
    from mvlearn.decomposition import AJIVE
    import seaborn as sns
    import pandas as pd

    ## spatial source inference
    teacher_spatial_source_rna_emb, teacher_spatial_source_atac_emb = run_inference(
        teacher_source_mgate, source_graph_tf, source_gp_tf, source_x1, source_x2, device
    )    
    ## non-spatial source inference
    student_non_spatial_source_rna_emb, student_non_spatial_source_atac_emb = run_inference(
        source_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )

    ## AJIVE analysis
    X1 = teacher_spatial_source_rna_emb.copy()
    X2 = student_non_spatial_source_rna_emb.copy()

    U, S, V = np.linalg.svd(X1)
    vars1 = S**2 / np.sum(S**2)
    cumsum_vars1 = np.cumsum(vars1)
    U, S, V = np.linalg.svd(X2)
    vars2 = S**2 / np.sum(S**2)
    cumsum_vars2 = np.cumsum(vars2)
    n_plot_cps = X1.shape[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axes[0].plot(np.arange(n_plot_cps) + 1, cumsum_vars1[:n_plot_cps], 'ro-', linewidth=2)
    axes[1].plot(np.arange(n_plot_cps) + 1, cumsum_vars2[:n_plot_cps], 'ro-', linewidth=2)
    axes[0].set_title('Cumulative Variance View 1')
    axes[1].set_title('Cumulative Variance View 2')
    axes[0].set_xlabel('Number of top singular values')
    axes[1].set_xlabel('Number of top singular values')
    axes[0].set_ylabel('Cumulative percent variance explained')
    # Add grid to both plots
    axes[0].grid(True, color='#dddddd', linewidth=0.5)
    axes[1].grid(True, color='#dddddd', linewidth=0.5)
    # Add horizontal line at cumsum_vars==0.9
    axes[0].axhline(y=0.9, color='gray', linestyle='--', linewidth=1)
    axes[1].axhline(y=0.9, color='gray', linestyle='--', linewidth=1)
    plt.show()

    ## fit AJIVE
    source_rna_emb_list = [X1, X2]

    ajive = AJIVE(
        #init_signal_ranks=[7, 10],
        joint_rank=None,
        n_jobs=1)

    source_rna_joint = ajive.fit_transform(source_rna_emb_list)

    def plot_blocks(blocks, names):
        n_views = len(blocks[0])
        n_blocks = len(blocks)
        for i in range(n_views):
            for j in range(n_blocks):
                plt.subplot(n_blocks, n_views, j*n_views+i+1)
                sns.heatmap(blocks[j][i], xticklabels=False, yticklabels=False,
                            cmap="RdBu")
                plt.title(f"View {i}: {names[j]}")

    ## plot AJIVE blocks
    plt.figure(figsize=[15, 10])
    plt.title('Different Views')
    individual_mats = ajive.individual_mats_
    Xs_inv = ajive.inverse_transform(source_rna_joint)
    residuals = [v - X for v, X in zip(source_rna_emb_list, Xs_inv)]
    plot_blocks([source_rna_emb_list, source_rna_joint, individual_mats, residuals],
                ["Raw Data", "Joint", "Individual", "Noise"])

    ## UMAP of AJIVE projection embeddings
    import anndata as ad
    obs_index = np.concatenate([source_rna.obs_names, source_rna.obs_names])
    obs_df = pd.DataFrame(
        {
            "teacher_or_student": (["teacher"] * source_rna.n_obs + ["student"] * source_rna.n_obs),
            "RNA_clusters": source_rna.obs["RNA_clusters"].tolist() + source_rna.obs["RNA_clusters"].tolist(),
        },
        index=obs_index
    )
    concat_ajive_joint = ad.AnnData(
        X=np.concatenate([source_rna_joint[0], source_rna_joint[1]], axis=0),
        obs=obs_df,
    )
    concat_individual_mats = ad.AnnData(
        X=np.concatenate([individual_mats[0], individual_mats[1]], axis=0),
        obs=obs_df,
    )
    concat_residuals = ad.AnnData(
        X=np.concatenate([residuals[0], residuals[1]], axis=0),
        obs=obs_df,
    )
    compute_concat_umap(
        concat_ajive_joint,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        concat_individual_mats,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        concat_residuals,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )

    sc.pl.umap(concat_ajive_joint, color=['teacher_or_student', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()

    sc.pl.umap(concat_individual_mats, color=['teacher_or_student', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()

    sc.pl.umap(concat_residuals, color=['teacher_or_student', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()

    teacher_individual_mat = concat_individual_mats[concat_individual_mats.obs["teacher_or_student"].eq("teacher")]
    student_individual_mat = concat_individual_mats[concat_individual_mats.obs["teacher_or_student"].eq("student")]

    compute_concat_umap(
        teacher_individual_mat,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        student_individual_mat,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    sc.pl.umap(teacher_individual_mat, color=['teacher_or_student'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()
    sc.pl.umap(student_individual_mat, color=['teacher_or_student'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()

    #%% source student & teacher analysis

    corr_matrix = np.corrcoef(teacher_source_rna_emb, source_rna_emb, rowvar=False)
    n_dims = source_rna_emb.shape[1]
    #corr_matrix = corr_matrix[:n_dims, n_dims:]

    plt.figure(figsize=(6, 6))
    plt.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation between teacher and student source embeddings')
    plt.show()

    # Compute per-dimension Pearson correlation between teacher and student source embeddings
    C = teacher_source_rna_emb - source_rna_emb
    mu = C.mean(axis=0)
    sigma = C.std(axis=0)
    C_std = (C - mu) / (sigma + 1e-6)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_dims)
    pca.fit(C_std)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # Plot barcharts of the first 5 principal component loadings using seaborn as different series
    import seaborn as sns
    import pandas as pd

    n_pcs = 5
    loadings_df = pd.DataFrame(
        pca.components_[:n_pcs].T,
        columns=[f'PC{i+1}' for i in range(n_pcs)]
    )
    loadings_df['Original Dimension'] = loadings_df.index

    loadings_melted = loadings_df.melt(id_vars='Original Dimension', var_name='Principal Component', value_name='Loading')

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=loadings_melted,
        hue='Original Dimension',
        x='Principal Component',
        y='Loading',
        alpha=0.7
    )
    plt.xlabel('Original Dimension')
    plt.ylabel('Component Loading Value')
    plt.title('Barcharts of PCA Component Loadings (First 5 PCs)')
    plt.legend(title='Principal Component')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.bar(range(n_dims), corr_sorted)
    plt.title('Per-dimension Pearson correlation between teacher and student source embeddings')
    plt.xticks(range(n_dims), corr_sort, rotation=45)
    plt.ylabel('Pearson r')
    plt.show()

    ## remove top-k dimensions with largest spatial difference
    top_k = 5
    #keep_dims = np.arange(n_dims)[np.intersect1d(np.arange(n_dims), corr_sort[:-(top_k+1)])]
    keep_dims = np.arange(n_dims)[np.intersect1d(np.arange(n_dims), corr_sort[top_k:])]

    set_multigate_embeddings(source_rna, source_atac,
        teacher_source_rna_emb[:, keep_dims],
        teacher_source_atac_emb[:, keep_dims],
        key_added="MultiGATE_teacher_trunc")
    set_multigate_embeddings(source_rna, source_atac,
        source_rna_emb[:, keep_dims],
        source_atac_emb[:, keep_dims],
        key_added="MultiGATE_trunc")
    print("  Teacher source truncated embeddings: shape {}".format(teacher_source_rna_emb[:, keep_dims].shape))
    print("  Student source truncated embeddings: shape {}".format(source_rna_emb[:, keep_dims].shape))

    teacher_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_teacher_trunc")
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_trunc")
    compute_concat_umap(
        teacher_source_concat_adata,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        source_concat_adata,
        n_neighbors=10,
        resolution=1.5,
        deterministic=True,
        random_state=deterministic_seed,
    )

    # plot teacher source truncated UMAPs
    teacher_source_trunc_umap_colors = ['modality', 'leiden', 'RNA_clusters']
    fig, axs = plt.subplots(1, len(teacher_source_trunc_umap_colors), figsize=(18, 5))
    for i, color in enumerate(teacher_source_trunc_umap_colors):
        sc.pl.umap(teacher_source_concat_adata, color=color, ncols=3, wspace=0.2, size=25, ax=axs[i], show=False)
    plt.tight_layout(); plt.show()

    # plot source truncated UMAPs
    source_trunc_umap_colors = ['modality', 'leiden', 'RNA_clusters']
    fig, axs = plt.subplots(1, len(source_trunc_umap_colors), figsize=(18, 5))
    for i, color in enumerate(source_trunc_umap_colors):
        sc.pl.umap(source_concat_adata, color=color, ncols=3, wspace=0.2, size=25, ax=axs[i], show=False)
    plt.tight_layout(); plt.show()


    #%% ── Inference, combined target embeddings ────────────────────────────────────────────────────────────
    source_rna_emb, source_atac_emb = run_inference(
        target_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, source_rna_emb, source_atac_emb)
    print("  Source embeddings: shape {}".format(source_rna_emb.shape))

    target_rna_emb, target_atac_emb = run_inference(
        target_mgate, target_graph_tf, target_gp_tf, target_x1, target_x2, device
    )
    set_multigate_embeddings(target_rna, target_atac, target_rna_emb, target_atac_emb)
    print("  Target embeddings: shape {}".format(target_rna_emb.shape))

    # Plot combined UMAP
    source_target_rna = sc.concat([source_rna, target_rna], axis=0)
    source_target_atac = sc.concat([source_atac, target_atac], axis=0)
    source_target_rna.obs["source_or_target"] = ["source"] * source_rna.n_obs + ["target"] * target_rna.n_obs
    source_target_atac.obs["source_or_target"] = ["source"] * source_atac.n_obs + ["target"] * target_atac.n_obs

    source_target_adata = build_concat_adata_for_umap(source_target_rna, source_target_atac, embedding_key="MultiGATE")
    compute_concat_umap(
        source_target_adata,
        n_neighbors=50,
        resolution=0.5,
        deterministic=True,
        random_state=deterministic_seed,
    )
    # Randomly permute the rows before plotting the UMAP
    permuted_idx = np.random.RandomState(deterministic_seed).permutation(source_target_adata.n_obs)
    sc.pl.umap(
        source_target_adata[permuted_idx],
        color=["modality", "source_or_target", "leiden"],
        ncols=3,
        wspace=0.2,
        size=25,
    )

    #%% plot source and target spatial and UMAP
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    ## define bounding indices for source and target data
    bounds = [
        source_rna.n_obs,
        source_rna.n_obs + target_rna.n_obs,
        source_rna.n_obs + target_rna.n_obs + source_atac.n_obs,
    ]

    ## get connectivities between source and target data
    conns = source_target_adata.obsp['connectivities']
    conns_source_target_rna = conns[:bounds[0], bounds[0]:bounds[1]].toarray()
    conns_source_target_atac = conns[bounds[1]:bounds[2], bounds[2]:].toarray()

    ## get optimal assignment of source and target data
    rna_source_idx, rna_target_idx = linear_sum_assignment(1 - conns_source_target_rna)
    atac_source_idx, atac_target_idx = linear_sum_assignment(1 - conns_source_target_atac)

    ## create arrays for target spatial coordinates
    target_rna_spatial = np.empty((target_rna.n_obs, source_rna.obsm["spatial"].shape[1]))
    target_atac_spatial = np.empty((target_atac.n_obs, source_atac.obsm["spatial"].shape[1]))
    target_rna_spatial[rna_target_idx] = source_rna.obsm["spatial"][rna_source_idx]
    target_atac_spatial[atac_target_idx] = source_atac.obsm["spatial"][atac_source_idx]

    ## add spatial coordinates to source_target_adata
    source_target_adata.obsm["spatial"] = np.concatenate([
        source_rna.obsm["spatial"],
        target_rna_spatial,
        source_atac.obsm["spatial"],
        target_atac_spatial,
    ], axis=0)

    ## assign target spatial coordinates to source_target_adata
    source_adata = source_target_adata[source_target_adata.obs["source_or_target"].eq("source")]
    target_adata = source_target_adata[source_target_adata.obs["source_or_target"].eq("target")]

    ## plot source and target spatial and UMAP
    _, axs = plt.subplots(2, 2, figsize=(10, 8))
    sc.pl.embedding(source_adata, basis="spatial", color="leiden", s=50, show=False, ax=axs[0, 0], legend_loc='None')
    sc.pl.umap(source_adata, color="leiden", ax=axs[0, 1], size=50, show=False)
    sc.pl.embedding(target_adata, basis="spatial", color="leiden", s=50, show=False, ax=axs[1, 0], legend_loc='None')
    sc.pl.umap(target_adata, color="leiden", ax=axs[1, 1], size=50, show=False)
    axs[0, 0].set_title('Source Spatial'); axs[0, 1].set_title('Source UMAP'); axs[1, 0].set_title('Target Spatial'); axs[1, 1].set_title('Target UMAP')
    plt.tight_layout(); plt.show()
    
    #%% attention matrix analysis
    from post_hoc_utils import run_gene_peak_attention_tutorial
    import scipy.sparse as sp

    # Find the artifact path for the attention matrix
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="matrices/source_peak_gene_attention.npz", dst_path=tmp_dir)
        source_peak_gene_attention = sp.load_npz(local_path)

    attention_analysis_summary = run_gene_peak_attention_tutorial(
        peak_gene_attention=source_peak_gene_attention,
        adata_rna=source_rna,
        adata_atac=source_atac,
    )

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
