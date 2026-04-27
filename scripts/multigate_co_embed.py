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

  When combined_gp_dict is loaded (omit --no-combined-gp-dict), pathway-vs-embedding
  Spearman analysis runs in memory; ``main()`` returns those results as the 5th value.
"""
#%%
import argparse
import json
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

baklava_repo_root = os.path.join(BAKLAVA_BASE_DIR, "BAKLAVA")
if os.path.isdir(baklava_repo_root) and baklava_repo_root not in sys.path:
    sys.path.insert(0, baklava_repo_root)

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
from joblib import Parallel, delayed
import seaborn as sns

import MultiGATE
from MultiGATE.model_MultiGATE import MGATE

print("Using MultiGATE module:", MultiGATE.__file__)

from mouse_brain_spatial_rna_atac import (  # noqa: E402
    apply_hvg_and_gp_filtering,
    build_concat_adata_for_umap,
    build_graph_inputs,
    build_source_student_graph_tf,
    compute_concat_umap,
    load_nichecompass_combined_gp_dict_mouse,
    pair_and_subsample_target,
    prepare_target_for_spatial_graph_type,
    set_multigate_embeddings,
)


def run_co_embed_pathway_embedding_analysis(
    *,
    combined_gp_dict,
    source_rna,
    target_rna,
    embedding_key="MultiGATE",
    include_teacher=False,
    cluster_obs_key="leiden",
):
    """
    Spearman correlation between NicheCompass pathway activity scores and MultiGATE obsm columns.

    Returns a dict of ``PathwayEmbeddingResult`` objects (from ``multigate_pathway_embedding_analysis``)
    keyed by ``source_rna``, ``target_rna``, and optionally ``source_rna_teacher``. Values are ``None``
    if that run was skipped (missing embedding) or failed. Returns ``None`` if ``combined_gp_dict`` is
    missing.

    Inspect in a session: ``result["source_rna"].correlation``, ``.pathway_scores``, ``.p_values``, etc.
    If ``cluster_obs_key`` is set (e.g. ``"leiden"`` or ``"RNA_clusters"``) and that column exists on
    ``adata.obs``, each result includes ``pathway_mean_by_cluster`` (groups × pathways) and
    ``cluster_obs_key_used``. Pass ``cluster_obs_key=None`` to skip group means.
    To persist, call ``save_pathway_embedding_results(result[k], out_dir)`` on any non-None entry.
    """
    if combined_gp_dict is None:
        print(
            "[pathway_embedding_analysis] Skipped: no combined_gp_dict "
            "(omit --no-combined-gp-dict to load it)."
        )
        return None

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from multigate_pathway_embedding_analysis import (  # noqa: E402
        PathwayEmbeddingConfig,
        run_pathway_embedding_analysis,
    )

    results = {}

    def _one(label, adata, key):
        if key not in adata.obsm:
            print(
                "[pathway_embedding_analysis] Skip {}: obsm['{}'] missing (keys: {}).".format(
                    label,
                    key,
                    list(adata.obsm.keys()),
                )
            )
            results[label] = None
            return
        cfg = PathwayEmbeddingConfig(embedding_key=key, cluster_obs_key=cluster_obs_key)
        try:
            res = run_pathway_embedding_analysis(adata, combined_gp_dict, cfg)
            results[label] = res
            cluster_msg = ""
            if res.pathway_mean_by_cluster is not None:
                cluster_msg = "; mean by obs['{}'] {} × {}".format(
                    res.cluster_obs_key_used or cluster_obs_key,
                    res.pathway_mean_by_cluster.shape[0],
                    res.pathway_mean_by_cluster.shape[1],
                )
            cluster_link_msg = ""
            if res.pathway_embedding_correlation_by_cluster is not None:
                cluster_link_msg = "; cluster-centroid corr {} pathways x {} dims".format(
                    res.pathway_embedding_correlation_by_cluster.shape[0],
                    res.pathway_embedding_correlation_by_cluster.shape[1],
                )
            print(
                "[pathway_embedding_analysis] {}: {} pathways x {} dims{}{}.".format(
                    label,
                    res.correlation.shape[0],
                    res.correlation.shape[1],
                    cluster_msg,
                    cluster_link_msg,
                )
            )
        except Exception as exc:
            print("[pathway_embedding_analysis] Failed for {}: {}".format(label, exc))
            results[label] = None

    _one("source_rna", source_rna, embedding_key)
    _one("target_rna", target_rna, embedding_key)
    if include_teacher:
        _one("source_rna_teacher", source_rna, "MultiGATE_teacher")

    return results


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
        "--mlflow-backend",
        choices=["auto", "sqlite", "server"],
        default="auto",
        help=(
            "Tracking backend mode. "
            "'auto' infers backend from --tracking-uri / MLFLOW_TRACKING_URI. "
            "'sqlite' uses sqlite:///.../mlflow.db (auto-builds from --mlflow-base-dir when URI is absent). "
            "'server' requires an http(s) tracking URI."
        ),
    )
    p.add_argument(
        "--tracking-uri",
        default=None,
        help=(
            "Explicit MLflow tracking URI override. "
            "Takes precedence over MLFLOW_TRACKING_URI."
        ),
    )
    p.add_argument(
        "--mlflow-base-dir",
        default=None,
        help=(
            "Base directory for MultiGATE MLflow namespace. "
            "Used to construct sqlite:///.../mlflow.db when --mlflow-backend sqlite "
            "and no tracking URI is provided."
        ),
    )
    p.add_argument(
        "--experiment-name",
        default="multigate_mouse_brain_live_zeroshot",
        help=(
            "MLflow experiment name used to resolve --run-name. "
            "Defaults to the training script experiment."
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
    p.add_argument(
        "--no-combined-gp-dict",
        action="store_true",
        default=False,
        help=(
            "Skip loading the NicheCompass combined_gp_dict (same as mouse_brain_spatial_rna_atac.py). "
            "Also skips in-memory pathway-vs-embedding analysis."
        ),
    )
    if notebook:
        return p.parse_known_args()[0]
    else:
        return p.parse_args()


# ─── MLflow helpers ───────────────────────────────────────────────────────────

def _normalize_mlflow_base_dir(mlflow_base_dir=None):
    if mlflow_base_dir is not None:
        raw_base_dir = mlflow_base_dir
    elif os.environ.get("MLFLOW_BASE_DIR"):
        raw_base_dir = os.environ["MLFLOW_BASE_DIR"]
    else:
        raw_base_dir = os.path.join(BAKLAVA_BASE_DIR, "mlflow_tracking")

    normalized = os.path.abspath(raw_base_dir)
    if os.path.basename(normalized.rstrip(os.sep)) != "MultiGATE":
        normalized = os.path.join(normalized, "MultiGATE")
    return normalized


def _infer_tracking_backend_from_uri(tracking_uri):
    if tracking_uri.startswith("sqlite:///"):
        return "sqlite"
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        return "server"
    raise ValueError(
        "Unsupported tracking URI '{}'. Expected sqlite:///..., http://..., or https://...".format(
            tracking_uri
        )
    )


def _tracking_backend_hint(tracking_config):
    if not tracking_config:
        return ""

    backend = tracking_config.get("backend")
    if backend == "sqlite":
        db_path = tracking_config.get("mlflow_db_path")
        return (
            " Backend=sqlite. Ensure this matches the training DB ({}) or pass "
            "--tracking-uri sqlite:///.../mlflow.db / --mlflow-base-dir explicitly."
        ).format(db_path)

    if backend == "server":
        uri = tracking_config.get("tracking_uri")
        return (
            " Backend=server at {}. Ensure the MLflow host is reachable from this job; "
            "on remote/Slurm, 127.0.0.1 only works when the server is on the same node."
        ).format(uri)

    return ""


def setup_mlflow_tracking(
    backend_type="auto",
    tracking_uri=None,
    mlflow_base_dir=None,
    experiment_name="multigate_mouse_brain_live_zeroshot",
):
    if backend_type == "postgres":
        backend_type = "server"

    valid_backends = ["auto", "sqlite", "server"]
    if backend_type not in valid_backends:
        raise ValueError(
            "Invalid MLflow backend type '{}'. Expected one of {}.".format(
                backend_type, valid_backends
            )
        )

    normalized_base_dir = _normalize_mlflow_base_dir(mlflow_base_dir)
    resolved_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    resolved_backend = backend_type

    if backend_type == "auto":
        if resolved_tracking_uri is None:
            raise ValueError(
                "MLflow backend is 'auto' but no tracking URI is configured. "
                "Set --tracking-uri or MLFLOW_TRACKING_URI, or choose "
                "--mlflow-backend sqlite to use a local mlflow.db."
            )
        resolved_backend = _infer_tracking_backend_from_uri(resolved_tracking_uri)
    elif backend_type == "sqlite":
        if resolved_tracking_uri is None:
            mlflow_db_path = os.path.join(normalized_base_dir, "mlflow.db")
            resolved_tracking_uri = "sqlite:///{}".format(mlflow_db_path)
        inferred_backend = _infer_tracking_backend_from_uri(resolved_tracking_uri)
        if inferred_backend != "sqlite":
            raise ValueError(
                "--mlflow-backend sqlite requires sqlite:///... URI, got '{}'.".format(
                    resolved_tracking_uri
                )
            )
    else:  # server
        if resolved_tracking_uri is None:
            raise ValueError(
                "--mlflow-backend server requires --tracking-uri or MLFLOW_TRACKING_URI "
                "(http://... or https://...)."
            )
        inferred_backend = _infer_tracking_backend_from_uri(resolved_tracking_uri)
        if inferred_backend != "server":
            raise ValueError(
                "--mlflow-backend server requires http(s) URI, got '{}'.".format(
                    resolved_tracking_uri
                )
            )

    mlflow_db_path = None
    if resolved_tracking_uri.startswith("sqlite:///"):
        mlflow_db_path = resolved_tracking_uri[len("sqlite:///"):]

    os.environ["MLFLOW_TRACKING_URI"] = resolved_tracking_uri
    mlflow.set_tracking_uri(resolved_tracking_uri)

    tracking_config = {
        "backend": resolved_backend,
        "tracking_uri": resolved_tracking_uri,
        "mlflow_base_dir": normalized_base_dir,
        "mlflow_db_path": mlflow_db_path,
        "experiment_name": experiment_name,
    }
    print(
        "MLflow config: backend={}, tracking_uri={}, experiment={}".format(
            tracking_config["backend"],
            tracking_config["tracking_uri"],
            tracking_config["experiment_name"],
        )
    )
    if mlflow_db_path:
        print("MLflow sqlite db:", mlflow_db_path)
    return tracking_config


def resolve_run_id_from_name(
    client,
    run_name,
    experiment_name="multigate_mouse_brain_live_zeroshot",
    tracking_config=None,
):
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
            "Ensure the tracking URI points to the correct tracking store.{}".format(
                experiment_name
            ) + _tracking_backend_hint(tracking_config)
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.run_name = '{}'".format(run_name),
    )
    if len(runs) == 0:
        raise ValueError(
            "No run named '{}' found in experiment '{}'.{}".format(
                run_name, experiment_name
            ) + _tracking_backend_hint(tracking_config)
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
SPLIT_ARTIFACT = "splits/domain_splits.json"


def resolve_stage2_model_run(
    client,
    experiment_id,
    parent_run_id,
    stage2_run_name=None,
    tracking_config=None,
):
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
    backend_hint = _tracking_backend_hint(tracking_config)

    if stage2_run_name is not None:
        matches = [r for r in stage2_children if r["run_name"] == stage2_run_name]
        if len(matches) == 0:
            raise ValueError(
                "No stage2 child run named '{}' found under parent run {}. "
                "Available stage2 child runs: {}.{}".format(
                    stage2_run_name,
                    parent_run_id,
                    _format_stage2_run_choices(stage2_children),
                    backend_hint,
                )
            )
        if len(matches) > 1:
            raise ValueError(
                "Multiple stage2 child runs named '{}' found under parent run {}: {}. "
                "Use unique child run names.{}".format(
                    stage2_run_name,
                    parent_run_id,
                    _format_stage2_run_choices(matches),
                    backend_hint,
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
            "has none either. Confirm the Run ID matches and that stage-2 logging finished.{}".format(
                stage2_run_name,
                selected["run_id"],
                STAGE2_MODEL_ARTIFACT,
                parent_run_id,
                backend_hint,
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
            "Re-run stage-2 or remove the empty nested run.{}".format(
                selected["run_name"],
                selected["run_id"],
                STAGE2_MODEL_ARTIFACT,
                parent_run_id,
                backend_hint,
            )
        )

    if len(stage2_children) > 1:
        raise ValueError(
            "Found multiple stage2 child runs under parent run {}: {}. "
            "Pass --stage2-run-name to select one.{}".format(
                parent_run_id,
                _format_stage2_run_choices(stage2_children),
                backend_hint,
            )
        )

    if parent_has_legacy:
        return parent_run_id, None, "parent_legacy"

    raise ValueError(
        "No stage2 child runs found under parent run {} and no legacy "
        "parent artifact {} exists.{}".format(
            parent_run_id,
            STAGE2_MODEL_ARTIFACT,
            backend_hint,
        )
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


def load_domain_splits_artifact(client, run_id, dst_dir):
    if not artifact_exists(client, run_id, SPLIT_ARTIFACT):
        return None
    local_path = client.download_artifacts(run_id, SPLIT_ARTIFACT, dst_dir)
    with open(local_path, "r") as f:
        return json.load(f)


def subset_domain_to_saved_split(rna, atac, split_metadata, domain_name, split_name="eval"):
    if split_metadata is None:
        raise ValueError("split_metadata cannot be None when applying saved splits.")

    domain_payload = split_metadata.get("domains", {}).get(domain_name)
    if domain_payload is None:
        raise KeyError("Split metadata missing domain '{}'.".format(domain_name))
    split_payload = domain_payload.get("splits", {}).get(split_name)
    if split_payload is None:
        raise KeyError("Split metadata missing split '{}' for domain '{}'.".format(split_name, domain_name))

    split_obs_names = [str(v) for v in split_payload.get("obs_names", [])]
    if len(split_obs_names) == 0:
        raise ValueError("Split '{}' for domain '{}' has zero observations.".format(split_name, domain_name))

    missing_rna = [obs for obs in split_obs_names if obs not in rna.obs_names]
    missing_atac = [obs for obs in split_obs_names if obs not in atac.obs_names]
    if missing_rna or missing_atac:
        raise KeyError(
            "Saved split '{}' for domain '{}' does not match current AnnData obs_names "
            "(missing in RNA: {}, missing in ATAC: {}).".format(
                split_name,
                domain_name,
                len(missing_rna),
                len(missing_atac),
            )
        )
    return rna[split_obs_names].copy(), atac[split_obs_names].copy()


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


def rho_mask_mode_from_state_dict(state_dict):
    rho_mask_mode_tensor = state_dict.get("rho_mask_mode_code")
    if rho_mask_mode_tensor is not None:
        code = int(torch.as_tensor(rho_mask_mode_tensor).item())
        if code == 1:
            return "fixed"
        if code == 2:
            return "trainable_masked"

    rho_is_fixed_tensor = state_dict.get("rho_is_fixed_mask")
    rho_is_fixed = bool(int(torch.as_tensor(rho_is_fixed_tensor).item())) if rho_is_fixed_tensor is not None else False
    if rho_is_fixed:
        return "fixed"
    return None


def linear_decoder_kwargs_from_state_dict(state_dict):
    kwargs = {}
    alpha = state_dict.get("alpha")
    if alpha is not None:
        kwargs["etm_emb_dim"] = int(alpha.shape[1])

    rho_mask_mode = rho_mask_mode_from_state_dict(state_dict)
    if rho_mask_mode == "fixed" and ("rho_rna" in state_dict) and ("rho_atac" in state_dict):
        kwargs["rho_mask_mode"] = "fixed"
        kwargs["rho_rna_mask"] = state_dict["rho_rna"].detach().cpu().numpy()
        kwargs["rho_atac_mask"] = state_dict["rho_atac"].detach().cpu().numpy()
    elif (
        rho_mask_mode == "trainable_masked"
        and ("rho_rna_mask" in state_dict)
        and ("rho_atac_mask" in state_dict)
    ):
        kwargs["rho_mask_mode"] = "trainable_masked"
        kwargs["rho_rna_mask"] = state_dict["rho_rna_mask"].detach().cpu().numpy()
        kwargs["rho_atac_mask"] = state_dict["rho_atac_mask"].detach().cpu().numpy()
    return kwargs


def rho_mask_mode_from_mgate(mgate):
    rho_mask_mode_code = getattr(mgate, "rho_mask_mode_code", None)
    if rho_mask_mode_code is not None:
        code = int(rho_mask_mode_code.detach().cpu().item())
        if code == 1:
            return "fixed"
        if code == 2:
            return "trainable_masked"

    rho_mask_mode = getattr(mgate, "rho_mask_mode", None)
    if rho_mask_mode in {"fixed", "trainable_masked"}:
        return rho_mask_mode

    if hasattr(mgate, "rho_is_fixed_mask"):
        rho_is_fixed = bool(int(mgate.rho_is_fixed_mask.detach().cpu().item()))
        if rho_is_fixed:
            return "fixed"
    return None


def linear_decoder_kwargs_from_mgate(mgate):
    kwargs = {}
    if hasattr(mgate, "alpha"):
        kwargs["etm_emb_dim"] = int(mgate.alpha.shape[1])
    rho_mask_mode = rho_mask_mode_from_mgate(mgate)
    if rho_mask_mode == "fixed":
        kwargs["rho_mask_mode"] = "fixed"
        kwargs["rho_rna_mask"] = mgate.rho_rna.detach().cpu().numpy()
        kwargs["rho_atac_mask"] = mgate.rho_atac.detach().cpu().numpy()
    elif rho_mask_mode == "trainable_masked":
        kwargs["rho_mask_mode"] = "trainable_masked"
        kwargs["rho_rna_mask"] = mgate.rho_rna_mask.detach().cpu().numpy()
        kwargs["rho_atac_mask"] = mgate.rho_atac_mask.detach().cpu().numpy()
    return kwargs


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
        **linear_decoder_kwargs_from_state_dict(state_dict),
    ).to(device)
    mgate.load_state_dict(state_dict, strict=False)
    mgate.eval()
    return mgate, hidden_dims1, hidden_dims2, vgp_anchor_mode


def unpack_mgate_checkpoint_payload(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("pathway_metadata")
    return payload, None


def apply_pathway_metadata_to_mgate(mgate, pathway_metadata):
    if pathway_metadata is None:
        return
    for attr_name in ("pathway_names", "source_pathway_names", "target_pathway_names"):
        values = pathway_metadata.get(attr_name)
        if values is None:
            continue
        setattr(mgate, attr_name, np.asarray(values, dtype=object))
    for attr_name in ("n_zero_source_pathways", "n_zero_target_pathways"):
        value = pathway_metadata.get(attr_name)
        if value is None:
            continue
        setattr(mgate, attr_name, int(value))


def load_mgate(client, run_id, artifact_name, device, dst_dir):
    """Download a model .pth artifact and return a reconstructed, eval-mode MGATE."""
    local_path = download_model_artifact(client, run_id, artifact_name, dst_dir)
    payload = torch.load(local_path, map_location=device, weights_only=False)
    state_dict, pathway_metadata = unpack_mgate_checkpoint_payload(payload)
    mgate, hidden_dims1, hidden_dims2, vgp_anchor_mode = mgate_from_state_dict(state_dict, device)
    apply_pathway_metadata_to_mgate(mgate, pathway_metadata)
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
        **linear_decoder_kwargs_from_mgate(source_mgate),
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

    infer_mgate = mgate
    infer_mode = getattr(mgate, "vgp_anchor_mode", "spot")
    if infer_mode == "spot":
        current_spot_num = int(mgate.vgp0.shape[0])
        if current_spot_num != int(x1_t.shape[0]):
            infer_mgate = build_zero_shot_mgate(
                source_mgate=mgate,
                target_spot_num=int(x1_t.shape[0]),
                vgp_anchor_mode=infer_mode,
                device=device,
            )

    infer_mgate.eval()
    with torch.no_grad():
        outputs = infer_mgate(a_t, a_t, gp_t, x1_t, x2_t)

    rna_emb  = outputs[5].detach().cpu().numpy()
    atac_emb = outputs[6].detach().cpu().numpy()
    return rna_emb, atac_emb

def _concat_obs_to_barcode(concat_obs: pd.Index, modality: str) -> np.ndarray:
    suffix = "_rna" if modality == "rna" else "_atac"
    idx = pd.Index(concat_obs).astype(str)
    if not idx.str.endswith(suffix).all():
        raise ValueError("concat obs must end with %r for modality %r" % (suffix, modality))
    return idx.str.slice(0, -len(suffix)).values


def _spatial_target_from_source_reference(
    target_concat_adata, source_rna, source_atac,
    *,
    mapping_col: str = "source_obs_names",
):
    """One source barcode per target row (source-anchored UMAP + ingest)."""
    rna_t = target_concat_adata.obs["modality"].eq("rna")
    atac_t = target_concat_adata.obs["modality"].eq("atac")
    rna_keys = pd.Index(target_concat_adata.obs.loc[rna_t, mapping_col]).astype(str).str.rsplit("_", n=1).str[0]
    atac_keys = pd.Index(target_concat_adata.obs.loc[atac_t, mapping_col]).astype(str).str.rsplit("_", n=1).str[0]
    target_rna_spatial = np.asarray(source_rna[rna_keys].obsm["spatial"])
    target_atac_spatial = np.asarray(source_atac[atac_keys].obsm["spatial"])
    return target_rna_spatial, target_atac_spatial


def _spatial_target_from_target_reference(
    source_concat_adata,
    target_concat_adata,
    source_rna,
    source_atac,
    *,
    mapping_col: str = "target_obs_names",
):
    """Mean source spatial over source rows mapping to each target (target-anchored UMAP + ingest)."""
    def _block(tgt_mask, src_mask, modality, source_adata):
        tgt_names = target_concat_adata.obs_names[tgt_mask]
        src_ix = source_concat_adata.obs_names[src_mask]
        barcodes = _concat_obs_to_barcode(src_ix, modality)
        src_map = source_concat_adata.obs.loc[src_ix, mapping_col]
        coords = np.asarray(source_adata[barcodes].obsm["spatial"])
        out = np.full((tgt_names.size, coords.shape[1]), np.nan, dtype=float)
        for i, tname in enumerate(tgt_names):
            sel = src_map.to_numpy() == tname
            if np.any(sel):
                out[i] = coords[sel].mean(axis=0)
        return out

    rna_t = target_concat_adata.obs["modality"].eq("rna")
    atac_t = target_concat_adata.obs["modality"].eq("atac")
    rna_s = source_concat_adata.obs["modality"].eq("rna")
    atac_s = source_concat_adata.obs["modality"].eq("atac")
    return (
        _block(rna_t, rna_s, "rna", source_rna),
        _block(atac_t, atac_s, "atac", source_atac),
    )


def run_alignment_and_spatial_plot(
    model,
    source_mgate,
    target_mgate,
    source_rna,
    source_atac,
    source_concat_adata,
    target_concat_adata,
    deterministic_seed: int,
    umap_n_neighbors: int = 50,
    umap_resolution: float = 0.5,
    leiden_neighbors: int = 100,
    leiden_resolution: float = 0.5,
    embedding_point_size: float = 50.0,
):
    """
    Full path: inference → MultiGATE in obsm → concat → UMAP ref → ingest → Leiden →
    joint AnnData → spatial → sc.pl.embedding (spatial + UMAP panels).
    """

    # --- 3) Reference-specific UMAP + ingest + Leiden ----------------------------
    if model is source_mgate:
        compute_concat_umap(
            source_concat_adata,
            n_neighbors=umap_n_neighbors,
            resolution=umap_resolution,
            deterministic=True,
            random_state=deterministic_seed,
        )
        source_concat_adata.obs["source_obs_names"] = source_concat_adata.obs_names
        sc.tl.ingest(
            target_concat_adata,
            source_concat_adata,
            embedding_method="umap",
            obs="source_obs_names",
            k=1,
        )
        sc.pp.neighbors(source_concat_adata, use_rep="X", n_neighbors=leiden_neighbors)
        sc.tl.leiden(source_concat_adata, resolution=leiden_resolution)
        target_concat_adata.obs["leiden"] = (
            source_concat_adata.obs["leiden"]
            .loc[target_concat_adata.obs["source_obs_names"]]
            .values
        )
        target_rna_spatial, target_atac_spatial = _spatial_target_from_source_reference(
            target_concat_adata, source_rna, source_atac, mapping_col="source_obs_names"
        )

    elif model is target_mgate:
        compute_concat_umap(
            target_concat_adata,
            n_neighbors=umap_n_neighbors,
            resolution=umap_resolution,
            deterministic=True,
            random_state=deterministic_seed,
        )
        target_concat_adata.obs["target_obs_names"] = target_concat_adata.obs_names
        sc.tl.ingest(
            source_concat_adata,
            target_concat_adata,
            embedding_method="umap",
            obs="target_obs_names",
            k=1,
        )
        sc.pp.neighbors(target_concat_adata, use_rep="X", n_neighbors=leiden_neighbors)
        sc.tl.leiden(target_concat_adata, resolution=leiden_resolution)
        source_concat_adata.obs["leiden"] = (
            target_concat_adata.obs["leiden"]
            .loc[source_concat_adata.obs["target_obs_names"]]
            .values
        )
        target_rna_spatial, target_atac_spatial = _spatial_target_from_target_reference(
            source_concat_adata,
            target_concat_adata,
            source_rna,
            source_atac,
            mapping_col="target_obs_names",
        )
    else:
        raise ValueError("model must be source_mgate or target_mgate (same object identity).")

    # --- 4) Joint object for plotting ------------------------------------------
    source_target_adata = sc.concat([source_concat_adata, target_concat_adata], axis=0)
    source_target_adata.obs["source_or_target"] = (
        ["source"] * source_concat_adata.n_obs + ["target"] * target_concat_adata.n_obs
    )

    # --- 4b) Per-reference hit counts (must run before joint concat) ----------
    if model is source_mgate:
        count_series = target_concat_adata.obs["source_obs_names"].value_counts().rename("map_count")
    else:
        count_series = source_concat_adata.obs["target_obs_names"].value_counts().rename("map_count")
        
    source_target_adata.obs = source_target_adata.obs.merge(
        count_series,
        left_index=True,
        right_index=True,
        how="left",
    )

    # --- 5) Spatial coordinates (source truth + imputed target) ----------------
    source_target_adata.obsm["spatial"] = np.concatenate(
        [
            np.asarray(source_rna.obsm["spatial"]),
            np.asarray(source_atac.obsm["spatial"]),
            target_rna_spatial,
            target_atac_spatial,
        ],
        axis=0,
    )

    # --- 6) Split + plot spatial (sc.pl.embedding) and UMAP --------------------
    source_adata = source_target_adata[source_target_adata.obs["source_or_target"].eq("source")]
    target_adata = source_target_adata[source_target_adata.obs["source_or_target"].eq("target")]

    _, axs = plt.subplots(2, 2, figsize=(10, 8))
    sc.pl.embedding(
        source_adata,
        basis="spatial",
        color="leiden",
        s=embedding_point_size,
        show=False,
        ax=axs[0, 0],
        legend_loc="none",
    )
    sc.pl.umap(source_adata, color="leiden", ax=axs[0, 1], size=embedding_point_size, show=False)
    sc.pl.embedding(
        target_adata,
        basis="spatial",
        color="leiden",
        s=embedding_point_size,
        show=False,
        ax=axs[1, 0],
        legend_loc="none",
    )
    sc.pl.umap(target_adata, color="leiden", ax=axs[1, 1], size=embedding_point_size, show=False)
    axs[0, 0].set_title("Source spatial")
    axs[0, 1].set_title("Source UMAP")
    axs[1, 0].set_title("Target spatial (imputed from source)")
    axs[1, 1].set_title("Target UMAP")
    plt.tight_layout()
    plt.show()

    return source_target_adata, source_concat_adata, target_concat_adata


#%% ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    #%%
    NOTEBOOK = is_notebook()
    args = parse_args(notebook=NOTEBOOK)
    
    if args.stage2_run_name is not None and args.target_model != "stage2":
        raise ValueError("--stage2-run-name is only valid with --target-model stage2.")

    pathway_embedding_results = None

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
    #bash /home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/scripts/start_mlflow_services.sh all

    #args.run_name = '20260402_153455' #'20260402_153455'
    #args.stage2_run_name = '20260402_153455_stage2_20260402_165006'
    #sqlite_tracking_uri = "sqlite:////home/mcb/users/dmannk/BAKLAVA_base/mlflow_tracking/MultiGATE/mlflow.db"
    #postgres_tracking_uri = "http://127.0.0.1:5000"
    #args.tracking_uri = postgres_tracking_uri
    ## args.target_model = "zero_shot" # for zero-shot co-embedding if stage2 model is not available

    #lsof /home/mcb/users/dmannk/BAKLAVA_base/mlflow_tracking/MultiGATE/mlflow.db
    #curl -i http://127.0.0.1:5000

    tracking_config = setup_mlflow_tracking(
        backend_type=args.mlflow_backend,
        tracking_uri=args.tracking_uri,
        mlflow_base_dir=args.mlflow_base_dir,
        experiment_name=args.experiment_name,
    )
    client = MlflowClient()
    run_id = resolve_run_id_from_name(
        client,
        args.run_name,
        experiment_name=args.experiment_name,
        tracking_config=tracking_config,
    )
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
    legacy_target_subsample_n = int(run_params.get("target_subsample_n", 5000))
    legacy_target_subsample_seed = int(run_params.get("target_subsample_seed", 0))
    if legacy_target_subsample_n <= 0:
        raise ValueError("--target-subsample-n must be a positive integer.")

    split_metadata = None
    with tempfile.TemporaryDirectory() as split_tmpdir:
        split_metadata = load_domain_splits_artifact(client, run_id, split_tmpdir)
    if split_metadata is not None:
        print(
            "Loaded split artifact '{}' (evaluation split: {}). Using source/target eval subsets by default.".format(
                SPLIT_ARTIFACT,
                split_metadata.get("evaluation_split", "val_plus_test"),
            )
        )
    else:
        print(
            "Split artifact '{}' not found. Falling back to legacy target subsampling.".format(
                SPLIT_ARTIFACT
            )
        )

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
            tracking_config=tracking_config,
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
        'stage1_nonspatial': "model_stage1_nonspatial.pth",
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
            nonspatial_source_mgate, _, _, _ = load_mgate(
                client, run_id, source_artifact_map["stage1_nonspatial"], device, tmpdir
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

    #%% ── Source/Target graph preparation ─────────────────────────────────────
    print("\nBuilding source spatial graph...")
    MultiGATE.Cal_Spatial_Net(source_rna, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_rna)
    MultiGATE.Cal_Spatial_Net(source_atac, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_atac)
    source_atac.uns["gene_peak_Net"] = gp_net.copy()

    print("Building target graph (type: '{}')...".format(spatial_graph_type))
    target_rna, target_atac = prepare_target_for_spatial_graph_type(
        target_rna=target_rna,
        target_atac=target_atac,
        source_rna=source_rna,
        source_atac=source_atac,
        spatial_graph_type=spatial_graph_type,
        gtf_path=gtf_path,
    )
    '''
    if split_metadata is not None:
        source_rna, source_atac = subset_domain_to_saved_split(
            source_rna,
            source_atac,
            split_metadata=split_metadata,
            domain_name="source",
            split_name="eval",
        )
        target_rna, target_atac = subset_domain_to_saved_split(
            target_rna,
            target_atac,
            split_metadata=split_metadata,
            domain_name="target",
            split_name="eval",
        )
        print(
            "Applied split artifact eval subsets: source={} cells, target={} cells".format(
                source_rna.n_obs,
                target_rna.n_obs,
            )
        )
    else:
        print("\nPairing and subsampling target cells (legacy fallback)...")
        target_rna, target_atac = pair_and_subsample_target(
            target_rna,
            target_atac,
            subsample_n=legacy_target_subsample_n,
            seed=legacy_target_subsample_seed,
        )
        print("  {} cells after pairing/subsampling".format(target_rna.n_obs))
    '''
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

    ## nonspatial source inference
    nonspatial_source_rna_emb, nonspatial_source_atac_emb = run_inference(
        nonspatial_source_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, nonspatial_source_rna_emb, nonspatial_source_atac_emb, key_added="MultiGATE_nonspatial")
    print("  Nonspatial source embeddings: shape {}".format(nonspatial_source_rna_emb.shape))

    # Plot source/target concat UMAPs with the same helper as training script.
    teacher_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_teacher")
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE")
    nonspatial_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_nonspatial")

    ## set leiden resolution
    leiden_resolution = 0.5
    leiden_neighbors = 10

    ## compute UMAPs
    compute_concat_umap(
        teacher_source_concat_adata,
        n_neighbors=leiden_neighbors,
        resolution=leiden_resolution,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        source_concat_adata,
        n_neighbors=leiden_neighbors,
        resolution=leiden_resolution,
        deterministic=True,
        random_state=deterministic_seed,
    )
    compute_concat_umap(
        nonspatial_source_concat_adata,
        n_neighbors=leiden_neighbors,
        resolution=leiden_resolution,
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

    # plot nonspatial source UMAPs
    nonspatial_source_umap_colors = ['modality', 'leiden', 'RNA_clusters']
    fig, axs = plt.subplots(1, len(nonspatial_source_umap_colors), figsize=(18, 5))
    for i, color in enumerate(nonspatial_source_umap_colors):
        sc.pl.umap(nonspatial_source_concat_adata, color=color, ncols=3, wspace=0.2, size=25, ax=axs[i], show=False)
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

    #%% Inference, all same model (except nonspatial source inference)

    model = source_mgate

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

    ## nonspatial source inference
    nonspatial_source_rna_emb, nonspatial_source_atac_emb = run_inference(
        nonspatial_source_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, nonspatial_source_rna_emb, nonspatial_source_atac_emb, key_added="MultiGATE_nonspatial")
    print("  Nonspatial source embeddings: shape {}".format(nonspatial_source_rna_emb.shape))

    #%% OT from target to source
    import ot
    import gc

    X = torch.tensor(target_rna.obsm['MultiGATE'], device='cuda')
    Y = torch.tensor(source_rna.obsm['MultiGATE'], device='cuda')
    res = ot.solve_sample(X, Y, metric='euclidean', reg=0.001)
    X_to_Y = res.plan @ Y * len(X)
    target_rna.obsm['MultiGATE_source_aligned'] = X_to_Y.detach().cpu().numpy()
    del X, Y, res, X_to_Y

    X = torch.tensor(target_atac.obsm['MultiGATE'], device='cuda')
    Y = torch.tensor(source_atac.obsm['MultiGATE'], device='cuda')
    res = ot.solve_sample(X, Y, metric='euclidean', reg=0.001)
    X_to_Y = res.plan @ Y * len(X)
    target_atac.obsm['MultiGATE_source_aligned'] = X_to_Y.detach().cpu().numpy()
    del X, Y, res, X_to_Y

    torch.cuda.empty_cache()
    gc.collect()

    target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key="MultiGATE_source_aligned")

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

    ## concatenate source and target data
    source_rna.obsm['MultiGATE_source_aligned'] = source_rna.obsm['MultiGATE']
    source_atac.obsm['MultiGATE_source_aligned'] = source_atac.obsm['MultiGATE']
    source_target_rna = sc.concat([source_rna, target_rna], axis=0)
    source_target_atac = sc.concat([source_atac, target_atac], axis=0)
    source_target_adata = build_concat_adata_for_umap(source_target_rna, source_target_atac, embedding_key="MultiGATE_source_aligned")

    source_or_target = np.concatenate([
        np.full(source_rna.n_obs, 'source'),
        np.full(target_rna.n_obs, 'target'),
        np.full(source_atac.n_obs, 'source'),
        np.full(target_atac.n_obs, 'target'),
        ])
    source_target_adata.obs['source_or_target'] = source_or_target
    
    compute_concat_umap(
        source_target_adata,
        n_neighbors=10,
        resolution=0.5,
        deterministic=True,
        random_state=deterministic_seed,
    )

    sc.pl.umap(source_target_adata, color=['modality', 'leiden', 'source_or_target'], ncols=3, wspace=0.2, size=25)


    #%% Analysis on concatenated data

    ## concatenate modalities
    teacher_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_teacher")
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE")
    target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key="MultiGATE")
    nonspatial_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_nonspatial")

    ## add spatial coordinates to concatenated source data
    spatial_coords = np.concatenate([source_rna.obsm['spatial'], source_atac.obsm['spatial']], axis=0)
    teacher_source_concat_adata.obsm['spatial'] = spatial_coords
    source_concat_adata.obsm['spatial'] = spatial_coords
    nonspatial_source_concat_adata.obsm['spatial'] = spatial_coords
    #target_concat_adata.obsm['spatial'] = np.concatenate([target_rna.obsm['spatial'], target_atac.obsm['spatial']], axis=0)

    source_target_adata, source_concat_adata, target_concat_adata = run_alignment_and_spatial_plot(
        model,
        source_mgate,
        target_mgate,
        source_rna,
        source_atac,
        source_concat_adata,
        target_concat_adata,
        deterministic_seed,
    )

    ## transfer spatial coordinates to target data
    target_rna.obsm['spatial'] = source_target_adata[
        source_target_adata.obs['modality'].eq('rna') &
        source_target_adata.obs['source_or_target'].eq('target')
    ].obsm['spatial'].copy()
 
    target_concat_adata.obs["arc_gex_kmeans_5_clusters_Cluster"] = target_concat_adata.obs["arc_gex_kmeans_5_clusters_Cluster"].astype("category")

    ## ingest source embeddings into target data
    if model is source_mgate:
        sc.tl.ingest(target_concat_adata, source_concat_adata, embedding_method='umap', obs='RNA_clusters')
        #sc.tl.ingest(teacher_source_concat_adata, source_concat_adata, embedding_method='umap', obs='RNA_clusters')
        ## confirm that the ingested representations are the same as the original embeddings
        #assert (teacher_source_concat_adata.obsm['rep'] == teacher_source_concat_adata.X).all()
        assert (target_concat_adata.obsm['rep'] == target_concat_adata.X).all()
    elif model is target_mgate:
        rna_target_obs_names = source_concat_adata.obs.loc[source_concat_adata.obs['modality'].eq('rna'), 'target_obs_names']
        atac_target_obs_names = source_concat_adata.obs.loc[source_concat_adata.obs['modality'].eq('atac'), 'target_obs_names']
        target_concat_adata.obs.loc[rna_target_obs_names.values, 'RNA_clusters'] = source_concat_adata.obs.loc[rna_target_obs_names.index, 'RNA_clusters'].values
        target_concat_adata.obs.loc[atac_target_obs_names.values, 'RNA_clusters'] = source_concat_adata.obs.loc[atac_target_obs_names.index, 'RNA_clusters'].values
        ## confirm that the ingested representations are the same as the original embeddings
        #assert (target_concat_adata.obsm['rep'] == target_concat_adata.X).all()
        assert (source_concat_adata.obsm['rep'] == source_concat_adata.X).all()

    # assign RNA_clusters to target_rna specifically
    ingested_rna_clusters = target_concat_adata.obs.loc[target_concat_adata.obs['modality'].eq('rna'), 'RNA_clusters']
    ingested_rna_clusters.index = ingested_rna_clusters.index.str.replace('_rna', '')
    target_rna.obs['RNA_clusters'] = ingested_rna_clusters

    combined_gp_dict = None
    if not getattr(args, "no_combined_gp_dict", False):
        print("\nLoading NicheCompass combined_gp_dict...")
        combined_gp_dict = load_nichecompass_combined_gp_dict_mouse(
            load_from_disk=True,
            verbose=True,
        )

    pathway_embedding_results = run_co_embed_pathway_embedding_analysis(
        combined_gp_dict=combined_gp_dict,
        source_rna=source_rna,
        target_rna=target_rna,
        embedding_key="MultiGATE",
        include_teacher=True,
        cluster_obs_key="RNA_clusters",
    )

    try:
        #sc.pl.umap(teacher_source_concat_adata, color=['modality', 'leiden', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
        #plt.tight_layout(); plt.show()
        sc.pl.umap(source_concat_adata, color=['modality', 'leiden', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
        plt.tight_layout(); plt.show()
        sc.pl.umap(target_concat_adata, color=['modality', 'leiden', 'arc_gex_kmeans_5_clusters_Cluster'], ncols=3, wspace=0.2, size=25)
        plt.tight_layout(); plt.show()
    except:
        #sc.pl.umap(teacher_source_concat_adata, color=['modality', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
        #plt.tight_layout(); plt.show()
        sc.pl.umap(source_concat_adata, color=['modality', 'RNA_clusters'], ncols=3, wspace=0.2, size=25)
        plt.tight_layout(); plt.show()
        sc.pl.umap(target_concat_adata, color=['modality', 'arc_gex_kmeans_5_clusters_Cluster'], ncols=3, wspace=0.2, size=25)
        plt.tight_layout(); plt.show()

    #%% Save adata for FASTopic analysis

    ## write to disk
    # for keys in (source_rna.obsm.keys(), source_atac.obsm.keys(), target_rna.obsm.keys(), target_atac.obsm.keys()):
    #     assert 'MultiGATE_source_aligned' in keys, f"MultiGATE_source_aligned not found in {keys}"
    #source_rna.write_h5ad(os.path.join(base_path, "source_rna_aligned_with_latents.h5ad"))
    #source_atac.write_h5ad(os.path.join(base_path, "source_atac_aligned_with_latents.h5ad"))
    #target_rna.write_h5ad(os.path.join(base_path, "target_rna_aligned_with_latents.h5ad"))
    #target_atac.write_h5ad(os.path.join(base_path, "target_atac_aligned_with_latents.h5ad"))

    ## load from disk, after FASTopic analysis
    source_rna = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned_with_fastopic.h5ad"))
    source_atac = sc.read_h5ad(os.path.join(base_path, "source_atac_aligned_with_fastopic.h5ad"))
    target_rna = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned_with_fastopic.h5ad"))
    target_atac = sc.read_h5ad(os.path.join(base_path, "target_atac_aligned_with_fastopic.h5ad"))

    ## extract fastopic results
    def _extract_fastopic_topic_gene_weights(adata, net):
        """Build the topic-by-gene weight matrix from FASTopic obsm/uns/varm keys.

        Returns
        -------
        topic_gene_weight_mat : pd.DataFrame, shape (n_topics, n_genes)
            Topics ordered by descending global weight, cleaned of inf/NaN columns.
        sorted_topics : np.ndarray
            Topic indices in descending global-weight order.
        net : pd.DataFrame
            Hallmark net filtered to genes present in topic_gene_weight_mat.
        """
        topic_global_weights = adata.uns['fastopic_global_weights'].copy()
        topic_by_genes = adata.varm['fastopic_genes_topic_weights'].T.copy()
        topic_by_genes_df = pd.DataFrame(
            topic_by_genes,
            columns=adata.var_names,
            index=[f'topic_{i}' for i in range(len(topic_by_genes))],
        )
        sorted_topics = np.flip(np.argsort(topic_global_weights))

        plt.figure(figsize=(12, 5))
        plt.plot(topic_by_genes_df.index[sorted_topics], topic_global_weights[sorted_topics], marker='o')
        plt.xticks(rotation=45); plt.tight_layout(); plt.show()

        topic_gene_weight_mat = topic_by_genes_df.loc[[f"topic_{i}" for i in sorted_topics]].copy()
        topic_gene_weight_mat = topic_gene_weight_mat.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")

        net_filtered = net[net["target"].isin(topic_gene_weight_mat.columns)].copy()
        if net_filtered.empty:
            raise ValueError("No Hallmark mouse genes overlap with topic-gene weight columns.")

        return topic_gene_weight_mat, sorted_topics, net_filtered

    def run_topic_gsea(topic_gene_weight_mat, net, tmin=3, times=1000, seed=42, padj_thresh=0.05):
        """Run decoupler GSEA over a topic-by-gene weight matrix.

        Each row (topic) is treated as an independent ranked gene list; topic-gene
        weights serve as the ranking statistic, not expression counts.

        Returns
        -------
        dict with keys:
            gsea_scores, gsea_padj, gsea_long, gsea_long_filt, top_terms_per_topic
        """
        import decoupler as dc

        gsea_scores, gsea_padj = dc.mt.gsea(
            topic_gene_weight_mat,
            net,
            tmin=tmin,
            times=times,
            seed=seed,
            verbose=True,
        )
        gsea_long = (
            gsea_scores.stack()
            .rename("nes")
            .to_frame()
            .join(gsea_padj.stack().rename("padj"))
            .reset_index()
            .rename(columns={"level_0": "topic", "level_1": "pathway"})
            .sort_values(["topic", "padj", "nes"], ascending=[True, True, False])
        )
        top_terms_per_topic = gsea_long.groupby("topic", group_keys=False).head(10)
        print(top_terms_per_topic)

        significant_pathways_per_topic = (
            gsea_padj.apply(lambda row: row[row.le(padj_thresh)].index.tolist(), axis=1)
            .to_dict()
        )
        significant_pathways_per_topic = {
            k: v for k, v in significant_pathways_per_topic.items() if len(v) > 0
        }
        for topic_to_plot, pathways in significant_pathways_per_topic.items():
            leading_edge_df = (
                topic_gene_weight_mat.loc[topic_to_plot]
                .rename("topic_gene_weight")
                .to_frame()
            )
            for pathway in pathways:
                leading_edge_fig, _ = dc.pl.leading_edge(
                    leading_edge_df,
                    net=net,
                    stat="topic_gene_weight",
                    name=pathway,
                    return_fig=True,
                )
                nes = gsea_scores.loc[topic_to_plot, pathway]
                padj = gsea_padj.loc[topic_to_plot, pathway]
                for ax in leading_edge_fig.axes:
                    ax.set_title("")
                leading_edge_fig.suptitle(
                    f"{topic_to_plot} | {pathway}\nNES={nes:.2f}, adj. p={padj:.2e}",
                    y=1.02,
                )
                leading_edge_fig.show()

        return dict(
            gsea_scores=gsea_scores,
            gsea_padj=gsea_padj,
            gsea_long=gsea_long,
            gsea_long_filt=gsea_long[gsea_long["padj"].le(padj_thresh)],
            top_terms_per_topic=top_terms_per_topic,
        )

    def run_topic_ora(topic_gene_weight_mat, net, ora_tmin=3, padj_thresh=0.05):
        """Run hypergeometric ORA for each topic using a triangle-threshold gene cutoff.

        Top genes per topic are selected by a global triangle threshold applied to the
        flattened topic-gene weight distribution.  Background is the scored gene universe
        (columns of topic_gene_weight_mat).

        Returns
        -------
        dict with keys:
            knee_value, top_genes_per_topic, ora_long, ora_long_filt, top_ora_terms_per_topic
        """
        from skimage.filters import threshold_triangle
        from statsmodels.stats.multitest import multipletests
        import scipy.stats

        global_topic_gene_weights = (
            topic_gene_weight_mat.melt()["value"]
            .dropna()
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        knee_value = threshold_triangle(global_topic_gene_weights.values)
        knee_index = int(np.argmin(np.abs(global_topic_gene_weights.values - knee_value)))

        plt.figure(figsize=(3, 3))
        global_topic_gene_weights.plot()
        plt.scatter(knee_index, knee_value, color='red', label=f"knee={knee_value:.2e}")
        plt.legend(); plt.tight_layout(); plt.show()
        plt.close()

        top_genes_per_topic = {
            topic: (
                topic_gene_weight_mat.loc[topic][topic_gene_weight_mat.loc[topic].ge(knee_value)]
                .sort_values(ascending=False)
                .index.astype(str)
                .tolist()
            )
            for topic in topic_gene_weight_mat.index
        }

        ora_net = net[net["target"].isin(topic_gene_weight_mat.columns)].copy()
        ora_net = ora_net[
            ora_net["source"].isin(
                ora_net.groupby("source")["target"].nunique()[lambda s: s.ge(ora_tmin)].index
            )
        ].copy()

        background_genes = set(topic_gene_weight_mat.columns.astype(str))
        background_n = len(background_genes)
        ora_results = []

        for topic, top_genes in top_genes_per_topic.items():
            top_gene_set = set(top_genes)
            query_n = len(top_gene_set)
            for pathway, pathway_df in ora_net.groupby("source"):
                pathway_genes = set(pathway_df["target"].astype(str)) & background_genes
                pathway_n = len(pathway_genes)
                if pathway_n < ora_tmin:
                    continue
                overlap_genes = sorted(top_gene_set & pathway_genes)
                overlap_n = len(overlap_genes)
                pval = scipy.stats.hypergeom.sf(overlap_n - 1, background_n, pathway_n, query_n)
                ora_results.append(dict(
                    topic=topic,
                    pathway=pathway,
                    overlap_n=overlap_n,
                    query_n=query_n,
                    pathway_n=pathway_n,
                    background_n=background_n,
                    overlap_genes=",".join(overlap_genes),
                    pval=pval,
                ))

        ora_long = pd.DataFrame(ora_results)
        if ora_long.empty:
            print("No ORA results produced. Consider lowering ora_tmin.")
            return dict(
                knee_value=knee_value,
                top_genes_per_topic=top_genes_per_topic,
                ora_long=ora_long,
                ora_long_filt=ora_long,
                top_ora_terms_per_topic=ora_long,
            )

        ora_long["padj"] = multipletests(ora_long["pval"], method="fdr_bh")[1]
        ora_long = ora_long.sort_values(["topic", "padj", "overlap_n"], ascending=[True, True, False])
        ora_long_filt = ora_long[ora_long["padj"].le(padj_thresh)]
        top_ora_terms_per_topic = ora_long.groupby("topic", group_keys=False).head(10)
        print(top_ora_terms_per_topic)

        return dict(
            knee_value=knee_value,
            top_genes_per_topic=top_genes_per_topic,
            ora_long=ora_long,
            ora_long_filt=ora_long_filt,
            top_ora_terms_per_topic=top_ora_terms_per_topic,
        )

    def run_gsea_ora_overlap(gsea_long_filt, ora_long_filt):
        """Compute per-topic Jaccard similarity between significant GSEA and ORA pathways.

        Returns
        -------
        gsea_ora_jaccard : pd.Series
            Per-topic Jaccard index, sorted descending.
        """
        gsea_hits_per_topic = (
            gsea_long_filt.groupby("topic", group_keys=False)["pathway"].apply(np.unique)
        )
        ora_hits_per_topic = (
            ora_long_filt.groupby("topic", group_keys=False)["pathway"].apply(np.unique)
        )
        topics_union = set(gsea_hits_per_topic.index) | set(ora_hits_per_topic.index)
        gsea_ora_jaccard = {}
        for topic in topics_union:
            gsea_paths = set(gsea_hits_per_topic.get(topic, []))
            ora_paths = set(ora_hits_per_topic.get(topic, []))
            union_n = len(gsea_paths | ora_paths)
            gsea_ora_jaccard[topic] = (
                len(gsea_paths & ora_paths) / union_n if union_n > 0 else 0.0
            )
        gsea_ora_jaccard = pd.Series(gsea_ora_jaccard).sort_values(ascending=False)
        print("GSEA vs. ORA Jaccard similarity:")
        print(gsea_ora_jaccard)
        return gsea_ora_jaccard

    ## build shared Hallmark mouse net once
    import decoupler as dc

    hallmark_human = dc.op.hallmark(organism="human")
    map_path = os.path.join(os.environ["DATAPATH"], "gene_annotations", "human_mouse_gene_orthologs.csv")
    map_df = (
        pd.read_csv(map_path)
        .rename(columns={"Gene name": "target_human", "Mouse gene name": "target"})[
            ["target_human", "target"]
        ]
        .dropna()
        .drop_duplicates()
    )
    hallmark_mouse_net = (
        hallmark_human.rename(columns={"target": "target_human"})
        .merge(map_df, on="target_human", how="inner")[["source", "target"]]
        .drop_duplicates()
    )

    ## run for source_rna
    source_topic_mat, source_sorted_topics, source_net = _extract_fastopic_topic_gene_weights(
        source_rna, hallmark_mouse_net,
    )
    source_gsea = run_topic_gsea(source_topic_mat, source_net)
    source_ora  = run_topic_ora(source_topic_mat, source_net)

    ## run for target_rna
    target_topic_mat, target_sorted_topics, target_net = _extract_fastopic_topic_gene_weights(
        target_rna, hallmark_mouse_net,
    )
    target_gsea = run_topic_gsea(target_topic_mat, target_net)
    target_ora  = run_topic_ora(target_topic_mat, target_net)

    ## compute jaccard similarity between GSEA and ORA results
    source_jaccard = run_gsea_ora_overlap(source_gsea["gsea_long_filt"], source_ora["ora_long_filt"])
    target_jaccard = run_gsea_ora_overlap(target_gsea["gsea_long_filt"], target_ora["ora_long_filt"])

    gsea_jaccard = run_gsea_ora_overlap(source_gsea["gsea_long_filt"], target_gsea["gsea_long_filt"])
    ora_jaccard = run_gsea_ora_overlap(source_ora["ora_long_filt"], target_ora["ora_long_filt"])


    #%% analysis of linear decoder
    from sklearn.metrics.pairwise import euclidean_distances
    from scipy.special import softmax
    from post_hoc_utils import topic_betas_hallmark_gsea_mouse

    assert teacher_source_mgate.linear_etm_decoder

    ## extract decoder parameters
    alpha = teacher_source_mgate.alpha.detach().cpu().numpy()
    rho_rna = teacher_source_mgate.rho_rna.detach().cpu().numpy()
    rho_atac = teacher_source_mgate.rho_atac.detach().cpu().numpy()

    rho_rna_mask = teacher_source_mgate.rho_rna_mask.detach().cpu().numpy()
    rho_atac_mask = teacher_source_mgate.rho_atac_mask.detach().cpu().numpy()
    rho_rna_overlap = ((np.abs(rho_rna) == np.abs(rho_rna).max(1, keepdims=True)) * rho_rna_mask).any(1).mean()
    rho_atac_overlap = ((np.abs(rho_atac) == np.abs(rho_atac).max(1, keepdims=True)) * rho_atac_mask).any(1).mean()
    print(f"RNA overlap: {rho_rna_overlap}, ATAC overlap: {rho_atac_overlap}")

    rho_rna_mask_gain = (np.abs(rho_rna) * rho_rna_mask).sum(1) / rho_rna_mask.sum(1) / np.abs(rho_rna).mean(1)
    rho_atac_mask_gain = (np.abs(rho_atac) * rho_atac_mask).sum(1) / rho_atac_mask.sum(1) / np.abs(rho_atac).mean(1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    ax[0].hist(rho_rna_mask_gain, bins=50)
    ax[1].hist(rho_atac_mask_gain, bins=50)
    plt.tight_layout(); plt.show()

    topk=25
    topk_genes = []
    for topic in range(alpha.shape[0]):
        topk_genes_topic = pd.Series(alpha[topic]).nlargest(topk).index
        topk_genes.append(topk_genes_topic)
    topk_genes = np.concatenate(topk_genes)
    top_alpha = alpha[:,topk_genes]
    plt.figure(figsize=(10, 10))
    plt.matshow(top_alpha, aspect='auto'); plt.colorbar()
    plt.tight_layout(); plt.show()

    pathway_names = teacher_source_mgate.pathway_names
    source_pathway_names = [pw for pw in pathway_names if pw.endswith('source')]
    target_pathway_names = [pw for pw in pathway_names if pw.endswith('target')]

    alpha_df = pd.DataFrame(alpha, index=[f'topic_{i}' for i in range(30)], columns=pathway_names)
    pd.concat([
        alpha_df.abs().max(1),
        alpha_df.abs().median(1)
    ], axis=1).sort_values(0).plot(kind='bar')


    # alpha = F.normalize(alpha, dim=1)
    # rho_rna = F.normalize(rho_rna, dim=1)
    # rho_atac = F.normalize(rho_atac, dim=1)
    # topic_var = teacher_source_mgate.topic_var.detach().cpu().numpy()

    beta_rna = alpha @ rho_rna
    beta_atac = alpha @ rho_atac

    ## derive theta from delta embeddings
    source_theta = softmax(np.concatenate([source_rna_emb, source_atac_emb], axis=0), axis=1)
    source_theta_df = pd.DataFrame(source_theta, index=np.concatenate([source_rna.obs_names, source_atac.obs_names]))

    target_theta = softmax(np.concatenate([target_rna_emb, target_atac_emb], axis=0), axis=1)
    target_theta_df = pd.DataFrame(target_theta, index=np.concatenate([target_rna.obs_names, target_atac.obs_names]))

    ## cluster delta by leiden clusters, then apply softmax to get theta
    source_clust_by_topic_theta = (
        pd.DataFrame(source_rna_emb, index=source_rna.obs_names)
        .assign(RNA_leiden=source_concat_adata.obs.loc[
            source_concat_adata.obs['modality'].eq('rna'),
            'leiden'
            ].values)
        .groupby('RNA_leiden')
        .mean()
        .apply(softmax, axis=1, result_type='expand')
    )

    target_clust_by_topic_theta = (
        pd.DataFrame(target_rna_emb, index=target_rna.obs_names)
        .assign(RNA_leiden=target_concat_adata.obs.loc[
            target_concat_adata.obs['modality'].eq('rna'),
            'leiden'
            ].values)
        .groupby('RNA_leiden')
        .mean()
        .apply(softmax, axis=1, result_type='expand')
    )
    
    ## get top topics for each cluster and plot staircase heatmap
    topk = 5
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    topk_topics = []
    topics = source_clust_by_topic_theta.columns.values
    clusters = source_clust_by_topic_theta.max(1).sort_values(ascending=False).index
    for cluster in clusters:
        topk_topics_cluster = source_clust_by_topic_theta.loc[cluster].nlargest(topk).index
        topk_topics_cluster = topk_topics_cluster[topk_topics_cluster.isin(topics)]
        topics = topics[~np.isin(topics, topk_topics_cluster)]
        topk_topics.append(topk_topics_cluster)
    source_topk_topics = np.concatenate(topk_topics + [pd.Index(topics)]) # could also just keep topk_topics
    source_clust_by_topic_theta = source_clust_by_topic_theta.loc[clusters, source_topk_topics]
    sns.heatmap(source_clust_by_topic_theta.T, cmap='viridis', ax=axs[0])

    topk_topics = []
    topics = target_clust_by_topic_theta.columns.values
    clusters = target_clust_by_topic_theta.max(1).sort_values(ascending=False).index
    for cluster in clusters:
        topk_topics_cluster = target_clust_by_topic_theta.loc[cluster].nlargest(topk).index
        topk_topics_cluster = topk_topics_cluster[topk_topics_cluster.isin(topics)]
        topics = topics[~np.isin(topics, topk_topics_cluster)]
        topk_topics.append(topk_topics_cluster)
    target_topk_topics = np.concatenate(topk_topics + [pd.Index(topics)])
    target_clust_by_topic_theta = target_clust_by_topic_theta.loc[clusters, target_topk_topics]
    sns.heatmap(target_clust_by_topic_theta.T, cmap='viridis', ax=axs[1])
    plt.tight_layout(); plt.show()

    ## add alpha embeddings to source and target data
    source_alpha_embs = source_theta @ alpha
    target_alpha_embs = target_theta @ alpha
    source_target_adata.obsm['alpha_embs'] = np.concatenate([source_alpha_embs, target_alpha_embs], axis=0)

    import liana as li
    
    pathways_adata = sc.AnnData(
        source_alpha_embs,
        obs=source_target_adata[source_target_adata.obs['source_or_target'].eq('source')].obs,
        var=pd.DataFrame(
            data=np.vstack([
                [pw.split('__')[0] for pw in pathway_names],
                ['sender' if pw.endswith('source') else 'receiver' for pw in pathway_names],
            ]).T,
            index=pathway_names,
            columns=['basename', 'sender_or_receiver']),
        obsm={'spatial': source_target_adata[source_target_adata.obs['source_or_target'].eq('source')].obsm['spatial']},
        )
    pathways_adata.var['paired'] = pathways_adata.var['basename'].isin(pathways_adata.var['basename'].value_counts().index[pathways_adata.var['basename'].value_counts().eq(2)])

    li.ut.spatial_neighbors(pathways_adata, bandwidth=1000, set_diag=False)
    paired_pathways_adata = pathways_adata[:,pathways_adata.var['paired']].copy()
    sender_pathways_adata = paired_pathways_adata[:,paired_pathways_adata.var['sender_or_receiver'].eq('sender')]
    receiver_pathways_adata = paired_pathways_adata[:,paired_pathways_adata.var['sender_or_receiver'].eq('receiver')]
    assert np.all(sender_pathways_adata.var['basename'].values == receiver_pathways_adata.var['basename'].values)

    X = pathways_adata.X.copy()
    W = pathways_adata.obsp['spatial_connectivities']
    plt.matshow(X.T @ W @ X, aspect='auto'); plt.colorbar()
    plt.tight_layout(); plt.show()

    S = sender_pathways_adata.X.copy()
    R = receiver_pathways_adata.X.copy()
    W = pathways_adata.obsp['spatial_connectivities']
    plt.scatter(S.sum(axis=0), R.sum(axis=0)); plt.colorbar()

    #S = softmax(S, axis=0)
    #R = softmax(R, axis=0)

    M = S.T @ W @ R
    plt.matshow(M); plt.colorbar()

    sc.pp.pca(pathways_adata, n_comps=50)
    sc.external.pp.bbknn(pathways_adata, batch_key='source_or_target', use_rep='X_pca', neighbors_within_batch=10)
    sc.tl.umap(pathways_adata, min_dist=0.3)
    sc.pl.umap(pathways_adata, color=['source_or_target'], ncols=3, wspace=0.2, size=25)
    plt.tight_layout(); plt.show()

    ## compute mean distance between source and target for each cluster
    source_adata = source_target_adata[source_target_adata.obs['source_or_target'].eq('source')]
    target_adata = source_target_adata[source_target_adata.obs['source_or_target'].eq('target')]
    mean_dists = pd.DataFrame(index=source_adata.obs['leiden'].cat.categories, columns=['multigate', 'alpha'])
    for cluster in source_adata.obs['leiden'].cat.categories:
        source_dat = source_adata[source_adata.obs['leiden'].eq(cluster)]
        target_dat = target_adata[target_adata.obs['leiden'].eq(cluster)]
        mean_dists.loc[cluster, 'multigate'] = np.mean(euclidean_distances(source_dat.X, target_dat.X)) # could replace with wasserstein distance
        mean_dists.loc[cluster, 'alpha'] = np.mean(euclidean_distances(source_dat.obsm['alpha_embs'], target_dat.obsm['alpha_embs']))
    mean_dists = mean_dists.sort_values(by='multigate', ascending=True)
    mean_dists.loc['all', 'multigate'] = np.mean(euclidean_distances(source_adata.X, target_adata.X))
    mean_dists.loc['all', 'alpha'] = np.mean(euclidean_distances(source_adata.obsm['alpha_embs'], target_adata.obsm['alpha_embs']))
    print(mean_dists)

    ## get top active topics and perform GSEA
    top_active_cluster = mean_dists.index[0] #'5'
    top_active_topics = source_clust_by_topic_theta.loc[top_active_cluster].nlargest(topk).index
    top_active_gsea = topic_betas_hallmark_gsea_mouse(
        beta_rna=beta_rna,
        gene_names=np.asarray(source_rna.var_names).astype(str),
        topic_indices=top_active_topics,
        datapath=os.environ["DATAPATH"],
    )
    top_active_genes = top_active_gsea["top_active_genes"]
    top_active_rna_betas = top_active_gsea["ranked_gene_indices"]
    print("\nHallmark GSEA (mouse) — top terms per active topic:")
    print(
        top_active_gsea["top_terms_per_row"].loc[
            top_active_gsea["top_terms_per_row"]['padj'].le(0.05)
        ]
    )

    # Categorical.map(dict) looks up category values with their native dtype; string
    # keys in lut won't match int categories and yield NaN floats mixed with RGB
    # tuples, which breaks pandas' Index reconstruction (TypeError).
    cluster_key = source_theta_df.obs['RNA_clusters'].astype(str)
    unique_clusters = cluster_key.unique()
    network_pal = sns.husl_palette(len(unique_clusters), s=0.45)
    lut = dict(zip(unique_clusters, network_pal))
    row_colors = cluster_key.map(lut)

    cg = sns.clustermap(source_theta_df, cmap='viridis', row_colors=row_colors)
    cg.ax_heatmap.set_yticklabels([])

    alpha = teacher_source_mgate.alpha.detach().cpu().numpy()
    rho_rna = teacher_source_mgate.rho_rna.detach().cpu().numpy()
    rho_atac = teacher_source_mgate.rho_atac.detach().cpu().numpy()
    rho_rna_mask = teacher_source_mgate.rho_rna_mask.detach().cpu().numpy()
    rho_atac_mask = teacher_source_mgate.rho_atac_mask.detach().cpu().numpy()

    beta_rna = alpha @ rho_rna
    beta_atac = alpha @ rho_atac

    ## topic-topic correlation
    topic_topic_corr = np.corrcoef(alpha)
    topic_topic_corr[np.eye(topic_topic_corr.shape[0]) == 1] = 0
    plt.figure(figsize=(8, 6))
    sns.heatmap(topic_topic_corr, cmap='viridis')
    plt.tight_layout()
    plt.show()

    ## topic norms
    topic_norm = np.linalg.norm(alpha, axis=1)
    beta_rna_norm = np.linalg.norm(beta_rna, axis=1)
    beta_atac_norm = np.linalg.norm(beta_atac, axis=1)
    fig, ax = plt.subplots(3, 1, figsize=(6, 8))
    ax[0].bar(np.arange(alpha.shape[0]), topic_norm)
    ax[1].bar(np.arange(beta_rna.shape[0]), beta_rna_norm)
    ax[2].bar(np.arange(beta_atac.shape[0]), beta_atac_norm)
    ax[0].set_title("Topic norm")
    ax[1].set_title("Beta RNA norm")
    ax[2].set_title("Beta ATAC norm")
    plt.tight_layout()
    plt.show()

    ## topic & feature co-embedding
    alpha_norm = alpha / np.linalg.norm(alpha, axis=1, keepdims=True)
    rho_rna_norm = rho_rna / np.linalg.norm(rho_rna, axis=1, keepdims=True)
    rho_atac_norm = rho_atac / np.linalg.norm(rho_atac, axis=1, keepdims=True)

    alpha_rho_rna_adata = sc.AnnData(
        np.concatenate([rho_rna_norm.T, alpha_norm], axis=0),
    )
    alpha_rho_rna_adata.obs_names = np.concatenate([[f'gp_{gp}' for gp in range(rho_rna.shape[1])], [f'topic_{topic}' for topic in range(alpha.shape[0])]])
    alpha_rho_rna_adata.obs['gene_or_topic'] = np.concatenate([['gene'] * rho_rna.shape[1], ['topic'] * alpha.shape[0]])
    alpha_rho_rna_adata.obs['max_abs_beta'] = list(np.abs(beta_rna).argmax(0)) + list(np.arange(alpha.shape[0]))

    sc.pp.pca(alpha_rho_rna_adata, n_comps=100)
    sc.pp.neighbors(alpha_rho_rna_adata, use_rep='X_pca', n_neighbors=30)
    #sc.external.pp.bbknn(alpha_rho_rna_adata, batch_key='gene_or_topic', neighbors_within_batch=30)
    sc.tl.umap(alpha_rho_rna_adata, min_dist=0.3)

    topic_gene_dists = euclidean_distances(
        alpha_rho_rna_adata.obsm['X_pca'],
    )[:rho_rna.shape[1], -alpha.shape[0]:].argmin(axis=1)
    alpha_rho_rna_adata.obs['topic_gene_dist'] = list(topic_gene_dists) + list(np.arange(alpha.shape[0]))

    # 1. Extract coordinates and groups into a lightweight dataframe
    df = pd.DataFrame(alpha_rho_rna_adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'], index=alpha_rho_rna_adata.obs_names)
    df = df.join(alpha_rho_rna_adata.obs[['gene_or_topic', 'max_abs_beta']])

    # 2. Plot using seaborn, adding labels for topics
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(
        data=df, 
        x='UMAP1', 
        y='UMAP2', 
        hue='gene_or_topic',   # Colors by group
        style='gene_or_topic', # Assigns different markers by group
        palette='Set2',
        s=list([5] * rho_rna.shape[1]) + list([60] * alpha.shape[0]),  # Marker size
        edgecolor='none'
    )

    # Add text labels for topics
    topic_indices = df[df['gene_or_topic'] == 'topic'].index
    topic_labels = [f'topic_{i}' for i in range(alpha.shape[0])]
    for idx, label in zip(topic_indices, topic_labels):
        plt.annotate(
            label,
            (df.loc[idx, 'UMAP1'], df.loc[idx, 'UMAP2']),
            textcoords="offset points",
            xytext=(0,5),
            ha='center',
            fontsize=8,
            color='black',
            alpha=0.8
        )

    # 3. Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    sc.pl.umap(
        alpha_rho_rna_adata,
        color=['gene_or_topic', 'topic_gene_dist', 'max_abs_beta'],
        color_map='Set2',
        ncols=3, wspace=0.2, size=25)

    #%% staircase heatmap for gene-program p-values

    _gene_programs_plots_artifact_suffix = "source_mgate" if model is source_mgate else "target_mgate"
    _gene_programs_plots_artifact_dir = "gene_programs_plots/{}".format(_gene_programs_plots_artifact_suffix)

    def _log_mlflow_figure(fig, artifact_filename):
        with tempfile.TemporaryDirectory() as _fig_tmp:
            _path = os.path.join(_fig_tmp, artifact_filename)
            fig.savefig(_path, format="svg", bbox_inches="tight")
            client.log_artifact(run_id, _path, artifact_path=_gene_programs_plots_artifact_dir)
        print(
            "Logged {}/{} to MLflow run {}.".format(
                _gene_programs_plots_artifact_dir, artifact_filename, run_id
            )
        )

    def staircase_heatmap(pathway_embedding_results, adata, adata_label, plot_spatial=False, gp_name=None):
        import seaborn as sns

        cluster_gp_scores = pathway_embedding_results[adata_label].pathway_mean_by_cluster.T
        cluster_gp_scores.columns = pd.Categorical(cluster_gp_scores.columns, ordered=True, categories=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'])
        cluster_gp_scores = cluster_gp_scores.sort_index(axis=1)

        topks = []
        for clust in cluster_gp_scores.columns:
            clust_p = cluster_gp_scores[clust]
            topk_gps = clust_p.nlargest(3).index
            topks.append(topk_gps)

        topks = np.hstack(topks)

        cluster_gp_scores_topks = cluster_gp_scores.loc[topks]

        fig_heatmap, axs = plt.subplots(1, 2, figsize=(12, 12), sharey=True)
        sns.heatmap(cluster_gp_scores_topks, cmap='viridis', ax=axs[0])
        sign = np.sign(cluster_gp_scores_topks)
        cluster_gp_scores_topks_abs = np.abs(cluster_gp_scores_topks)
        col_min = cluster_gp_scores_topks_abs.min(axis=0)
        col_max = cluster_gp_scores_topks_abs.max(axis=0)
        cluster_gp_scores_topks_scaled = sign * (cluster_gp_scores_topks_abs - col_min) / (col_max - col_min)
        sns.heatmap(cluster_gp_scores_topks_scaled, cmap='viridis', ax=axs[1])
        plt.tight_layout()
        _log_mlflow_figure(fig_heatmap, "staircase_gp_heatmap_{}.svg".format(adata_label))
        plt.show()

        fig_spatial = None
        if plot_spatial:
            top_idxs = np.stack(np.where(cluster_gp_scores_topks == cluster_gp_scores_topks.values.max())).flatten()
            top_gp_name = cluster_gp_scores_topks.iloc[top_idxs[0]].name
            if gp_name is not None:
                top_gp_name = gp_name
            top_gp_scores = pathway_embedding_results[adata_label].pathway_scores.loc[:,top_gp_name]
            adata.obs[top_gp_name] = top_gp_scores
            sc.pl.embedding(
                adata,
                basis='spatial',
                color=['RNA_clusters', top_gp_name],
                ncols=3,
                wspace=0.2,
                size=75,
                show=False,
            )
            fig_spatial = plt.gcf()
            plt.tight_layout()
            _log_mlflow_figure(fig_spatial, "staircase_gp_spatial_{}.svg".format(adata_label))
            plt.show()

        return cluster_gp_scores, fig_heatmap, fig_spatial

    def cluster_pathway_embedding_heatmap(pathway_embedding_results, adata_label, top_k_pathways=3, row_ind=None, col_ind=None, pathway_order=None):
        import seaborn as sns

        result = pathway_embedding_results[adata_label]
        corr = result.pathway_embedding_correlation_by_cluster
        if corr is None or corr.empty:
            return None, None

        if pathway_order is None:
            #pathway_order = corr.abs().max(axis=1).sort_values(ascending=False).head(top_k_pathways).index
            pathway_order = pd.Series(corr.apply(lambda x: corr.index[np.argsort(x)[:top_k_pathways]], axis=0).T.values.flatten()).drop_duplicates(keep='first').values
            corr_top = corr.loc[pathway_order]
            cg = sns.clustermap(corr_top, cmap='coolwarm', center=0.0)
            _log_mlflow_figure(cg.figure, "pathway_embedding_by_cluster_{}.svg".format(adata_label))
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            corr_top = corr.loc[pathway_order]
            corr_top = corr_top.iloc[row_ind, col_ind]
            sns.heatmap(corr_top, cmap='coolwarm', center=0.0, ax=ax)
            _log_mlflow_figure(fig, "pathway_embedding_by_cluster_{}.svg".format(adata_label))
            cg=None

        plt.show()

        return corr_top, cg, pathway_order

    # Myc_TF_target_genes_GP, Apex1_TF_target_genes_GP
    source_rna_cluster_gp_scores, _fig_stair_heatmap_source, _fig_stair_spatial_source = staircase_heatmap(
        pathway_embedding_results, source_rna, 'source_rna', plot_spatial=True, gp_name='Apex1_TF_target_genes_GP'
    )
    target_rna_cluster_gp_scores, _fig_stair_heatmap_target, _fig_stair_spatial_target = staircase_heatmap(
        pathway_embedding_results, target_rna, 'target_rna', plot_spatial=True, gp_name='Apex1_TF_target_genes_GP'
    )


    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd


    ## remove terms from correlation matrix, e.g. TF targets
    terms_blacklist = None #['TF_target', 'combined']
    if terms_blacklist is not None:
        pathway_embedding_results_cp = pathway_embedding_results.copy()
        pathway_embedding_results_cp['source_rna'].pathway_embedding_correlation_by_cluster = \
            pathway_embedding_results_cp['source_rna'].pathway_embedding_correlation_by_cluster.loc[
                ~pathway_embedding_results_cp['source_rna'].pathway_embedding_correlation_by_cluster.index.str.contains('|'.join(terms_blacklist))
            ]

    # 1) Get source corr matrix and pathway order
    source_corr, _, pathway_order_source = cluster_pathway_embedding_heatmap(
        pathway_embedding_results, "source_rna", top_k_pathways=2
    )
    source_corr = source_corr.loc[pathway_order_source]

    # 2) Temporary clustermap only to extract row/col ordering
    cg = sns.clustermap(source_corr, cmap="coolwarm", center=0.0)
    row_ind = cg.dendrogram_row.reordered_ind
    col_ind = cg.dendrogram_col.reordered_ind
    plt.close(cg.figure)

    # Reorder source
    source_corr_ord = source_corr.iloc[row_ind, col_ind]

    # 3) Build target using same pathway + same row/col order
    target_corr = pathway_embedding_results["target_rna"].pathway_embedding_correlation_by_cluster
    target_corr = target_corr.loc[pathway_order_source].iloc[row_ind, col_ind]

    # 4) Overlap
    overlap = softmax(source_corr_ord * target_corr / 5.0)
    overlap = (overlap - overlap.min()) / (overlap.max() - overlap.min())
    overlap_df = pd.DataFrame(overlap, index=target_corr.index, columns=target_corr.columns)

    '''
    ## rename "emb" to "topic"
    source_corr_ord.columns = source_corr_ord.columns.str.replace('emb', 'topic')
    target_corr.columns = target_corr.columns.str.replace('emb', 'topic')
    overlap_df.columns = overlap_df.columns.str.replace('emb', 'topic')

    ## compare with alpha
    alpha_df_ord = alpha_df.copy()
    alpha_df_ord.columns = alpha_df_ord.columns.str.split('__').str[0]
    alpha_df_ord = alpha_df_ord.groupby(alpha_df_ord.columns, axis=1).mean()
    alpha_df_ord = alpha_df_ord.loc[overlap_df.columns, overlap_df.index]
    plt.figure(figsize=(10, 12))
    sns.heatmap(alpha_df_ord.T, cmap='coolwarm'); plt.tight_layout(); plt.show()
    '''

    # 5) One combined figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), constrained_layout=True, sharey=True, sharex=True)
    sns.heatmap(source_corr_ord, cmap="coolwarm", center=0.0, ax=axes[0], cbar=True)
    sns.heatmap(target_corr, cmap="coolwarm", center=0.0, ax=axes[1], cbar=True)
    sns.heatmap(overlap_df, cmap="coolwarm", ax=axes[2], cbar=True)
    axes[0].set_title("Source data")
    axes[1].set_title("Target data")
    axes[2].set_title("Source-target overlap")
    _log_mlflow_figure(fig, "pathway_embedding_triptych.svg")
    plt.show()

    ## compare with p-values
    corr_pvals = pathway_embedding_results['source_rna'].pathway_embedding_p_values_by_cluster.copy()
    corr_pvals = corr_pvals.loc[overlap_df.index, overlap_df.columns]
    corr_pvals_bin = corr_pvals < 0.05
    sns.heatmap(corr_pvals_bin, center=0.0, cbar=False); plt.show()

    # Cluster-by-cluster comparison: grid of panels (scatter + linear fit per cluster)
    from scipy.stats import pearsonr

    overlap_clusters = source_rna_cluster_gp_scores.columns.intersection(target_rna_cluster_gp_scores.columns)
    all_data = []
    for cluster in overlap_clusters:
        source_cluster_gp_scores = source_rna_cluster_gp_scores[cluster]
        target_cluster_gp_scores = target_rna_cluster_gp_scores[cluster]
        for gp in source_cluster_gp_scores.index:
            all_data.append({
                'source': source_cluster_gp_scores[gp],
                'target': target_cluster_gp_scores[gp],
                'Gene Program': gp,
                'Cluster': cluster
            })
    all_data_df = pd.DataFrame(all_data)
    import seaborn as sns

    n_clust = len(overlap_clusters)
    rna_cluster_colormaps = dict(zip(
        source_concat_adata.obs['RNA_clusters'].cat.categories.tolist(),
        source_concat_adata.uns['RNA_clusters_colors']
    ))
    if n_clust > 0:
        ncols = int(np.ceil(n_clust / 2))
        nrows = 2
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.4 * ncols, 3.4 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.flatten()
        line_color = '#1a1a1a'
        for i, cluster in enumerate(overlap_clusters):
            ax = axes_flat[i]
            sub = all_data_df.loc[all_data_df['Cluster'] == cluster]
            marker_color = rna_cluster_colormaps[cluster]
            # Compute Pearson r for this cluster
            if len(sub) > 1:
                pearson_r, _ = pearsonr(sub['source'], sub['target'])
                pearson_r_str = f" (r={pearson_r:.2f})"
            else:
                pearson_r_str = ""
            sns.regplot(
                data=sub,
                x='source',
                y='target',
                ax=ax,
                ci=None,
                scatter_kws={
                    's': 36,
                    'alpha': 0.75,
                    'marker': 'x',
                    'color': marker_color,
                    'edgecolor': marker_color,
                },
                line_kws={'color': line_color, 'lw': 2, 'alpha': 0.95},
            )
            ax.set_title(f"{cluster}{pearson_r_str}", fontsize=10)
            ax.grid(True, color='#dddddd', linewidth=0.5)
        for j in range(n_clust, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle('Source vs target gene program scores (per cluster)', y=1.02)
        fig.supxlabel('source cluster GP score')
        fig.supylabel('target cluster GP score')
        plt.tight_layout(rect=[0.04, 0.04, 1, 0.98])
        _log_mlflow_figure(fig, "source_vs_target_gene_program_scores_per_cluster.svg")
        plt.show()

    #%% LIANA+ inflow analysis
    import liana as li
    import plotnine as p9
    import squidpy as sq
    import ot
    import gc
    from sklearn.model_selection import StratifiedShuffleSplit

    def liana_spatial_analysis(
        adata,
        subsample_n=5000,
        resource=None,
        spatial_key="spatial",
        cell_type_col="RNA_clusters",
        labels=["R2", "R4", "R7"], interaction='R1^Mdk^Alk', ncomps=30, bandwidth=40, s=60
        ):
        """
        Performs LIANA+ inflow and associated spatial ligand-receptor analyses on `source_rna` AnnData object.

        Parameters:
            source_rna: AnnData
                Source RNA AnnData object with spatial information.
            source_rna_emb: np.ndarray
                Embedding array for reference for GW distance computation.
            labels: list
                Labels for feature_by_group (cell types).
            interaction: str
                Ligand-receptor interaction string to visualize, e.g., 'R1^Mdk^Alk'.
            ncomps: int
                Number of NMF components for factorization.
            bandwidth: int
                Bandwidth for spatial neighbors.
            s: int
                Dot size for spatial plots.
            cell_type_col: str
                Column in .obs with cell type/cluster labels.
            spatial_key: str
                Key in .obsm for spatial coordinates.
        Returns:
            dict of intermediate outputs (optional).
        """

        sc.pl.embedding(adata, basis=spatial_key, color=[cell_type_col], wspace=0.4, s=s)

        plot, df = li.ut.query_bandwidth(
            coordinates=adata.obsm[spatial_key],
            start=5,         
            end=60,           
            interval_n=40    
        )
        plot + p9.scale_y_continuous(breaks=range(int(df.neighbours.min()), int(df.neighbours.max())+1))

        li.ut.spatial_neighbors(adata=adata, bandwidth=bandwidth, spatial_key=spatial_key)
        li.pl.connectivity(adata, idx=5500, size=1, figure_size=(6, 5), spatial_key=spatial_key)

        sq.gr.spatial_autocorr(adata, mode='moran', use_raw=False, show_progress_bar=True)
        svgs = adata.uns['moranI'].index[(adata.uns['moranI']['pval_norm_fdr_bh'] < 0.05) & (adata.uns['moranI']['I'] > 0.01)]
        adata = adata[:, svgs]
        print(f"Number of spatially variable genes: {len(svgs)}")

        map_df = pd.read_csv(os.path.join(os.environ.get('DATAPATH'), 'gene_annotations', 'human_mouse_gene_orthologs.csv')) 
        map_df = map_df.rename(columns={"Gene name": "source", "Mouse gene name": "target"})
        map_df = map_df.drop(columns=["Gene stable ID", "Mouse gene stable ID"])

        if resource is None:
            resource = li.rs.select_resource('consensus') # NOTE: mouse_consensus could be used in future
            resource = li.rs.translate_resource(
                resource,
                map_df=map_df,
                columns=['ligand', 'receptor'],
                replace=True,
                one_to_many=2,
            )

        lrdata = li.mt.inflow(
            adata,
            groupby=cell_type_col,
            resource=resource,
            use_raw=False,
        )

        sq.gr.spatial_autocorr(lrdata, mode='moran', use_raw=False)
        svis = lrdata.uns['moranI'].index[(lrdata.uns['moranI']['pval_norm_fdr_bh'] <= 0.05) & (lrdata.uns['moranI']['I'] > 0.01)]
        print(f"Number of spatially variable ligand-receptor interactions: {len(svis)}")
        lrdata = lrdata[:, svis]
        lrdata.uns['moranI'].sort_values("I").tail(30)

        try:
            ligands = lrdata.var_names.str.split("^").str[1]
            receptors = lrdata.var_names.str.split("^").str[2]
            fused_I = adata.uns['moranI'].loc[ligands, 'I'].values * adata.uns['moranI'].loc[receptors, 'I'].values
            print(lrdata.var_names[np.flip(fused_I.argsort())])
        except:
            fused_I = None

        comp = interaction.split("^")

        sc.pl.embedding(
            lrdata,
            basis=spatial_key,
            color=interaction,
            s=s,
            ncols=2,
        )

        sc.pl.embedding(
            adata,
            basis=spatial_key,
            color=[comp[1], comp[2]],
            s=s,
            use_raw=False,
            ncols=2,
        )

        fig, ax = plt.subplots(figsize=(14, 5)) 
        sc.pl.violin(lrdata, groupby=cell_type_col, keys=interaction, size=0.5, rotation=90, ax=ax)
        plt.tight_layout()
        plt.show()

        li.pl.feature_by_group(
            adata=lrdata,
            spatial_key=spatial_key,
            feature=interaction,
            groupby=cell_type_col,
            percentile_scaling=(1,97),
            labels=labels,
            show_counts=False,
            normalize=True,
            figure_size=(10,8),
        )
        ## stopped at "Global Summaries"

        ## NMF
        li.multi.nmf(lrdata, n_components=ncomps, inplace=True, random_state=0, max_iter=200, verbose=True)
        lr_loadings = li.ut.get_variable_loadings(lrdata, varm_key='NMF_H').set_index('index')
        factor_scores = li.ut.get_factor_scores(lrdata, obsm_key='NMF_W')

        X_nmf = lrdata.obsm['NMF_W']
        keep_nmf = X_nmf.sum(axis=1) > 0
        lrdata_nmf = lrdata[keep_nmf].copy()
        nmf = sc.AnnData(X=lrdata_nmf.obsm['NMF_W'],
                        obs=lrdata_nmf.obs,
                        var=pd.DataFrame(index=lr_loadings.columns),
                        uns=lrdata.uns,
                        obsm=lrdata_nmf.obsm)
        lr_loadings.head(10)

        sc.pp.neighbors(nmf, use_rep='X', metric='euclidean', n_neighbors=30)
        sc.tl.leiden(nmf, resolution=0.1)
        sc.tl.umap(nmf, min_dist=0.5)
        sc.pl.umap(nmf, color=[cell_type_col, 'leiden'], size=25, ncols=2)

        sc.pl.embedding(nmf, basis=spatial_key, color=[cell_type_col, 'leiden'], size=s, ncols=2)

        stratified_subsample_applied = False
        if subsample_n is not None and subsample_n < nmf.n_obs:
            y = nmf.obs[cell_type_col].astype(str).to_numpy()
            try:
                sss = StratifiedShuffleSplit(
                    n_splits=1, train_size=subsample_n, random_state=0
                )
                train_idx, _ = next(sss.split(np.zeros((nmf.n_obs, 1)), y))
            except ValueError:
                # Too few cells per class (or similar) for stratified split.
                train_idx = np.random.default_rng(0).choice(
                    nmf.n_obs, size=subsample_n, replace=False
                )
            chosen = nmf.obs_names[train_idx].tolist()
            nmf = nmf[chosen].copy()
            adata = adata[chosen].copy()
            stratified_subsample_applied = True

        '''
        ## GW distance
        X_nmf = nmf.X
        if stratified_subsample_applied:
            X_source = np.asarray(adata.obsm["MultiGATE"])
        else:
            X_source = adata.obsm["MultiGATE"][keep_nmf]
        C_nmf = ot.utils.dist(X_nmf, metric='euclidean')
        C_source = ot.utils.dist(X_source, metric='euclidean')
        C_nmf = torch.from_numpy(C_nmf / C_nmf.max()).to('cuda:2', dtype=torch.float64)
        C_source = torch.from_numpy(C_source / C_source.max()).to('cuda:2', dtype=torch.float64)
        gw_distance = ot.gromov.entropic_gromov_wasserstein2(C_nmf, C_source, eps=1e-2)
        #gw_distance = ot.gaussian.gaussian_gromov_wasserstein_distance(C_nmf, C_source, log=True)[0]
        gw_distance = 0.5 * gw_distance.sqrt().item()

        del C_nmf, C_source
        torch.cuda.empty_cache()
        gc.collect()
        '''

        # Return a dictionary of useful results, or could just not return if only for plotting side-effects
        return {
            "adata": adata,
            "lrdata": lrdata,
            "nmf": nmf,
            "fused_I": fused_I,
            "svgs": svgs,
            "svis": svis,
            "lr_loadings": lr_loadings,
            "factor_scores": factor_scores
        }

    ## map ATAC peaks to genes
    gpnet = source_atac.uns['gene_peak_Net'].copy()
    gpmap = gpnet.groupby("Gene", sort=True)["Peak"].unique()
    missing_genes = set(source_rna.var_names) - set(gpmap.index)
    gene_names = pd.Index(gpmap.index).intersection(source_rna.var_names, sort=False)

    gp_distances = pd.pivot(gpnet, index='Gene', columns='Peak', values='Distance').fillna(np.inf)
    for gene in missing_genes:
        gp_distances.loc[gene] = np.inf
    gp_distances = gp_distances.loc[source_rna.var_names, source_atac.var_names]
    gp_weights = 1 / (gp_distances + 1)

    gp_pairs = gpmap.loc[gene_names].explode().reset_index().rename(columns={"index": "Gene"})
    peak_indexer = source_atac.var_names.get_indexer(gp_pairs["Peak"])
    missing_peak_mask = peak_indexer < 0
    if missing_peak_mask.any():
        missing_peaks = gp_pairs.loc[missing_peak_mask, "Peak"].unique()
        raise KeyError(
            "gene_peak_Net contains peaks not present in source_atac.var_names: "
            f"{missing_peaks[:10].tolist()}"
        )

    gene_indexer = gene_names.get_indexer(gp_pairs["Gene"])
    weight_row_indexer = source_rna.var_names.get_indexer(gp_pairs["Gene"])
    edge_weights = gp_weights.to_numpy()[weight_row_indexer, peak_indexer].astype(
        np.float32,
        copy=False,
    )

    peak_to_gene = sp.coo_matrix(
        (edge_weights, (peak_indexer, gene_indexer)),
        shape=(source_atac.n_vars, gene_names.size),
    ).tocsc()

    atac_matrix = source_atac.X.tocsr() if sp.issparse(source_atac.X) else np.asarray(source_atac.X)
    n_jobs = min(os.cpu_count() or 1, gene_names.size)
    chunk_size = max(1, (gene_names.size + n_jobs - 1) // n_jobs)
    chunk_bounds = [
        (start, min(start + chunk_size, gene_names.size))
        for start in range(0, gene_names.size, chunk_size)
    ]

    def _score_gene_chunk(start: int, stop: int) -> np.ndarray:
        peak_to_gene_chunk = peak_to_gene[:, start:stop]
        if sp.issparse(atac_matrix):
            chunk_scores = atac_matrix @ peak_to_gene_chunk
            return chunk_scores.toarray()
        return np.asarray(atac_matrix @ peak_to_gene_chunk.toarray())

    gp_score_chunks = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_score_gene_chunk)(start, stop) for start, stop in chunk_bounds
    )
    gp_scores = pd.DataFrame(
        np.concatenate(gp_score_chunks, axis=1),
        index=source_rna.obs_names,
        columns=gene_names,
    )

    for missing_gene in missing_genes:
        gp_scores[missing_gene] = np.nan

    gp_scores = gp_scores.loc[:, source_rna.var_names]

    assert gp_scores.index.equals(source_rna.obs_names)
    assert gp_scores.columns.equals(source_rna.var_names)

    from scipy.sparse import csr_matrix
    source_rna.layers['fusion_scores'] = \
        csr_matrix(gp_scores.values.astype(source_rna.X.dtype)).multiply(
        source_rna.X
    )

    source_fusion = source_rna.copy()
    source_fusion.X = source_fusion.layers['fusion_scores'].copy()
    del source_rna.layers['fusion_scores']

    ## format combined_gp_dict to adapt to LIANA+
    basenames = [gp.split('__')[0] for gp in source_mgate.pathway_names]
    combined_gp_df = pd.DataFrame()
    for gp in basenames:
        gp_dict = combined_gp_dict[gp]
        sources_df = pd.DataFrame(np.stack([gp_dict['sources'], gp_dict['sources_categories']], axis=1), columns=['ligand', 'source_category'])
        targets_df = pd.DataFrame(np.stack([gp_dict['targets'], gp_dict['targets_categories']], axis=1), columns=['receptor', 'target_category'])
        all_pairs_df = sources_df.merge(targets_df, how='cross')
        combined_gp_df = pd.concat([combined_gp_df, all_pairs_df])
   
    enforce_ligand_receptor = True
    if enforce_ligand_receptor:
        combined_gp_df = combined_gp_df.loc[
            (combined_gp_df['source_category'] == 'ligand') & (combined_gp_df['target_category'] == 'receptor')
        ]

    resource = combined_gp_df.drop(columns=['source_category', 'target_category']).drop_duplicates()

    ## LIANA+ inflow analysis
    source_liana_results = liana_spatial_analysis(
        source_rna,
        subsample_n=5000,
        resource=resource,
        labels=["R2", "R4", "R7"], interaction='R1^Mdk^Alk', ncomps=30, bandwidth=40, s=60, cell_type_col="RNA_clusters", spatial_key="spatial"
        )

    target_liana_results = liana_spatial_analysis(
        target_rna,
        subsample_n=5000,
        resource=resource,
        labels=["R2", "R4", "R7"], interaction='R1^Mdk^Alk', ncomps=30, bandwidth=40, s=60, cell_type_col="RNA_clusters", spatial_key="spatial"
        )

    ## LIANA-MultiGATE heatmaps
    from scipy.stats import rankdata
    adata = source_liana_results['adata'].copy()
    nmf = source_liana_results['nmf'].copy()
    x1 = adata.obsm['MultiGATE']
    x2 = nmf.X
    x1 = rankdata(x1, axis=0)
    x2 = rankdata(x2, axis=0)
    #x1 = np.argsort(x1, axis=0)
    #x2 = np.argsort(x2, axis=0)
    corrs = np.corrcoef(x1, x2, rowvar=False)
    corrs = corrs[x1.shape[1]:, :x1.shape[1]]
    corrs_df = pd.DataFrame(corrs, index=nmf.var_names.str.replace('Factor', 'NMF'))
    topk=3
    corrs_order = pd.Series(corrs_df.apply(lambda x: corrs_df.index[np.argsort(x)[:topk]], axis=0).T.values.flatten()).drop_duplicates(keep='first').values.tolist()
    corr_top = corrs_df.loc[corrs_order]
    cg = sns.clustermap(corr_top, cmap='coolwarm', center=0.0, vmax=0.4, figsize=(9.5, 5))
    cg.ax_heatmap.set_xlabel('MultiGATE dimensions')
    plt.show()

    ## compare ARI between NMF leiden and source, target and nonspatial-source leiden
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ## source ARI
    nmf = source_liana_results['nmf'].copy()
    source_leiden = source_concat_adata[
            source_concat_adata.obs_names.str.split("_").str[0].isin(nmf.obs_names) &
            source_concat_adata.obs['modality'].eq('rna')
            ].obs['leiden']
    source_leiden.index = source_leiden.index.str.split("_").str[0]
    source_leiden = source_leiden.loc[nmf.obs_names]
    nmf_leiden = nmf.obs['leiden']
    assert source_leiden.index.equals(nmf_leiden.index)
    source_ari = adjusted_rand_score(source_leiden, nmf_leiden)
    source_nmi = normalized_mutual_info_score(source_leiden, nmf_leiden)
    print(f'Adjusted Rand Score: {source_ari:.2f}')
    print(f'Normalized Mutual Information Score: {source_nmi:.2f}')

    ## target ARI
    nmf = target_liana_results['nmf'].copy()
    target_leiden = target_concat_adata[
            #target_concat_adata.obs['source_obs_names'].str.split("_").str[0].isin(nmf.obs_names) &
            target_concat_adata.obs_names.str.split("_").str[0].isin(nmf.obs_names) &
            target_concat_adata.obs['modality'].eq('rna')
            #].obs.set_index('source_obs_names')['leiden']
            ].obs['leiden']
    target_leiden.index = target_leiden.index.str.split("_").str[0]
    overlap = target_leiden.index.intersection(nmf.obs_names)

    # Drop duplicate indices in target_leiden via majority voting
    # Find duplicated indices
    duplicated = target_leiden.index[target_leiden.index.duplicated(keep=False)]
    if len(duplicated) > 0:
        # For each duplicated index, assign label by majority vote
        maj_labels = (
            target_leiden[duplicated]
            .groupby(level=0)
            .agg(lambda x: x.value_counts().idxmax())
        )
        # Remove all duplicates
        target_leiden = target_leiden[~target_leiden.index.duplicated(keep=False)]
        # Add back the majority-vote labels
        target_leiden = pd.concat([target_leiden, maj_labels]).sort_index()

    target_leiden = target_leiden.loc[overlap]
    nmf_leiden = nmf.obs['leiden'].loc[overlap]
    target_leiden = target_leiden.loc[nmf_leiden.index]
    assert target_leiden.index.equals(nmf_leiden.index)
    target_ari = adjusted_rand_score(target_leiden, nmf_leiden)
    target_nmi = normalized_mutual_info_score(target_leiden, nmf_leiden)
    print(f'Adjusted Rand Score: {target_ari:.2f}')
    print(f'Normalized Mutual Information Score: {target_nmi:.2f}')

    ## nonspatial source ARI
    nmf = source_liana_results['nmf'].copy()
    nonspatial_source_leiden = nonspatial_source_concat_adata[
            nonspatial_source_concat_adata.obs_names.str.split("_").str[0].isin(nmf.obs_names) &
            nonspatial_source_concat_adata.obs['modality'].eq('rna')
            ].obs['leiden']
    nonspatial_source_leiden.index = nonspatial_source_leiden.index.str.split("_").str[0]
    nonspatial_source_leiden = nonspatial_source_leiden.loc[nmf.obs_names]
    nmf_leiden = nmf.obs['leiden']
    assert nonspatial_source_leiden.index.equals(nmf_leiden.index)
    nonspatial_source_ari = adjusted_rand_score(nonspatial_source_leiden, nmf_leiden)
    nonspatial_source_nmi = normalized_mutual_info_score(nonspatial_source_leiden, nmf_leiden)
    print(f'Adjusted Rand Score: {nonspatial_source_ari:.2f}')
    print(f'Normalized Mutual Information Score: {nonspatial_source_nmi:.2f}')

    ## ingested nonspatial source ARI
    ingested_nonspatial_source_concat_adata = sc.tl.ingest(nonspatial_source_concat_adata, source_concat_adata, embedding_method='umap', obs='leiden', inplace=False)
    nmf = source_liana_results['nmf'].copy()
    nonspatial_source_leiden = ingested_nonspatial_source_concat_adata[
            ingested_nonspatial_source_concat_adata.obs_names.str.split("_").str[0].isin(nmf.obs_names) &
            ingested_nonspatial_source_concat_adata.obs['modality'].eq('rna')
            ].obs['leiden']
    nonspatial_source_leiden.index = nonspatial_source_leiden.index.str.split("_").str[0]
    nonspatial_source_leiden = nonspatial_source_leiden.loc[nmf.obs_names]
    nmf_leiden = nmf.obs['leiden']
    assert nonspatial_source_leiden.index.equals(nmf_leiden.index)
    ingested_nonspatial_source_ari = adjusted_rand_score(nonspatial_source_leiden, nmf_leiden)
    ingested_nonspatial_source_nmi = normalized_mutual_info_score(nonspatial_source_leiden, nmf_leiden)
    print(f'Adjusted Rand Score: {ingested_nonspatial_source_ari:.2f}')
    print(f'Normalized Mutual Information Score: {ingested_nonspatial_source_nmi:.2f}')

    ## compare ARI and NMI between source, target and nonspatial-source
    ari_nmi_df = pd.DataFrame({
        'ARI': [source_ari, target_ari, nonspatial_source_ari, ingested_nonspatial_source_ari],
        'NMI': [source_nmi, target_nmi, nonspatial_source_nmi, ingested_nonspatial_source_nmi]
    }, index=['Source', 'Target', 'Nsp Source', 'Nsp Source (ingst.)'])
    ari_nmi_df.plot(kind='bar', rot=30, cmap='Set2', figsize=(4, 5))

    ## plot UMAP of source, target, nonspatial-source and ingested nonspatial-source
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sc.pl.umap(source_concat_adata, color='leiden', size=25, ax=axs[0, 0], show=False)
    sc.pl.umap(target_concat_adata, color='leiden', size=25, ax=axs[0, 1], show=False)
    sc.pl.umap(nonspatial_source_concat_adata, color='leiden', size=25, ax=axs[1, 0], show=False)
    sc.pl.umap(ingested_nonspatial_source_concat_adata, color='leiden', size=25, ax=axs[1, 1], show=False)
    plt.tight_layout()
    plt.show()

    #%% embedding-to-gene modelling
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression, PoissonRegressor
    from sklearn.multioutput import MultiOutputRegressor

    #X = pathway_embedding_results['source_rna'].pathway_scores.to_numpy()
    X = source_rna.X.toarray()
    Z = source_rna.obsm['MultiGATE'].copy()
    C = source_rna.obs['RNA_clusters'].copy()

    pls = PLSRegression(n_components=30)
    pls.fit(Z, X)
    T = pls.transform(Z)

    model = MultiOutputRegressor(PoissonRegressor())
    model.fit(T, X)
    #model.fit(Z, X)

    B = np.column_stack([est.coef_ for est in model.estimators_])
    b = np.array([est.intercept_ for est in model.estimators_])

    genes_to_embs = (pls.x_weights_ @ B).T
    #genes_to_embs = B.T
    n_genes, n_embs = genes_to_embs.shape # shape: (n_emb, n_genes)

    # Get gene names and embedding names
    gene_names = source_rna.var_names
    emb_names = [f"emb_{i}" for i in range(n_embs)]

    # For each embedding dimension, find top 5 genes by absolute coefficient
    topk = 5
    top_genes = []

    for emb_idx in range(n_embs):
        col = genes_to_embs[:, emb_idx]
        col = abs(col)
        top_gene_indices = np.argsort(col)[::-1][:topk]
        top_genes.append(gene_names[top_gene_indices])

    top_genes = np.hstack(top_genes)

    genes_to_embs_df = pd.DataFrame(genes_to_embs, index=gene_names, columns=emb_names)
    top_genes_to_embs_df = genes_to_embs_df.loc[top_genes]
    top_genes_to_embs_df = top_genes_to_embs_df.abs()

    sns.heatmap(top_genes_to_embs_df, cmap='viridis', cbar=True)
    plt.title("Top 5 genes per embedding (PLS coefficients)")
    plt.ylabel("Gene")
    plt.xlabel("Embedding dimension")
    plt.tight_layout()
    plt.show()

    #%% AJIVE analysis
    from mvlearn.decomposition import AJIVE
    import seaborn as sns

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



    
    #%% compare spatial graph with latent knn graph
    n_neighbors = 100

    sc.pp.neighbors(source_rna, use_rep='MultiGATE', n_neighbors=n_neighbors)
    sc.pp.neighbors(source_atac, use_rep='MultiGATE', n_neighbors=n_neighbors)
    rna_multigate_knn_graph = source_rna.obsp['connectivities']
    atac_multigate_knn_graph = source_atac.obsp['connectivities']

    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE")
    source_concat_adata.obs['jaccard_similarity'] = jaccard_similarity
    source_concat_adata.obsm['spatial'] = np.concatenate([source_rna.obsm['spatial'], source_atac.obsm['spatial']], axis=0)

    MultiGATE.Cal_Spatial_Net(source_rna, model='KNN', k_cutoff=n_neighbors)
    MultiGATE.Cal_Spatial_Net(source_atac, model='KNN', k_cutoff=n_neighbors)

    spatial_knn_graph, _, _, _ = build_graph_inputs(source_rna, source_atac)
    indices, values, shape = spatial_knn_graph
    indices = np.asarray(indices)
    row, col = indices[:, 0], indices[:, 1]
    spatial_knn_graph = sp.coo_matrix((values, (row, col)), shape=shape).tocsr()
    assert indices.shape[1] == 2, "spatial_knn_graph should have 2 columns"

    def jaccard_per_sample_csr(
        a: sp.csr_matrix,
        b: sp.csr_matrix,
        *,
        binarize: bool = True,
        empty_value: float = 1.0,
    ) -> np.ndarray:
        if not sp.isspmatrix_csr(a):
            a = a.tocsr()
        else:
            a = a.copy()

        if not sp.isspmatrix_csr(b):
            b = b.tocsr()
        else:
            b = b.copy()

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        a.sum_duplicates()
        b.sum_duplicates()
        a.eliminate_zeros()
        b.eliminate_zeros()

        if binarize:
            a.data = np.ones_like(a.data, dtype=np.uint8)
            b.data = np.ones_like(b.data, dtype=np.uint8)

        # For binary matrices:
        # intersection count per row = number of coordinates nonzero in both
        inter = a.multiply(b).count_nonzero(axis=1)

        # union count per row = nnz(a_row) + nnz(b_row) - intersection
        a_nnz = a.count_nonzero(axis=1)
        b_nnz = b.count_nonzero(axis=1)
        union = a_nnz + b_nnz - inter

        inter = np.asarray(inter).ravel()
        union = np.asarray(union).ravel()

        out = np.empty(a.shape[0], dtype=float)
        mask = union == 0
        out[~mask] = inter[~mask] / union[~mask]
        out[mask] = empty_value
        return out
        
    # compute the Jaccard similarity between the two graphs
    rna_jaccard_similarity = jaccard_per_sample_csr(rna_multigate_knn_graph, spatial_knn_graph)
    atac_jaccard_similarity = jaccard_per_sample_csr(atac_multigate_knn_graph, spatial_knn_graph)
    jaccard_similarity = np.concatenate([rna_jaccard_similarity, atac_jaccard_similarity])
    plt.hist(jaccard_similarity, bins=50)

    sc.pl.embedding(source_concat_adata, basis="spatial", color="jaccard_similarity", s=50, legend_loc='None')
    sc.pl.embedding(source_concat_adata, basis="spatial", color="RNA_clusters", s=50, legend_loc='None')
    
    
    
    #%% attention matrix analysis
    from post_hoc_utils import run_gene_peak_attention_tutorial

    # Find the artifact path for the attention matrix
    with tempfile.TemporaryDirectory() as tmp_dir:
        attention_artifact_path = "matrices/source_peak_gene_attention.npz"
        if not artifact_exists(client, run_id, attention_artifact_path):
            raise FileNotFoundError(
                "Artifact '{}' not found for run {}.".format(
                    attention_artifact_path, run_id
                )
            )
        local_path = client.download_artifacts(
            run_id,
            attention_artifact_path,
            tmp_dir,
        )
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
    if pathway_embedding_results is not None:
        print(
            "Pathway embedding analysis: "
            + ", ".join(
                "{}={}".format(k, "ok" if v is not None else "skipped/failed")
                for k, v in pathway_embedding_results.items()
            )
        )
    return source_rna, source_atac, target_rna, target_atac, pathway_embedding_results

#%%
if __name__ == "__main__":
    main()
