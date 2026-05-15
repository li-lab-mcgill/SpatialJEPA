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
import shlex
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

## Plot default settings
try:
    # Set matplotlib backend to enable retina display, equivalent to `%config InlineBackend.figure_format = 'retina'`
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('config', 'InlineBackend.figure_format = "retina"')
except Exception:
    pass

import matplotlib as mpl
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'figure.titlesize': 18,
})


def rasterize_heavy_pdf_artists(fig):
    """Rasterize dense artists while keeping text and axes vector in PDFs."""
    for ax in fig.get_axes():
        for artist in ax.collections:
            artist.set_rasterized(True)
        for artist in ax.images:
            artist.set_rasterized(True)


def save_figure_pdf_vector_raster(fig, filepath, dpi=300):
    """Save a PDF with dense layers rasterized and text/axes vector (TrueType fonts)."""
    import logging

    rasterize_heavy_pdf_artists(fig)
    d = os.path.dirname(filepath)
    if d:
        os.makedirs(d, exist_ok=True)
    fonttools_logger = logging.getLogger("fontTools")
    prev_level = fonttools_logger.level
    fonttools_logger.setLevel(logging.WARNING)
    try:
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(filepath, format="pdf", dpi=dpi, bbox_inches="tight")
    finally:
        fonttools_logger.setLevel(prev_level)


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

os.chdir(os.path.join(BAKLAVA_BASE_DIR, "MultiGATE", "scripts"))

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

import liana as li
import decoupler as dc
import decoupler.op
import squidpy as sq
import ot

import MultiGATE
from MultiGATE.model_MultiGATE import MGATE

print("Using MultiGATE module:", MultiGATE.__file__)

from mouse_brain_spatial_rna_atac import (  # noqa: E402
    apply_hvg_and_gp_filtering,
    get_sct_genes_and_gp_filtering,
    build_concat_adata_for_umap,
    build_graph_inputs,
    build_source_student_graph_tf,
    compute_concat_umap,
    compute_scib_metrics_for_domain,
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


MODEL_METRIC_GROUP_PREFIXES = (
    ("source_train", "source_training_loss"),
    ("stage1_nonspatial_train", "nonspatial_training_loss"),
    ("stage1_student_distill", "distillation"),
    ("source_scib", "source_scib"),
    ("target_scib", "target_scib"),
    ("stage1_nonspatial_source_target", "nonspatial_alignment"),
    ("stage1_source_target", "alignment"),
)

# Ordered longest-first so more-specific prefixes match before shorter ones.
_DOMAIN_PREFIXES = (
    "stage2_target",
    "stage1_nonspatial",
    "stage1_student",
    "stage1",
    "stage2",
    "source",
    "target",
)


def _metric_group(metric_name):
    for prefix, group in MODEL_METRIC_GROUP_PREFIXES:
        if metric_name.startswith(prefix):
            return group
    return "other"


def parse_metric_domain_and_name(metric_name):
    """
    Split an MLflow metric key into ``(domain, base_metric_name)``.

    scIB keys logged by ``log_scib_metrics`` have the form
    ``{domain}_scib_{base}`` (e.g. ``source_scib_silhouette_label``).
    Other keys have the form ``{domain}_{rest}``
    (e.g. ``source_train_loss``, ``stage2_distill_loss``).

    Domains are matched against ``_DOMAIN_PREFIXES`` longest-first so that
    ``stage2_target`` is preferred over ``stage2`` or ``target``.
    """
    for prefix in _DOMAIN_PREFIXES:
        scib_tag = "{}_scib_".format(prefix)
        if metric_name.startswith(scib_tag):
            return prefix, metric_name[len(scib_tag):]
    for prefix in _DOMAIN_PREFIXES:
        tag = "{}_".format(prefix)
        if metric_name.startswith(tag):
            return prefix, metric_name[len(tag):]
    return "other", metric_name


def extract_logged_model_metrics(client, run_id, run_name=None, metric_keys=None):
    """
    Return all logged MLflow model metric history for a run as a tidy DataFrame.

    One row is emitted per logged metric point. ``domain`` and ``metric_name``
    are split via ``parse_metric_domain_and_name`` so that, e.g.,
    ``source_scib_silhouette_label`` becomes domain=``source``,
    metric_name=``silhouette_label``.  The latest-value table is recoverable
    via ``df.sort_values(["metric_name", "step", "timestamp"])``.
    """
    run = client.get_run(run_id)
    if metric_keys is None:
        metric_keys = sorted(run.data.metrics)

    rows = []
    for raw_key in metric_keys:
        domain, base_name = parse_metric_domain_and_name(raw_key)
        history = client.get_metric_history(run_id, raw_key)
        for point in history:
            rows.append(
                {
                    "run_id": run_id,
                    "run_name": run_name or run.info.run_name,
                    "metric_group": _metric_group(raw_key),
                    "domain": domain,
                    "metric_name": base_name,
                    "value": point.value,
                    "step": point.step,
                    "timestamp": point.timestamp,
                    "logged_at": pd.to_datetime(point.timestamp, unit="ms"),
                }
            )

    columns = [
        "run_id",
        "run_name",
        "metric_group",
        "domain",
        "metric_name",
        "value",
        "step",
        "timestamp",
        "logged_at",
    ]
    metrics_df = pd.DataFrame(rows, columns=columns)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(
            ["domain", "metric_group", "metric_name", "step", "timestamp"]
        ).reset_index(drop=True)
    return metrics_df


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


def mgate_from_state_dict(state_dict, device, skip_gp_attention=True):
    """
    Infer the MGATE architecture entirely from a state dict, instantiate
    the model, load the weights, and return it in eval mode.

    skip_gp_attention is not stored in the state dict; it must be supplied
    explicitly from the MLflow run params of the originating training run.
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
        skip_gp_attention=skip_gp_attention,
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


def load_mgate(client, run_id, artifact_name, device, dst_dir, skip_gp_attention=True):
    """Download a model .pth artifact and return a reconstructed, eval-mode MGATE."""
    local_path = download_model_artifact(client, run_id, artifact_name, dst_dir)
    payload = torch.load(local_path, map_location=device, weights_only=False)
    state_dict, pathway_metadata = unpack_mgate_checkpoint_payload(payload)
    mgate, hidden_dims1, hidden_dims2, vgp_anchor_mode = mgate_from_state_dict(
        state_dict, device, skip_gp_attention=skip_gp_attention
    )
    apply_pathway_metadata_to_mgate(mgate, pathway_metadata)
    print(
        "    hidden_dims1={}, hidden_dims2={}, vgp_anchor_mode={}, skip_gp_attention={}".format(
            hidden_dims1, hidden_dims2, vgp_anchor_mode, skip_gp_attention
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
        skip_gp_attention=source_mgate.skip_gp_attention,
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
    att_lgp = outputs[9][0]
    return rna_emb, atac_emb, att_lgp


def score_gene_peak_attention_links_by_pls(gene_peak_attention_links, rna_coef_df, atac_coef_df):
    """Rank gene-peak attention links by paired RNA/ATAC PLS coefficient ranks."""
    rna_coef_ranked_df = rna_coef_df.abs().rank(1, ascending=False).div(rna_coef_df.shape[1])
    atac_coef_ranked_df = atac_coef_df.abs().rank(1, ascending=False).div(atac_coef_df.shape[1])

    missing_genes = sorted(set(gene_peak_attention_links["Gene"]) - set(rna_coef_ranked_df.columns))
    missing_peaks = sorted(set(gene_peak_attention_links["Peak"]) - set(atac_coef_ranked_df.columns))
    if missing_genes or missing_peaks:
        raise KeyError(
            "PLS coefficient tables do not cover all attention links: "
            "{} missing genes, {} missing peaks.".format(len(missing_genes), len(missing_peaks))
        )

    pls_score_rows = []
    for gene, peak in gene_peak_attention_links[["Gene", "Peak"]].values:
        rna_pls_scores = rna_coef_ranked_df.loc[:, gene]
        atac_pls_scores = atac_coef_ranked_df.loc[:, peak]
        pls_scores = pd.merge(
            rna_pls_scores,
            atac_pls_scores,
            left_index=True,
            right_index=True,
            how="inner",
        )
        pls_scores["mean_rank"] = pls_scores.mean(axis=1)
        pls_scores.rename(columns={gene: "gene_rank", peak: "peak_rank"}, inplace=True)
        pls_score_rows.append(pls_scores.assign(gene=gene, peak=peak))

    all_pls_scores_df = pd.concat(pls_score_rows, ignore_index=False)
    all_pls_scores_df.sort_values("mean_rank", inplace=True)
    all_pls_scores_grouped_df = (
        all_pls_scores_df
        .reset_index()
        .groupby(["index", "gene"])
        .agg(
            mean_min_rank=("mean_rank", "min"),
            n_links=("mean_rank", "count"),
        )
        .sort_values("mean_min_rank")
    )
    return all_pls_scores_df, all_pls_scores_grouped_df


def select_gene_peak_link_df(gene_peak_attention_links, rna_coef_df, atac_coef_df, pls_cmp, gene):
    """Select one gene's peak links and annotate them with domain-specific PLS weights."""
    peaks = gene_peak_attention_links.loc[
        gene_peak_attention_links["Gene"].eq(gene),
        "Peak",
    ].values

    gp_link_df = gene_peak_attention_links.loc[
        gene_peak_attention_links["Gene"].eq(gene)
        & gene_peak_attention_links["Peak"].isin(peaks)
    ].copy()

    gp_link_df[f"{pls_cmp}_gene_weight"] = rna_coef_df.loc[pls_cmp, gene]
    gp_link_df = pd.merge(
        gp_link_df,
        atac_coef_df.loc[pls_cmp, peaks].rename(f"{pls_cmp}_peak_weight"),
        left_on="Peak",
        right_index=True,
        how="inner",
    )
    return gp_link_df, peaks


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
    axs[1, 0].set_title("Target spatial, aligned to source")
    axs[1, 1].set_title("Target UMAP, aligned to source")
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

    args.run_name = '20260505_190037' #'20260402_153455'
    args.stage2_run_name = '20260402_153455_stage2_20260402_165006'
    sqlite_tracking_uri = "sqlite:////home/mcb/users/dmannk/BAKLAVA_base/mlflow_tracking/MultiGATE/mlflow.db"
    postgres_tracking_uri = "http://127.0.0.1:5000"
    args.tracking_uri = postgres_tracking_uri
    args.target_model = "zero_shot" # for zero-shot co-embedding if stage2 model is not available

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

    bp_width           = int(run_params.get("bp_width", 400))
    graph_type         = run_params.get("graph_type", "ATAC")
    dual_source_kd     = run_params.get("stage1_dual_source_kd", "False").lower() == "true"
    skip_gp_attention  = str(run_params.get("skip_gp_attention", "True")).lower() == "true"
    print("skip_gp_attention (from run params):", skip_gp_attention)
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
    source_rna  = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned_SCT.h5ad"))
    source_atac = sc.read_h5ad(os.path.join(base_path, "source_atac_aligned.h5ad"))
    target_rna  = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned_SCT.h5ad"))
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
    use_sct_genes = True
    if use_sct_genes:
        source_rna, source_atac, target_rna, target_atac, gp_net = get_sct_genes_and_gp_filtering(
            source_rna=source_rna,
            source_atac=source_atac,
            target_rna=target_rna,
            target_atac=target_atac,
            gp_net=gp_net,
        )
    else:
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
            client, run_id, source_artifact_name, device, tmpdir,
            skip_gp_attention=skip_gp_attention,
        )

        if dual_source_kd:
            teacher_source_mgate, _, _, _ = load_mgate(
                client, run_id, source_artifact_map["stage1_teacher"], device, tmpdir,
                skip_gp_attention=skip_gp_attention,
            )
            nonspatial_source_mgate, _, _, _ = load_mgate(
                client, run_id, source_artifact_map["stage1_nonspatial"], device, tmpdir,
                skip_gp_attention=skip_gp_attention,
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
                client, resolved_stage2_run_id, "model_stage2.pth", device, tmpdir,
                skip_gp_attention=skip_gp_attention,
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

    #%% data figures
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sc.pl.embedding(source_rna, color='RNA_clusters', basis='spatial', size=40, ax=ax[0], show=False)
    sc.pl.embedding(source_atac, color='ATAC_clusters', basis='spatial', size=40, ax=ax[1], show=False)
    plt.tight_layout(); plt.show()

    import celltypist
    target_rna_ct = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned.h5ad"))

    '''
    import mudata
    import muon
    target_atac_ct = sc.read_h5ad(os.path.join(base_path, "target_atac_aligned.h5ad"))
    target_mudata_ct = mudata.MuData({
        'rna': target_rna_ct,
        'atac': target_atac_ct
    })
    muon.pp.neighbors(target_mudata_ct)
    '''

    # Ensure a neighborhood graph exists for clustering
    if 'neighbors' not in target_rna_ct.uns:
        if 'X_pca' not in target_rna_ct.obsm:
            sc.pp.pca(target_rna_ct)
        sc.pp.neighbors(target_rna_ct)

    # Run leiden clustering with lower resolution for smoother majority voting
    # (CellTypist defaults to res=15 for this dataset size, which is too high)
    sc.tl.leiden(target_rna_ct, resolution=0.5, key_added='over_clustering')
    predictions = celltypist.annotate(
        target_rna_ct,
        model='Mouse_Whole_Brain.pkl',
        majority_voting = True,
        over_clustering = 'over_clustering'
    )
    # Strip the 3-digit numeric prefix (e.g. "001 CLA-EPd-CTX Car3 Glut" -> "CLA-EPd-CTX Car3 Glut")
    cleaned_labels = predictions.predicted_labels['majority_voting'].str.replace(
        r'^\d{3}\s+', '', regex=True
    )
    target_rna_ct.obs['celltypist_predictions'] = cleaned_labels

    sc.tl.umap(target_rna_ct)

    ax = sc.pl.embedding(
        target_rna_ct, color='celltypist_predictions', basis='umap', size=40, show=False
    )
    plt.gcf().set_size_inches(6, 5)
    # Force legend into a single column so all labels align on the same axis
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        frameon=False,
    )
    save_figure_pdf_vector_raster(
        plt.gcf(),
        os.path.join(
            "/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures",
            "target_rna_umap_celltypist_predictions.pdf",
        ),
    )
    plt.show()

    target_rna.obs['celltypist_predictions'] = cleaned_labels
    target_atac.obs['celltypist_predictions'] = cleaned_labels
    del target_rna_ct

    #%% ── Inference, by dataset ────────────────────────────────────────────────────────────

    ## (teacher) source inference
    teacher_source_rna_emb, teacher_source_atac_emb, _ = run_inference(
        teacher_source_mgate, source_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, teacher_source_rna_emb, teacher_source_atac_emb, key_added="MultiGATE_full_teacher")
    print("  Teacher source embeddings: shape {}".format(teacher_source_rna_emb.shape))
    
    ## (student) source inference
    source_rna_emb, source_atac_emb, _ = run_inference(
        source_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, source_rna_emb, source_atac_emb)
    print("  Source embeddings: shape {}".format(source_rna_emb.shape))

    ## nonspatial source inference
    nonspatial_source_rna_emb, nonspatial_source_atac_emb, _ = run_inference(
        nonspatial_source_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, nonspatial_source_rna_emb, nonspatial_source_atac_emb, key_added="MultiGATE_nonspatial")
    print("  Nonspatial source embeddings: shape {}".format(nonspatial_source_rna_emb.shape))

    # Plot source/target concat UMAPs with the same helper as training script.
    teacher_source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_full_teacher")
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

    # flip x-axis coordinates of source_concat_adata UMAP
    source_concat_adata.obsm['X_umap'][:, 0] = -source_concat_adata.obsm['X_umap'][:, 0]

    # plot teacher source UMAPs
    teacher_source_umap_colors = ['modality', 'RNA_clusters', 'ATAC_clusters']
    fig, axs = plt.subplots(1, len(teacher_source_umap_colors), figsize=(18, 5))
    for i, color in enumerate(teacher_source_umap_colors):
        sc.pl.umap(teacher_source_concat_adata, color=color, ncols=3, wspace=0.1, size=25, ax=axs[i], show=False)
        axs[i].set_xlabel(''); axs[i].set_ylabel('')
    plt.tight_layout()
    save_figure_pdf_vector_raster(
        fig,
        os.path.join(
            "/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures",
            "teacher_source_concat_umap.pdf",
        ),
    )
    plt.show()

    # plot source UMAPs
    source_umap_colors = ['modality', 'RNA_clusters', 'ATAC_clusters']
    fig, axs = plt.subplots(1, len(source_umap_colors), figsize=(18, 5))
    for i, color in enumerate(source_umap_colors):
        sc.pl.umap(source_concat_adata, color=color, ncols=3, wspace=0.1, size=25, ax=axs[i], show=False)
        axs[i].set_xlabel(''); axs[i].set_ylabel('')
    plt.tight_layout()
    save_figure_pdf_vector_raster(
        fig,
        os.path.join(
            "/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures",
            "source_concat_umap.pdf",
        ),
    )
    plt.show()

    # plot nonspatial source UMAPs
    nonspatial_source_umap_colors = ['modality', 'RNA_clusters', 'ATAC_clusters']
    fig, axs = plt.subplots(1, len(nonspatial_source_umap_colors), figsize=(18, 5))
    for i, color in enumerate(nonspatial_source_umap_colors):
        sc.pl.umap(nonspatial_source_concat_adata, color=color, ncols=3, wspace=0.1, size=25, ax=axs[i], show=False)
        axs[i].set_xlabel(''); axs[i].set_ylabel('')
    plt.tight_layout()
    save_figure_pdf_vector_raster(
        fig,
        os.path.join(
            "/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures",
            "nonspatial_source_concat_umap.pdf",
        ),
    )
    plt.show()

    if args.target_model:

        ## student target inference
        target_rna_emb, target_atac_emb, _ = run_inference(
            target_mgate, target_graph_tf, target_gp_tf, target_x1, target_x2, device
        )
        set_multigate_embeddings(target_rna, target_atac, target_rna_emb, target_atac_emb)
        print("  Target embeddings: shape {}".format(target_rna_emb.shape))

        target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key="MultiGATE")

        ## subsample target data to test obs
        test_target_obs_names = split_metadata.get('domains').get('target').get('splits').get('test').get('obs_names')
        target_concat_adata = target_concat_adata[target_concat_adata.obs_names.str.split('_').str[0].isin(test_target_obs_names)].copy()

        sc.pp.neighbors(target_concat_adata, n_neighbors=30)
        sc.tl.umap(target_concat_adata, min_dist=0.1, spread=1.5)
        sc.tl.leiden(target_concat_adata, resolution=leiden_resolution)

        # plot target UMAPs
        target_umap_colors = ['modality', 'celltypist_predictions']
        fig, axs = plt.subplots(1, len(target_umap_colors), figsize=(12, 5))
        for i, color in enumerate(target_umap_colors):
            ax = axs[i]
            sc.pl.umap(target_concat_adata, color=color, size=25, ax=ax, show=False)
            ax.set_xlabel(''); ax.set_ylabel('')
            
            # Force legend into a single column
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Only modify if a legend exists
                ax.legend(
                    handles=handles,
                    labels=labels,
                    loc='center left',
                    bbox_to_anchor=(1, 0.5),
                    ncol=1,
                    frameon=False,
                )
        plt.tight_layout()
        save_figure_pdf_vector_raster(
            fig,
            os.path.join(
                "/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures",
                "target_test_concat_umap.pdf",
            ),
        )
        plt.show()

    #%% Inference, all same model (except nonspatial source inference)

    model = source_mgate

    ## (teacher) source inference
    teacher_source_rna_emb, teacher_source_atac_emb, teacher_source_att_lgp = run_inference(
        model, source_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, teacher_source_rna_emb, teacher_source_atac_emb, key_added="MultiGATE_teacher")
    print("  Teacher source embeddings: shape {}".format(teacher_source_rna_emb.shape))
    
    ## (student) source inference
    source_rna_emb, source_atac_emb, source_att_lgp = run_inference(
        model, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, source_rna_emb, source_atac_emb)
    print("  Source embeddings: shape {}".format(source_rna_emb.shape))

    ## (target) inference
    target_rna_emb, target_atac_emb, target_att_lgp = run_inference(
        model, target_graph_tf, target_gp_tf, target_x1, target_x2, device
    )
    set_multigate_embeddings(target_rna, target_atac, target_rna_emb, target_atac_emb)
    print("  Target embeddings: shape {}".format(target_rna_emb.shape))

    ## nonspatial source inference
    nonspatial_source_rna_emb, nonspatial_source_atac_emb, nonspatial_source_att_lgp = run_inference(
        nonspatial_source_mgate, source_infer_graph_tf, source_gp_tf, source_x1, source_x2, device
    )
    set_multigate_embeddings(source_rna, source_atac, nonspatial_source_rna_emb, nonspatial_source_atac_emb, key_added="MultiGATE_nonspatial")
    print("  Nonspatial source embeddings: shape {}".format(nonspatial_source_rna_emb.shape))

    #%% OT from target to source
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

    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key="MultiGATE_source_aligned")
    source_concat_adata.obs = source_concat_adata.obs.assign(source_obs_names = source_concat_adata.obs_names)
    source_concat_adata.obsm['spatial'] = np.concatenate([source_rna.obsm['spatial'], source_atac.obsm['spatial']], axis=0)

    source_target_adata, _, _ = run_alignment_and_spatial_plot(
        model,
        source_mgate,
        target_mgate,
        source_rna,
        source_atac,
        source_concat_adata,
        target_concat_adata,
        deterministic_seed,
    )

    ## transfer UMAP, spatial coordinates and ingest-transferred variables to original datasets
    source_rna.obsm['X_umap'] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['source', 'rna']).all(1)].obsm['X_umap'].copy()
    source_atac.obsm['X_umap'] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['source', 'atac']).all(1)].obsm['X_umap'].copy()

    source_rna.obs[['leiden', 'source_obs_names']] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['source', 'rna']).all(1)].obs[['leiden', 'source_obs_names']].values.copy()
    source_atac.obs[['leiden', 'source_obs_names']] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['source', 'atac']).all(1)].obs[['leiden', 'source_obs_names']].values.copy()

    target_rna.obsm['X_umap'] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['target', 'rna']).all(1)].obsm['X_umap'].copy()
    target_atac.obsm['X_umap'] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['target', 'atac']).all(1)].obsm['X_umap'].copy()

    target_rna.obsm['spatial'] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['target', 'rna']).all(1)].obsm['spatial'].copy()
    target_atac.obsm['spatial'] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['target', 'atac']).all(1)].obsm['spatial'].copy()

    target_rna.obs[['leiden', 'source_obs_names']] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['target', 'rna']).all(1)].obs[['leiden', 'source_obs_names']].values.copy()
    target_atac.obs[['leiden', 'source_obs_names']] = source_target_adata[source_target_adata.obs[['source_or_target', 'modality']].eq(['target', 'atac']).all(1)].obs[['leiden', 'source_obs_names']].values.copy()

    ## get top genes per leiden cluster
    sc.tl.rank_genes_groups(
        source_rna,
        groupby="leiden",
        method="t-test",   # or "t-test", "logreg"
        reference="rest"
    )
    source_rna_rank_genes_leidens_df = sc.get.rank_genes_groups_df(source_rna, group=None)
    
    sc.tl.rank_genes_groups(
        target_rna,
        groupby="leiden",
        method="t-test",   # or "t-test", "logreg"
        reference="rest"
    )
    target_rna_rank_genes_leidens_df = sc.get.rank_genes_groups_df(target_rna, group=None)

    ## merge source and target rank genes leidens dataframes, only keep significant genes
    merged_rank_genes_leidens_df = pd.merge(
        source_rna_rank_genes_leidens_df.drop(columns=['scores', 'logfoldchanges', 'pvals']),
        target_rna_rank_genes_leidens_df.drop(columns=['scores', 'logfoldchanges', 'pvals']),
        on=["group", "names"],
        how="outer",
        suffixes=("_source", "_target"),
    )
    prop_signif_per_leiden = merged_rank_genes_leidens_df.drop(columns=['group','names']).le(0.05).mean(axis=0)
    print(prop_signif_per_leiden)

    merged_rank_genes_leidens_df = merged_rank_genes_leidens_df[(
        merged_rank_genes_leidens_df['pvals_adj_source'].le(0.05) &
        merged_rank_genes_leidens_df['pvals_adj_target'].le(0.05)
    )]
    rank_genes_leidens_corr = merged_rank_genes_leidens_df.groupby('group')[['pvals_adj_source', 'pvals_adj_target']].corr(method='spearman')
    rank_genes_leidens_corr = rank_genes_leidens_corr.groupby(level=0).apply(lambda df: df.iloc[0, 1])

    # Prepare the data: rank-transform and then -log (as in original code)
    ranked_df = merged_rank_genes_leidens_df[['pvals_adj_source', 'pvals_adj_target']].apply(lambda x: -np.log(x)).rank()
    ranked_df['leiden'] = merged_rank_genes_leidens_df['group']

    import seaborn as sns
    from matplotlib.contour import QuadContourSet

    # Compatibility for seaborn 0.12.x with matplotlib >=3.10, where QuadContourSet
    # no longer exposes the collections attribute that seaborn labels internally.
    if not hasattr(QuadContourSet, 'collections'):
        QuadContourSet.collections = property(lambda self: [self])

    leiden_order = sorted(ranked_df['leiden'].dropna().unique())
    leiden_palette = dict(zip(leiden_order, sns.color_palette(n_colors=len(leiden_order))))
    from matplotlib.lines import Line2D

    g = sns.jointplot(
        data=ranked_df,
        x='pvals_adj_source',
        y='pvals_adj_target',
        kind="hex",
        joint_kws={
            'gridsize': 10,   # smaller number = larger hex bins
            'mincnt': 3,
            'cmap': 'Greys',
            'alpha': 0.5,
        },
    )
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()
    for leiden, leiden_df in ranked_df.groupby('leiden'):
        sns.regplot(
            data=leiden_df,
            x='pvals_adj_source',
            y='pvals_adj_target',
            ax=g.ax_joint,
            scatter=False,
            ci=None,
            color=leiden_palette[leiden],
            lowess=True,
            truncate=False,
            line_kws={'linewidth': 3, 'alpha': 1., 'zorder': 3},
        )
    legend_handles = [
        Line2D([0], [0], color=leiden_palette[leiden], lw=3, label=leiden)
        for leiden in leiden_order
    ]
    g.ax_joint.legend(
        handles=legend_handles,
        title='leiden',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    g.fig.subplots_adjust(right=0.78)
    plt.gcf().set_size_inches(6, 4.5)
    plt.xlabel('source gene rank'); plt.ylabel('target gene rank')
    plt.tight_layout(); plt.show()

    ## plot correlations per leiden as a stemplot
    plt.figure(figsize=(5.5, 2.5))
    x = rank_genes_leidens_corr.index
    y = rank_genes_leidens_corr.values

    # Calculate new vmax (upper y-limit) by increasing the max y by 0.01
    ymax = y.max() + 0.03

    markerline, stemlines, baseline = plt.stem(
        x, y, linefmt='grey', markerfmt='o', bottom=0.)
    markerline.set_visible(True)
    marker_colors = [leiden_palette[leiden] for leiden in x]
    plt.scatter(
        x,
        y,
        marker='o',
        s=80,
        facecolors=marker_colors,
        edgecolors='black',
        linewidths=1.5,
        zorder=3,
    )
    plt.xlabel('leiden'); plt.ylabel('Spearman corr.')
    plt.ylim(plt.ylim()[0], ymax)
    plt.tight_layout(); plt.show()

    ## combine p-values and get top genes per leiden cluster
    from scipy.stats import combine_pvalues
    merged_rank_genes_leidens_df['combined_pvals'] = merged_rank_genes_leidens_df.apply(
        lambda row: combine_pvalues([row['pvals_adj_source'], row['pvals_adj_target']], method='pearson')[1], axis=1)
    top_genes_per_leiden = merged_rank_genes_leidens_df.groupby('group').apply(
        lambda group: group.nsmallest(1, 'combined_pvals', keep='all').loc[:,['names','combined_pvals']]
    )
    print(top_genes_per_leiden)

    ## plot top genes per leiden cluster on UMAP
    marker_leiden_genes = top_genes_per_leiden.names.unique()
    striatal_markers = ['Pde10a', 'Rgs9', 'Gng7']
    wm_markers = ['Mobp', 'Mal']
    striatal_score_name = ','.join(striatal_markers)
    wm_score_name = ','.join(wm_markers)
    assert np.isin(striatal_markers, marker_leiden_genes).all() and np.isin(wm_markers, marker_leiden_genes).all()
    sc.tl.score_genes(source_rna, gene_list=striatal_markers, score_name=striatal_score_name)
    sc.tl.score_genes(source_rna, gene_list=wm_markers, score_name=wm_score_name)
    sc.tl.score_genes(target_rna, gene_list=striatal_markers, score_name=striatal_score_name)
    sc.tl.score_genes(target_rna, gene_list=wm_markers, score_name=wm_score_name)

    sc.pl.embedding(source_rna, color=[striatal_score_name, wm_score_name], basis='spatial', ncols=2, wspace=0.05, size=60)
    sc.pl.embedding(target_rna, color=[striatal_score_name, wm_score_name], basis='spatial', ncols=2, wspace=0.05, size=60)

    '''
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
    '''

    ## [COMMENTED OUT] write to disk
    # for keys in (source_rna.obsm.keys(), source_atac.obsm.keys(), target_rna.obsm.keys(), target_atac.obsm.keys()):
    #     assert 'MultiGATE_source_aligned' in keys, f"MultiGATE_source_aligned not found in {keys}"
    #source_rna.write_h5ad(os.path.join(base_path, "source_rna_aligned_with_latents.h5ad"))
    #source_atac.write_h5ad(os.path.join(base_path, "source_atac_aligned_with_latents.h5ad"))
    #target_rna.write_h5ad(os.path.join(base_path, "target_rna_aligned_with_latents.h5ad"))
    #target_atac.write_h5ad(os.path.join(base_path, "target_atac_aligned_with_latents.h5ad"))

    ## [COMMENTED OUT] load from disk
    #source_rna = sc.read_h5ad(os.path.join(base_path, "source_rna_aligned_with_latents.h5ad"))
    #source_atac = sc.read_h5ad(os.path.join(base_path, "source_atac_aligned_with_latents.h5ad"))
    #target_rna = sc.read_h5ad(os.path.join(base_path, "target_rna_aligned_with_latents.h5ad"))
    #target_atac = sc.read_h5ad(os.path.join(base_path, "target_atac_aligned_with_latents.h5ad"))

    #%% compute scib metrics for OT-aligned and nonspatial data, and plot model metrics


    #  ── Extract logged model metrics ──────────────────────────────────────
    model_metrics_df = extract_logged_model_metrics(client, run_id, run_name=args.run_name)
    model_metrics_df = model_metrics_df.loc[model_metrics_df['step'].eq(model_metrics_df['step'].max())]
    model_metrics_df.drop(columns=['run_id', 'run_name', 'timestamp', 'logged_at', 'step'], inplace=True)
    # columns now: metric_group, domain, metric_name, value

    ## scib metrics of full teacher data
    teacher_source_scib_metrics = compute_scib_metrics_for_domain(
        rna_adata=source_rna,
        atac_adata=source_atac,
        domain_name="source",
        label_key="RNA_clusters",
        scib_n_jobs=1,
        embedding_key="MultiGATE_full_teacher",
    )
    ## scib metrics of source data
    source_scib_metrics = compute_scib_metrics_for_domain(
        rna_adata=source_rna,
        atac_adata=source_atac,
        domain_name="source",
        label_key="RNA_clusters",
        scib_n_jobs=1,
        embedding_key="MultiGATE_source_aligned",
    )
    ## scib metrics of target data (non-OT-aligned)
    target_scib_metrics = compute_scib_metrics_for_domain(
        rna_adata=target_rna,
        atac_adata=target_atac,
        domain_name="target",
        label_key="celltypist_predictions",
        scib_n_jobs=1,
        embedding_key="MultiGATE",
    )
    ## scib metrics of OT-aligned data
    target_ot_scib_metrics = compute_scib_metrics_for_domain(
        rna_adata=target_rna,
        atac_adata=target_atac,
        domain_name="target",
        label_key="celltypist_predictions",
        scib_n_jobs=1,
        embedding_key="MultiGATE_source_aligned",
    )
    ## scib metrics of nonspatial data
    source_nonspatial_scib_metrics = compute_scib_metrics_for_domain(
        rna_adata=source_rna,
        atac_adata=source_atac,
        domain_name="source",
        label_key="RNA_clusters",
        scib_n_jobs=1,
        embedding_key="MultiGATE_nonspatial",
    )

    ## build tidy DataFrames for freshly computed scib metrics
    _SCIB_NUMERIC_KEYS = {"silhouette_label", "ilisi", "bras", "bio_conservation", "batch_correction", "total"}

    def _scib_metrics_to_df(metrics_dict, domain):
        rows = [
            {"domain": domain, "metric_name": k, "value": v}
            for k, v in metrics_dict.items()
            if k in _SCIB_NUMERIC_KEYS
        ]
        return pd.DataFrame(rows, columns=["domain", "metric_name", "value"])

    teacher_source_scib_metrics_df = _scib_metrics_to_df(teacher_source_scib_metrics, domain="teacher_source")
    source_scib_metrics_df = _scib_metrics_to_df(source_scib_metrics, domain="source")
    target_scib_metrics_df = _scib_metrics_to_df(target_scib_metrics, domain="target")
    target_ot_scib_metrics_df = _scib_metrics_to_df(target_ot_scib_metrics, domain="target_ot")
    source_nonspatial_scib_metrics_df = _scib_metrics_to_df(source_nonspatial_scib_metrics, domain="source_nonspatial")

    # model_metrics_df already has 'domain' and 'metric_name' (stripped) from
    # extract_logged_model_metrics / parse_metric_domain_and_name.
    # metric_group is only present in the MLflow rows; fill NaN for the new rows.
    model_metrics_df = pd.concat(
        #[model_metrics_df, teacher_source_scib_metrics_df, target_ot_scib_metrics_df, source_nonspatial_scib_metrics_df],
        [teacher_source_scib_metrics_df, source_scib_metrics_df, target_scib_metrics_df, target_ot_scib_metrics_df, source_nonspatial_scib_metrics_df],
        ignore_index=True,
    )

    model_metrics_df = model_metrics_df.loc[model_metrics_df['metric_name'].isin(['silhouette_label', 'ilisi', 'bio_conservation', 'total'])]
    model_metrics_df['metric_name'] = model_metrics_df['metric_name'].replace({'bio_conservation': 'bio_cons.', 'silhouette_label': 'silhouette'})

    #ax = sns.barplot(data=model_metrics_df, y='metric_name', x='value', hue='domain', orient='h')
    #ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=1, borderaxespad=0.)
    #plt.gcf().set_size_inches(4, 5)
    #ax.set_ylabel('')

    ax = sns.barplot(data=model_metrics_df, y='value', x='metric_name', hue='domain', palette='Set2', orient='v')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=2, borderaxespad=0.)
    plt.gcf().set_size_inches(5, 7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_xlabel(''); ax.set_ylabel('')    
    
    # Place legend on top of the plot
    plt.tight_layout()
    save_figure_pdf_vector_raster(
        plt.gcf(),
        os.path.join(
            "/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures",
            "scib_metrics_barplot.pdf",
        ),
    )
    plt.show()


    #%% LIANA+ inflow analysis
    import plotnine as p9
    import gc
    from sklearn.model_selection import StratifiedShuffleSplit

    def liana_spatial_analysis(
        adata,
        subsample_n=5000,
        resource=None,
        spatial_key="spatial",
        cell_type_col="RNA_clusters",
        labels=["R2", "R4", "R7"], interaction=None, ncomps=30, bandwidth=40, s=60
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
            resource = li.rs.select_resource('mouseconsensus') # NOTE: mouse_consensus could be used in future
            '''
            resource = li.rs.translate_resource(
                resource,
                map_df=map_df,
                columns=['ligand', 'receptor'],
                replace=True,
                one_to_many=2,
            )
            '''

        lrdata = li.mt.inflow(
            adata,
            groupby=cell_type_col,
            resource=resource,
            use_raw=False,
        )

        fused_I = None
        if interaction is not None:

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
            "lr_loadings": lr_loadings,
            "factor_scores": factor_scores
        }

    '''
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
    '''

    ## format combined_gp_dict to adapt to LIANA+
    resource = li.rs.select_resource('mouseconsensus')

    ## ingest RNA_cluster assign RNA_clusters to target_rna
    sc.tl.ingest(target_concat_adata, source_concat_adata, embedding_method='umap', obs='RNA_clusters', inplace=True)

    ingested_rna_clusters = target_concat_adata.obs.loc[
        target_concat_adata.obs['modality'].eq('rna'), 'RNA_clusters'
    ]
    ingested_rna_clusters.index = ingested_rna_clusters.index.str.replace('_rna', '')
    target_rna.obs['RNA_clusters'] = ingested_rna_clusters

    ## set SCT_data as default assay
    source_rna_liana = source_rna.copy()
    source_rna_liana.X = source_rna_liana.layers['SCT_data'].copy()
    target_rna_liana = target_rna.copy()
    target_rna_liana.X = target_rna_liana.layers['SCT_data'].copy()

    ## LIANA+ inflow analysis
    source_liana_results = liana_spatial_analysis(
        source_rna_liana,
        subsample_n=5000,
        resource=resource,
        labels=None, interaction=None, ncomps=30, bandwidth=40, s=60, cell_type_col="RNA_clusters", spatial_key="spatial"
        )

    target_liana_results = liana_spatial_analysis(
        target_rna_liana,
        subsample_n=5000,
        resource=resource,
        labels=None, interaction=None, ncomps=30, bandwidth=40, s=60, cell_type_col="RNA_clusters", spatial_key="spatial"
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
    cg = sns.clustermap(corr_top, cmap='coolwarm', center=0.0, vmax=0.4, figsize=(5, 3))
    cg.ax_heatmap.set_xlabel('MultiGATE dimensions')
    plt.show()

    save_plot_to_pdf(cg.fig, "liana_multigate_heatmap")
    plt.close(cg.fig)

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
    }, index=['Source', 'Target', 'Nsp Source', 'Nsp Source$^\dagger$'])
    ari_nmi_df.plot(kind='bar', rot=30, cmap='Set2', figsize=(4.5, 3)); plt.show()

    ## plot UMAP of source, target, nonspatial-source and ingested nonspatial-source
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    sc.pl.umap(source_concat_adata, color='leiden', size=25, ax=axs[0, 0], show=False)
    sc.pl.umap(target_concat_adata, color='leiden', size=25, ax=axs[0, 1], show=False)
    sc.pl.umap(nonspatial_source_concat_adata, color='leiden', size=25, ax=axs[1, 0], show=False)
    sc.pl.umap(ingested_nonspatial_source_concat_adata, color='leiden', size=25, ax=axs[1, 1], show=False)
    plt.tight_layout()
    plt.show()

    #%% PLS embedding-to-gene modelling
    from sklearn.cross_decomposition import PLSRegression
    import decoupler as dc
    import decoupler.op
    import liana as li

    def save_plot_to_pdf(fig, filename, save_dir="/home/mcb/users/dmannk/THESIS_base/overleaf-cibb-2026/figures/pls_analysis", dpi=300):
        """Save figure as a fast-rendering PDF with heavy plot layers rasterized."""
        filepath = os.path.join(save_dir, f"{filename}.pdf")
        save_figure_pdf_vector_raster(fig, filepath, dpi=dpi)
        print(f"Saved: {filepath}")
        plt.close(fig)

    def run_pls_embedding_to_gene_plots(
        adata, full_adata=None, umap_title=None, save_plots=False, plot_prefix='',
        umap_plot_prefix=None, spatial_leiden_plot_prefix=None,
        n_components=9, basis='spatial', weights_plot_type='staircase', top_n_genes=5, heatmap_cmap='YlGn',
        censor_gene_from_plot=None
    ):
        # X = pathway_embedding_results['source_rna'].pathway_scores.to_numpy()
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        Z = adata.obsm['MultiGATE_source_aligned'].copy()

        pls = PLSRegression(n_components=n_components)  
        pls.fit(Z, X)
        T = pls.transform(Z)

        ## plot PLS loadings onto embedding
        pls_cols = [f'PLS_{i}' for i in range(T.shape[1])]
        pls_df = pd.DataFrame(T, index=adata.obs_names, columns=pls_cols)

        if full_adata is not None:
            # Reindex onto the full cell set; cells filtered out get NaN and are
            # rendered as na_color ('lightgray') to form a spatial background.
            full_obs = full_adata.obs.copy().join(pls_df, how='left')
            adata_pls = sc.AnnData(
                X=np.zeros((full_adata.n_obs, 1)),
                obs=full_obs,
                obsm={'spatial': full_adata.obsm['spatial'].copy()},
            )
        else:
            adata_pls = adata.copy()
            adata_pls.obs = adata_pls.obs.merge(pls_df, left_index=True, right_index=True)

        if 'X_umap' in adata.obsm and 'leiden' in adata.obs:
            sc.pl.umap(adata_pls, color='leiden', size=25, show=False)
            if umap_title is not None:
                plt.gca().set_title(umap_title)
            plt.tight_layout()
            fig = plt.gcf()
            if save_plots:
                save_prefix = umap_plot_prefix or plot_prefix
                save_plot_to_pdf(fig, f"{save_prefix}_liana_umap_leiden")
            plt.show()

        if 'spatial' in adata_pls.obsm and 'leiden' in adata_pls.obs:
            sc.pl.embedding(
                adata_pls,
                basis='spatial',
                color='leiden',
                size=60,
                show=False,
                na_color='darkgray',
            )
            fig = plt.gcf()
            plt.tight_layout()
            if save_plots:
                save_prefix = spatial_leiden_plot_prefix or plot_prefix
                save_plot_to_pdf(fig, f"{save_prefix}_spatial_leiden")
            plt.show()

        sc.pl.embedding(
            adata_pls,
            basis=basis,  # 'spatial' or 'X_seurat_umap' or 'X_umap
            color=adata_pls.obs.columns[adata_pls.obs.columns.str.startswith('PLS_')],
            ncols=3, wspace=0.2, size=25, show=False, na_color='darkgray'
        )
        fig = plt.gcf()
        fig.set_size_inches(12, 10)  # or any desired figsize (width, height)
        for ax in fig.get_axes():
            ax.set_xlabel('')
            ax.set_ylabel('')
        plt.tight_layout()
        if save_plots:
            save_plot_to_pdf(fig, f"{plot_prefix}_pls_embedding_spatial")
        plt.show()

        ## plot heatmap of PLS coef
        pls_coef_df = pd.DataFrame(
            pls.y_weights_.T,
            index=[f'PLS_{i}' for i in range(pls.y_weights_.shape[1])],
            columns=adata.var_names
        )
        if weights_plot_type == 'staircase':
            top_genes_sort = pls_coef_df.abs().apply(
                lambda x: x.nlargest(top_n_genes).index, axis=1, result_type='reduce'
            ).explode().unique()
            pls_coef_df_sorted = pls_coef_df.loc[:, top_genes_sort]
            pls_coef_df_sorted_plot = pls_coef_df_sorted.copy()
            if censor_gene_from_plot is not None:
                pls_coef_df_sorted_plot = pls_coef_df_sorted_plot.loc[:, ~pls_coef_df_sorted_plot.columns.isin(censor_gene_from_plot)]

            fig = plt.figure(figsize=(4.25, 9))
            sns.heatmap(pls_coef_df_sorted_plot.T.abs(), cmap=heatmap_cmap, cbar=True)
            plt.tight_layout()
            if save_plots:
                save_plot_to_pdf(fig, f"{plot_prefix}_pls_weights_heatmap_staircase")
            plt.show()

        elif weights_plot_type == 'barplot':

            fig, ax = plt.subplots(figsize=(6, 4))
            n_components = len(pls_coef_df)
            n_genesets = len(pls_coef_df.columns)
            bar_width = 0.8 / n_genesets
            
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            geneset_colors = {col: colors[i % len(colors)] for i, col in enumerate(pls_coef_df.columns)}
            
            for i, (idx, row) in enumerate(pls_coef_df.iterrows()):
                sorted_row = row.sort_values(ascending=False)
                for j, (geneset, val) in enumerate(sorted_row.items()):
                    pos = i - 0.4 + (j + 0.5) * bar_width
                    ax.bar(pos, val, width=bar_width, color=geneset_colors[geneset])
            
            ax.set_xticks(range(n_components))
            ax.set_xticklabels(pls_coef_df.index, rotation=30)
            ax.set_ylabel('PLS weight')
            
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=geneset_colors[col], label=col) for col in pls_coef_df.columns]
            ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
            plt.tight_layout()
            if save_plots:
                save_plot_to_pdf(fig, f"{plot_prefix}_pls_weights_barplot")
            plt.show()

            pls_coef_df_sorted = None

        return pls, pls_coef_df, pls_coef_df_sorted

    ## train PLS models on source and target data
    pls_source_rna, pls_source_rna_coef_df, _ = run_pls_embedding_to_gene_plots(source_rna, save_plots=False, plot_prefix='01_source_rna_pls')
    pls_target_rna, pls_target_rna_coef_df, _ = run_pls_embedding_to_gene_plots(target_rna, save_plots=False, plot_prefix='02_target_rna_pls', censor_gene_from_plot=['C530008M17Rik'])
    pls_source_atac, pls_source_atac_coef_df, _ = run_pls_embedding_to_gene_plots(source_atac, save_plots=False, plot_prefix='03_source_atac_pls')
    pls_target_atac, pls_target_atac_coef_df, _ = run_pls_embedding_to_gene_plots(target_atac, save_plots=False, plot_prefix='04_target_atac_pls')

    ## plot correlation between source and target PLS dimensions
    fig, ax = plt.subplots(2, 1, figsize=(4, 8))
    pls_rna_corr = np.corrcoef(pls_source_rna.y_weights_, pls_target_rna.y_weights_, rowvar=False)
    pls_rna_corr = pls_rna_corr[:pls_source_rna.y_weights_.shape[1], pls_target_rna.y_weights_.shape[1]:]
    sns.heatmap(np.abs(pls_rna_corr), cmap='YlGn', cbar=True, ax=ax[0])
    ax[0].set_xlabel('Target PLS dimensions')
    ax[0].set_ylabel('Source PLS dimensions')
    ax[0].set_title('RNA')

    pls_atac_corr = np.corrcoef(pls_source_atac.y_weights_, pls_target_atac.y_weights_, rowvar=False)
    pls_atac_corr = pls_atac_corr[:pls_source_atac.y_weights_.shape[1], pls_target_atac.y_weights_.shape[1]:]
    sns.heatmap(np.abs(pls_atac_corr), cmap='YlGn', cbar=True, ax=ax[1])
    ax[1].set_xlabel('Target PLS dimensions')
    ax[1].set_ylabel('Source PLS dimensions')
    ax[1].set_title('ATAC')
    fig.tight_layout()
    plt.show()
    save_plot_to_pdf(fig, '11_pls_source_target_correlation_heatmaps')

    def get_liana_lrdata(adata, cell_type_col, lr_pairs):

        adata.X = adata.layers["SCT_data"].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        li.ut.spatial_neighbors(adata=adata, bandwidth=40, spatial_key='spatial')
        sq.gr.spatial_autocorr(adata, mode='moran', use_raw=False, show_progress_bar=True)
        svgs = adata.uns['moranI'].index[(adata.uns['moranI']['pval_norm_fdr_bh'] < 0.05) & (adata.uns['moranI']['I'] > 0.01)]
        adata = adata[:, svgs]

        lrdata = li.mt.inflow(
            adata,
            groupby=cell_type_col,
            resource = lr_pairs,
            use_raw=False,
        )
        lrdata.var['interaction'] = lrdata.var_names.str.split('^').to_series().apply(lambda x: f'{x[1]}^{x[2]}').values
        return lrdata

    ## build shared Hallmark mouse net once
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

    lr_pairs = li.rs.select_resource('mouseconsensus')
    
    lr_geneset = li.rs.generate_lr_geneset(
        lr_pairs,
        hallmark_mouse_net,
        lr_sep="^",
        weight=None,
    )

    ## build LR data and train PLS models on LR data
    lrdata_source_rna = get_liana_lrdata(source_rna, 'leiden', lr_pairs)
    lrdata_target_rna = get_liana_lrdata(target_rna, 'leiden', lr_pairs)
    pls_source_rna_liana, pls_source_rna_liana_coef_df, _ = run_pls_embedding_to_gene_plots(lrdata_source_rna, save_plots=True, plot_prefix='05_source_rna_liana_pls')
    pls_target_rna_liana, pls_target_rna_liana_coef_df, _ = run_pls_embedding_to_gene_plots(lrdata_target_rna, save_plots=True, plot_prefix='06_target_rna_liana_pls')

    def build_lrdata_aggregated_by_genesets(lrdata, unique_geneset_per_lr, umap_title=None, save_plots=False, plot_prefix=''):
        """Aggregate LIANA inflow by hallmark geneset; mutates lrdata.var with geneset columns."""
        lrdata.var = lrdata.var.merge(
            unique_geneset_per_lr, left_on='interaction', right_index=True, how='left'
        )
        geneset_cols = unique_geneset_per_lr.columns
        geneset_cols = list(geneset_cols[geneset_cols.isin(lrdata.var.columns)])
        liana_by_geneset = lrdata.var[geneset_cols].fillna(0)
        if not liana_by_geneset.values.any():
            print('No expression in LIANA genesets, returning None')
            return None
        
        X_genesets = lrdata.X @ sp.csr_matrix(liana_by_geneset.values)
        lrdata_genesets = sc.AnnData(
            X=X_genesets,
            obs=lrdata.obs,
            var=pd.DataFrame(index=liana_by_geneset.columns),
            obsm={'spatial': lrdata.obsm['spatial'].copy()},
        )
        sc.pp.filter_cells(lrdata_genesets, min_genes=3)
        sc.pp.filter_genes(lrdata_genesets, min_cells=2)

        sc.pp.neighbors(lrdata_genesets, use_rep='X', metric='cosine')
        sc.tl.umap(lrdata_genesets)
        sc.tl.leiden(lrdata_genesets, resolution=0.03)

        lrdata_genesets.obsm['MultiGATE_source_aligned'] = lrdata[
            lrdata_genesets.obs_names
        ].obsm['MultiGATE_source_aligned'].copy()
        lrdata_genesets.obsm['spatial'] = lrdata[lrdata_genesets.obs_names].obsm['spatial'].copy()
        return lrdata_genesets

    unique_geneset_per_lr = lr_geneset.groupby('interaction')['source'].unique().map(list).rename('genesets')
    unique_geneset_per_lr = unique_geneset_per_lr.explode().str.get_dummies().groupby(level=0).max()

    # Capture full spatial layout before filter_cells removes any cells, so
    # filtered-out cells can be rendered as a gray background in spatial plots.
    full_spatial_obsm_df_source = sc.AnnData(
        X=np.zeros((lrdata_source_rna.n_obs, 1)),
        obs=lrdata_source_rna.obs.copy(),
        obsm={'spatial': lrdata_source_rna.obsm['spatial'].copy()},
    )
    full_spatial_obsm_df_target = sc.AnnData(
        X=np.zeros((lrdata_target_rna.n_obs, 1)),
        obs=lrdata_target_rna.obs.copy(),
        obsm={
            'spatial': lrdata_target_rna.obsm['spatial'].copy(),
            'X_umap': lrdata_target_rna.obsm['X_umap'].copy(),
        },
    )

    lrdata_source_rna_genesets = build_lrdata_aggregated_by_genesets(
        lrdata_source_rna, unique_geneset_per_lr
    )
    lrdata_target_rna_genesets = build_lrdata_aggregated_by_genesets(
        lrdata_target_rna, unique_geneset_per_lr
    )

    if lrdata_source_rna_genesets is not None:
        pls_lrdata_source_rna_genesets, _, _ = run_pls_embedding_to_gene_plots(
            lrdata_source_rna_genesets,
            None, # full_spatial_obsm_df_source,
            umap_title='source RNA LIANA (geneset aggregate)',
            umap_plot_prefix='07_source_rna_liana_geneset',
            weights_plot_type='barplot',
            save_plots=True,
            plot_prefix='09_source_rna_liana_geneset_pls',
        )
    if lrdata_target_rna_genesets is not None:
        pls_lrdata_target_rna_genesets, _, _ = run_pls_embedding_to_gene_plots(
            lrdata_target_rna_genesets,
            None, # full_spatial_obsm_df_target,
            umap_title='target RNA LIANA (geneset aggregate)',
            umap_plot_prefix='08_target_rna_liana_geneset',
            weights_plot_type='barplot',
            save_plots=True,
            plot_prefix='10_target_rna_liana_geneset_pls',
        )
    
    #%% attention matrix analysis
    from gene_peak_attention_utils import (
        plot_signed_arcs_stacked,
        save_gene_peak_links_bedpe,
    )
    from post_hoc_utils import gene_peak_attention_links_from_att_lgp

    '''
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

    ## load gene-peak attention links
    gene_peak_attention_links = pd.read_csv(os.path.join(attention_analysis_summary['output_dir'], 'merged_df_threshold.csv'))
    duplicate_rows = gene_peak_attention_links[gene_peak_attention_links.duplicated(subset=['Gene', 'Peak'], keep=False)]
    gene_peak_attention_links = gene_peak_attention_links.loc[gene_peak_attention_links['chr'].str.startswith('chr')]
    gene_peak_attention_links = gene_peak_attention_links.drop_duplicates(subset=['Gene', 'Peak', 'Attention', 'gene_idx', 'peak_idx'])
    assert gene_peak_attention_links[['Gene', 'Peak']].value_counts().le(1).all()
    '''

    source_gene_peak_attention_links, source_attention_analysis_summary = gene_peak_attention_links_from_att_lgp(
        source_att_lgp, source_rna, source_atac
    )
    target_gene_peak_attention_links, target_attention_analysis_summary = gene_peak_attention_links_from_att_lgp(
        target_att_lgp, target_rna, target_atac
    )

    ## find high PLS-weighted gene-peak attention links
    source_all_pls_scores_df, source_all_pls_scores_grouped_df = score_gene_peak_attention_links_by_pls(
        source_gene_peak_attention_links,
        pls_source_rna_coef_df,
        pls_source_atac_coef_df,
    )
    target_all_pls_scores_df, target_all_pls_scores_grouped_df = score_gene_peak_attention_links_by_pls(
        target_gene_peak_attention_links,
        pls_target_rna_coef_df,
        pls_target_atac_coef_df,
    )
    merged_pls_scores_grouped_df = pd.merge(
        source_all_pls_scores_grouped_df.head(20),
        target_all_pls_scores_grouped_df.head(20),
        on=['index','gene'], suffixes=('source','target')
    )

    print("Source PLS-weighted gene-peak attention links:")
    print(source_all_pls_scores_df.head(5))
    print(source_all_pls_scores_grouped_df.head(10))
    print("Target PLS-weighted gene-peak attention links:")
    print(target_all_pls_scores_df.head(5))
    print(target_all_pls_scores_grouped_df.head(10))

    ## select PLS-gene combinations
    pls_cmp = 'PLS_0'
    gene = 'Pde10a'

    # DORC plot and annotate the marker corresponding to 'gene'
    plt.figure(figsize=(2.5, 3))
    value_counts = source_gene_peak_attention_links['Gene'].value_counts(ascending=True)
    ax = value_counts.plot(marker='.', color='black')
    plt.xlabel('Genes')
    plt.ylabel('Number of peaks')
    if gene in value_counts.index:
        y = value_counts[gene]
        x = list(value_counts.index).index(gene)
        ax.plot(x, y, 'ro', markersize=8, label=gene)
        ax.annotate(gene, (x, y), xytext=(-25, 8), textcoords='offset points', ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    plt.show()
    assert source_gene_peak_attention_links['Gene'].value_counts(ascending=True).equals(target_gene_peak_attention_links['Gene'].value_counts(ascending=True))

    ## isolate gene-peak link df
    source_gp_link_df, source_peaks = select_gene_peak_link_df(
        source_gene_peak_attention_links,
        pls_source_rna_coef_df,
        pls_source_atac_coef_df,
        pls_cmp,
        gene,
    )
    target_gp_link_df, target_peaks = select_gene_peak_link_df(
        target_gene_peak_attention_links,
        pls_target_rna_coef_df,
        pls_target_atac_coef_df,
        pls_cmp,
        gene,
    )
    ## create 'peak' scores out of peak set
    source_peak_scores = source_atac[:, source_gp_link_df['Peak']].X.multiply(source_gp_link_df[f'{pls_cmp}_peak_weight']).sum(axis=1)
    target_peak_scores = target_atac[:, target_gp_link_df['Peak']].X.multiply(target_gp_link_df[f'{pls_cmp}_peak_weight']).sum(axis=1)
    source_atac.obs[f'{gene}-linked peaks'] = source_peak_scores
    target_atac.obs[f'{gene}-linked peaks'] = target_peak_scores

    ## spatial plots of selected gene-peak links
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    sc.pl.embedding(source_rna, basis='spatial', color=gene, size=60, ax=ax[0, 0], show=False)
    sc.pl.embedding(target_rna, basis='spatial', color=gene, size=60, ax=ax[0, 1], show=False)
    sc.pl.embedding(source_atac, basis='spatial', color=f'{gene}-linked peaks', size=60, vmax=0.08, ax=ax[1, 0], show=False)
    sc.pl.embedding(target_atac, basis='spatial', color=f'{gene}-linked peaks', size=60, vmax=0.10, ax=ax[1, 1], show=False)
    fig.suptitle(f'{gene} - {pls_cmp}')
    ax[0, 0].set_title('Source gene'); ax[0, 0].set_xlabel(''); ax[0, 0].set_ylabel('')
    ax[0, 1].set_title('Target gene'); ax[0, 1].set_xlabel(''); ax[0, 1].set_ylabel('')
    ax[1, 0].set_title('Source peaks'); ax[1, 0].set_xlabel(''); ax[1, 0].set_ylabel('')
    ax[1, 1].set_title('Target peaks'); ax[1, 1].set_xlabel(''); ax[1, 1].set_ylabel('')
    plt.tight_layout(); plt.show()

    ## save gene-peak links
    safe_pls_cmp = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(pls_cmp))
    safe_gene = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in str(gene))

    gp_link_prefix = f"source_gene_peak_links_{safe_pls_cmp}_{safe_gene}"
    gp_link_csv_path = os.path.join(source_attention_analysis_summary["output_dir"], f"{gp_link_prefix}.csv")
    source_bedpe = os.path.join(source_attention_analysis_summary["output_dir"], f"{gp_link_prefix}.bedpe")
    source_gp_link_df.to_csv(gp_link_csv_path, index=False)
    save_gene_peak_links_bedpe(source_gp_link_df, source_bedpe, score_col="Attention")

    gp_link_prefix = f"target_gene_peak_links_{safe_pls_cmp}_{safe_gene}"
    gp_link_csv_path = os.path.join(target_attention_analysis_summary["output_dir"], f"{gp_link_prefix}.csv")
    target_bedpe = os.path.join(target_attention_analysis_summary["output_dir"], f"{gp_link_prefix}.bedpe")
    target_gp_link_df.to_csv(gp_link_csv_path, index=False)
    save_gene_peak_links_bedpe(target_gp_link_df, target_bedpe, score_col="Attention")

    ## Plot signed-weight arcs (blue = positive {pls_cmp}_peak_weight, orange = negative;
    ## transparency scales with |peak_weight|). Produces per-domain figures plus a
    ## single stacked figure with source on top and target below, sharing genomic coordinates.
    signed_arcs_out_dir = f"{os.getenv('OUTPATH')}/MultiGATE/attention_analysis"
    source_signed_arcs_pdf = os.path.join(
        signed_arcs_out_dir, f"source_gene_peak_links_signed_{safe_pls_cmp}_{safe_gene}.pdf"
    )
    target_signed_arcs_pdf = os.path.join(
        signed_arcs_out_dir, f"target_gene_peak_links_signed_{safe_pls_cmp}_{safe_gene}.pdf"
    )
    stacked_signed_arcs_pdf = os.path.join(
        signed_arcs_out_dir, f"stacked_gene_peak_links_signed_{safe_pls_cmp}_{safe_gene}.pdf"
    )

    plot_signed_arcs_stacked(
        {"Source": source_gp_link_df},
        pls_cmp=pls_cmp,
        gene=gene,
        out_path=source_signed_arcs_pdf,
        title=f"{gene} — {pls_cmp} (source)",
    )
    plt.show()
    plot_signed_arcs_stacked(
        {"Target": target_gp_link_df},
        pls_cmp=pls_cmp,
        gene=gene,
        out_path=target_signed_arcs_pdf,
        title=f"{gene} — {pls_cmp} (target)",
    )
    plt.show()
    plot_signed_arcs_stacked(
        {"Source": source_gp_link_df, "Target": target_gp_link_df},
        pls_cmp=pls_cmp,
        gene=gene,
        out_path=stacked_signed_arcs_pdf,
        title=f"{gene} — {pls_cmp}",
    )
    plt.show()
    print(f"Saved signed-arc plots:\n {source_signed_arcs_pdf}\n {target_signed_arcs_pdf}\n {stacked_signed_arcs_pdf}")

    # Plot the saved BEDPE with CoolBox (run from MultiGATE/; env needs bgzip, tabix, pairix).
    # Need environment with CoolBox installed. (e.g. torch_env_py39)
    # Use the BEDPE path printed above as --links; pass a sorted, tabix-indexed .gtf.bgz as --gtf (or set $DATAPATH).
    # python scripts/plot_gene_peak_attention_links_track.py --links gp_link_bedpe_path --gtf "$DATAPATH/gene_annotations/gencode.vM25.chr_patch_hapl_scaff.annotation.sorted.gtf.bgz" --out "$OUTPATH/MultiGATE/attention_analysis/source_gene_peak_links.pdf"
    # Write a bash script to file, with commands for both source and target plotting

    plot_script_path = os.path.join(os.path.dirname(__file__), "plot_gene_peak_attention_links_track.py")
    bash_script_path = os.path.join(f"{os.getenv('OUTPATH')}/MultiGATE/attention_analysis", "run_plot_gene_peak_attention_links.sh")

    # Fill in these file paths with your actual .bedpe, .gtf.bgz (tabix-indexed), and output paths
    gtf_bgz = f"{os.getenv('DATAPATH')}/gene_annotations/gencode.vM25.chr_patch_hapl_scaff.annotation.sorted.gtf.bgz"
    out_dir = f"{os.getenv('OUTPATH')}/MultiGATE/attention_analysis"
    multigate_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    source_signed_csv = os.path.splitext(source_bedpe)[0] + ".csv"
    target_signed_csv = os.path.splitext(target_bedpe)[0] + ".csv"

    bash_script_contents = f"""#!/bin/bash

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_env_py39

# Plot source gene-peak links
python {plot_script_path} --links "{source_bedpe}" --gtf "{gtf_bgz}" --out "{out_dir}/source_gene_peak_links.pdf"

# Plot target gene-peak links
python {plot_script_path} --links "{target_bedpe}" --gtf "{gtf_bgz}" --out "{out_dir}/target_gene_peak_links.pdf"

# Signed-arc PDFs (matplotlib; same stem as the .bedpe/.csv above). Not produced by CoolBox.
cd {shlex.quote(multigate_root)}
python - <<'PY'
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from gene_peak_attention_utils import plot_signed_arcs_stacked
_src_csv = {repr(source_signed_csv)}
_tgt_csv = {repr(target_signed_csv)}
_pls = {repr(pls_cmp)}
_gene = {repr(gene)}
src = pd.read_csv(_src_csv)
tgt = pd.read_csv(_tgt_csv)
plot_signed_arcs_stacked({{"Source": src}}, pls_cmp=_pls, gene=_gene, out_path={repr(source_signed_arcs_pdf)}, title={repr(f"{gene} — {pls_cmp} (source)")})
plot_signed_arcs_stacked({{"Target": tgt}}, pls_cmp=_pls, gene=_gene, out_path={repr(target_signed_arcs_pdf)}, title={repr(f"{gene} — {pls_cmp} (target)")})
plot_signed_arcs_stacked({{"Source": src, "Target": tgt}}, pls_cmp=_pls, gene=_gene, out_path={repr(stacked_signed_arcs_pdf)}, title={repr(f"{gene} — {pls_cmp}")})
print("Saved signed-arc plots:")
print({repr(source_signed_arcs_pdf)})
print({repr(target_signed_arcs_pdf)})
print({repr(stacked_signed_arcs_pdf)})
PY
"""

    with open(bash_script_path, "w") as f:
        f.write(bash_script_contents)

    print(f"\nBash script written to:\n {bash_script_path}")
    print(f'Bash command to run:\n bash {bash_script_path}')

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


    #%% gene-set enrichment analysis
    import gseapy as gp

    def run_topic_gsea(topic_gene_weight_mat, gene_sets_dict, min_size=3, permutation_num=1000, seed=42, padj_thresh=0.05):
        """Run gseapy prerank GSEA over a topic-by-gene weight matrix.

        Each row (topic) is treated as an independent pre-ranked gene list; topic-gene
        weights serve as the ranking statistic, not expression counts.

        Parameters
        ----------
        topic_gene_weight_mat : pd.DataFrame, shape (n_topics, n_genes)
        gene_sets_dict : dict {pathway: [gene, ...]}
            Hallmark gene sets filtered to the scored gene universe.
        min_size : int
            Minimum gene set overlap with the ranked list to test.
        permutation_num : int
            Number of permutations for p-value estimation.

        Returns
        -------
        dict with keys:
            gsea_results_per_topic, gsea_long, gsea_long_filt, top_terms_per_topic
        """
        import gseapy as gp

        gsea_results_per_topic = {}
        rows = []

        for topic in topic_gene_weight_mat.index:
            rnk = (
                topic_gene_weight_mat.loc[topic]
                .dropna()
                .sort_values(ascending=False)
            )
            pre = gp.prerank(
                rnk=rnk,
                gene_sets=gene_sets_dict,
                min_size=min_size,
                max_size=len(rnk),
                permutation_num=permutation_num,
                seed=seed,
                no_plot=True,
                outdir=None,
                verbose=False,
            )
            gsea_results_per_topic[topic] = pre
            res = pre.res2d.copy()
            res = res.rename(columns={"Term": "pathway"})
            res.insert(0, "topic", topic)
            rows.append(res)

        gsea_long = (
            pd.concat(rows, ignore_index=True)
            .rename(columns={"NOM p-val": "pval", "FDR q-val": "padj", "NES": "nes"})
            [["topic", "pathway", "nes", "pval", "padj", "Lead_genes", "Tag %", "Gene %"]]
            .sort_values(["topic", "padj", "nes"], ascending=[True, True, False])
        )
        top_terms_per_topic = gsea_long.groupby("topic", group_keys=False).head(10)
        print(top_terms_per_topic)

        gsea_long_filt = gsea_long[gsea_long["padj"].le(padj_thresh)]
        significant_pathways_per_topic = (
            gsea_long_filt.groupby("topic")["pathway"].apply(list).to_dict()
        )
        for topic_to_plot, pathways in significant_pathways_per_topic.items():
            pre = gsea_results_per_topic[topic_to_plot]
            for pathway in pathways:
                row = gsea_long_filt.query("topic == @topic_to_plot and pathway == @pathway").iloc[0]
                axes = gp.gseaplot(
                    rank_metric=pre.ranking,
                    term=pathway,
                    **pre.results[pathway],
                    ofname=None,
                )
                fig = axes[0].get_figure()
                for ax in axes:
                    ax.set_title("")
                fig.suptitle(
                    f"{topic_to_plot} | {pathway}\nNES={row['nes']:.2f}, adj. p={row['padj']:.2e}",
                    fontsize=9,
                    y=1.02,
                )
                plt.tight_layout(); plt.show()

        return dict(
            gsea_results_per_topic=gsea_results_per_topic,
            gsea_long=gsea_long,
            gsea_long_filt=gsea_long_filt,
            top_terms_per_topic=top_terms_per_topic,
        )

    def run_topic_ora(topic_gene_weight_mat, gene_sets_dict, ora_tmin=3, padj_thresh=0.05):
        """Run gseapy ORA for each topic using a triangle-threshold gene cutoff.

        Top genes per topic are selected by a global triangle threshold applied to the
        flattened topic-gene weight distribution.  Background is the scored gene universe
        (columns of topic_gene_weight_mat).

        Parameters
        ----------
        topic_gene_weight_mat : pd.DataFrame, shape (n_topics, n_genes)
        gene_sets_dict : dict {pathway: [gene, ...]}
            Hallmark gene sets filtered to the scored gene universe.

        Returns
        -------
        dict with keys:
            knee_value, top_genes_per_topic, ora_long, ora_long_filt, top_ora_terms_per_topic
        """
        from skimage.filters import threshold_triangle
        import gseapy as gp

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

        background_genes = topic_gene_weight_mat.columns.astype(str).tolist()

        top_genes_per_topic = {
            topic: (
                topic_gene_weight_mat.loc[topic][topic_gene_weight_mat.loc[topic].ge(knee_value)]
                .sort_values(ascending=False)
                .index.astype(str)
                .tolist()
            )
            for topic in topic_gene_weight_mat.index
        }

        rows = []
        for topic, top_genes in top_genes_per_topic.items():
            if not top_genes:
                continue
            enr = gp.enrich(
                gene_list=top_genes,
                gene_sets=gene_sets_dict,
                background=background_genes,
                cutoff=1.0,
                no_plot=True,
                outdir=None,
            )
            res = enr.res2d.copy()
            res.index.name = "pathway"
            res = res.reset_index()
            res.insert(0, "topic", topic)
            rows.append(res)

        if not rows:
            print("No ORA results produced. Consider lowering ora_tmin.")
            empty = pd.DataFrame()
            return dict(
                knee_value=knee_value,
                top_genes_per_topic=top_genes_per_topic,
                ora_long=empty,
                ora_long_filt=empty,
                top_ora_terms_per_topic=empty,
            )

        ora_long = (
            pd.concat(rows, ignore_index=True)
            .rename(columns={
                "Adjusted P-value": "padj",
                "P-value": "pval",
                "Overlap": "overlap",
                "Genes": "overlap_genes",
            })
            [["topic", "pathway", "pval", "padj", "overlap", "overlap_genes"]]
            .sort_values(["topic", "padj"], ascending=[True, True])
        )
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

    def run_topic_ora_leiden(feature_topic_adata, gene_sets_dict, background_genes, ora_tmin=3, padj_thresh=0.05):

        # Set categorical order so that 'topic' is before 'gene' in gene_or_topic
        gene_topic_cat = pd.CategoricalDtype(['topic', 'gene'], ordered=True)
        feature_topic_adata.obs['gene_or_topic'] = feature_topic_adata.obs['gene_or_topic'].astype(gene_topic_cat)
        n_gene_topics_per_leiden = feature_topic_adata.obs.groupby('leiden')['gene_or_topic'].value_counts().sort_index(level=['leiden', 'gene_or_topic'])
        
        index_per_leiden = feature_topic_adata.obs.groupby('leiden').apply(lambda x: x.index.tolist())
        filt_enr_per_leiden = {}
        for leiden, index in index_per_leiden.items():
            data = feature_topic_adata[index].copy()
            genes = data.obs.loc[data.obs['gene_or_topic'].eq('gene'), 'label'].tolist()
            
            enr = gp.enrich(
                gene_list=genes,
                gene_sets='GO_Biological_Process_2025' if gene_sets_dict is None else gene_sets_dict,
                background=background_genes,
                cutoff=1.0,
                no_plot=True,
                outdir=None,
            )
            if enr.res2d is None:
                print(f"No ORA results produced for leiden {leiden}. Consider lowering ora_tmin.")
                continue

            res = enr.res2d.copy()
            res_filt = res[res['Adjusted P-value'].le(padj_thresh)]
            filt_enr_per_leiden[leiden] = res_filt

        return filt_enr_per_leiden, n_gene_topics_per_leiden
            

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

    ## convert net DataFrame to dict {pathway: [gene, ...]} for gseapy
    def _net_to_gene_sets_dict(net, split_source_target=False):
        if net.columns.equals(pd.Index(['source', 'target'])):
            return net.groupby("source")["target"].apply(list).to_dict()
        
        if split_source_target: # split by source/sender and target/receiver (nothing to do with source/target data)
            combined_gene_sets = {}
            combined_gene_sets.update({str(idx) + '_source': genes for idx, genes in net['source_genes'].to_dict().items()})
            combined_gene_sets.update({str(idx) + '_target': genes for idx, genes in net['target_genes'].to_dict().items()})
            return combined_gene_sets
        else:
            return net['all_genes'].to_dict()
   

    #source_gene_sets_dict = _net_to_gene_sets_dict(source_net)
    #target_gene_sets_dict = _net_to_gene_sets_dict(target_net)

    source_gene_sets_dict = target_gene_sets_dict = _net_to_gene_sets_dict(lr_hallmark_grouped, split_source_target=True)
    source_gene_sets_dict = target_gene_sets_dict = _net_to_gene_sets_dict(lr_resource_grouped, split_source_target=True)

    ## fetch background genes
    background_genes = source_x1.columns.tolist()

    ## run ORA
    source_ora, source_n_gene_topics_per_leiden = run_topic_ora_leiden(feature_topic_adatas['source_rna'], source_gene_sets_dict, background_genes)
    target_ora, target_n_gene_topics_per_leiden = run_topic_ora_leiden(feature_topic_adatas['target_rna'], target_gene_sets_dict, background_genes)

    ## run GSEA
    source_gsea = run_topic_gsea(source_topic_mat, source_gene_sets_dict)
    target_gsea = run_topic_gsea(target_topic_mat, target_gene_sets_dict)

    ## compute jaccard similarity between GSEA and ORA results
    source_jaccard = run_gsea_ora_overlap(source_gsea["gsea_long_filt"], source_ora["ora_long_filt"])
    target_jaccard = run_gsea_ora_overlap(target_gsea["gsea_long_filt"], target_ora["ora_long_filt"])

    #gsea_jaccard = run_gsea_ora_overlap(source_gsea["gsea_long_filt"], target_gsea["gsea_long_filt"])
    #ora_jaccard = run_gsea_ora_overlap(source_ora["ora_long_filt"], target_ora["ora_long_filt"])

    shared_gsea_paths = set(source_gsea["gsea_long_filt"]['pathway']) & set(target_gsea["gsea_long_filt"]['pathway'])
    shared_ora_paths = set(source_ora["ora_long_filt"]['pathway']) & set(target_ora["ora_long_filt"]['pathway'])
    shared_paths = shared_gsea_paths & shared_ora_paths
    print("Shared GSEA and ORA pathways across source and target:\n", shared_paths)

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

    #%% FASTopic analysis

    import decoupler as dc
    import decoupler.op

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
        topic_global_weights = adata.uns['fastopic']['global_weights'].copy()
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

    def extract_fastopic_embeddings(source_adata, target_adata, modality, do_procrustes=False, keep_cells=False):

        ## extract gene and topic embeddings from source and target data
        source_gene_embeddings = source_adata.varm['fastopic_gene_embeddings'].copy()
        target_gene_embeddings = target_adata.varm['fastopic_gene_embeddings'].copy()
        source_topic_embeddings = source_adata.uns['fastopic']['topic_embeddings'].copy()
        target_topic_embeddings = target_adata.uns['fastopic']['topic_embeddings'].copy()
        source_cell_embeddings = source_adata.obsm['MultiGATE_source_aligned'].copy()
        target_cell_embeddings = target_adata.obsm['MultiGATE_source_aligned'].copy()

        if do_procrustes:
            from scipy.spatial import procrustes
            source_gene_embeddings, target_gene_embeddings, disparity = procrustes(source_gene_embeddings, target_gene_embeddings)
            source_topic_embeddings, target_topic_embeddings, disparity = procrustes(source_topic_embeddings, target_topic_embeddings)

        ## form sinlge adata object
        source_target_gene_embeddings = np.concatenate([source_gene_embeddings, target_gene_embeddings], axis=0)
        source_target_topic_embeddings = np.concatenate([source_topic_embeddings, target_topic_embeddings], axis=0)
        source_target_cell_embeddings = np.concatenate([source_cell_embeddings, target_cell_embeddings], axis=0)

        feature_type = 'gene' if modality == 'rna' else 'peak'
        gene_or_topic = np.concatenate([
            np.full(source_target_gene_embeddings.shape[0], feature_type),
            np.full(source_target_topic_embeddings.shape[0], 'topic'),
            np.full(source_target_cell_embeddings.shape[0], 'cell'),
        ])
        label = np.concatenate([
            source_adata.var_names,
            target_adata.var_names,
            [f'topic_{i}' for i in range(source_topic_embeddings.shape[0])],
            [f'topic_{i}' for i in range(target_topic_embeddings.shape[0])],
            source_adata.obs_names,
            target_adata.obs_names,
        ])
        source_or_target = np.concatenate([
            np.full(source_gene_embeddings.shape[0], 'source'),
            np.full(target_gene_embeddings.shape[0], 'target'),
            np.full(source_topic_embeddings.shape[0], 'source'),
            np.full(target_topic_embeddings.shape[0], 'target'),
            np.full(source_cell_embeddings.shape[0], 'source'),
            np.full(target_cell_embeddings.shape[0], 'target'),
        ])

        gene_topic_adata = sc.AnnData(
            X = np.concatenate(
                [source_target_gene_embeddings, source_target_topic_embeddings, source_target_cell_embeddings],
                axis=0,
            ),
            obs = pd.DataFrame(
                data = {
                'gene_or_topic': gene_or_topic,
                'source_or_target': source_or_target,
                'label': label},
            )
        )
        gene_topic_adata.obs['combination'] = gene_topic_adata.obs['source_or_target'].astype(str) + '_' + gene_topic_adata.obs['gene_or_topic'].astype(str)

        ## split source and target adatas
        source_adata = gene_topic_adata[gene_topic_adata.obs['source_or_target'].eq('source')].copy()
        target_adata = gene_topic_adata[gene_topic_adata.obs['source_or_target'].eq('target')].copy()

        if not keep_cells:
            source_adata = source_adata[~source_adata.obs['gene_or_topic'].eq('cell')]
            target_adata = target_adata[~target_adata.obs['gene_or_topic'].eq('cell')]

        return source_adata, target_adata

    ## extract topic embeddings from source and target data
    feature_topic_adatas = {}
    feature_topic_adatas['source_rna'], feature_topic_adatas['target_rna'] = extract_fastopic_embeddings(source_rna, target_rna, modality='rna')
    feature_topic_adatas['source_atac'], feature_topic_adatas['target_atac'] = extract_fastopic_embeddings(source_atac, target_atac, modality='atac')

    ## concat a subset of the feature topic adatas for visualization
    '''
    concat_feature_topic_adatas = sc.concat([feature_topic_adatas['target_rna'], feature_topic_adatas['target_atac']], axis=0)
    sc.pp.neighbors(concat_feature_topic_adatas, use_rep='X', n_neighbors=10)
    sc.tl.leiden(concat_feature_topic_adatas)
    sc.tl.umap(concat_feature_topic_adatas, min_dist=0.3)
    umap_axes = sc.pl.umap(
        concat_feature_topic_adatas,
        color=['gene_or_topic', 'leiden'],
        ncols=3,
        wspace=0.2,
        size=concat_feature_topic_adatas.obs['gene_or_topic'].map({'gene':20, 'peak':20, 'topic':250}),
        show=False,
    )
    topic_mask = concat_feature_topic_adatas.obs['gene_or_topic'].eq('topic').to_numpy()
    topic_coords = concat_feature_topic_adatas.obsm['X_umap'][topic_mask]
    n_topics = int(feature_topic_adatas['target_rna'].obs['gene_or_topic'].eq('topic').sum())
    topic_labels = [str(i % n_topics) for i in range(topic_coords.shape[0])]
    for ax in np.atleast_1d(umap_axes):
        for (x_coord, y_coord), topic_label in zip(topic_coords, topic_labels):
            ax.text(x_coord, y_coord, topic_label, fontsize=8, ha='center', va='center')
    plt.tight_layout(); plt.show()
    
    '''
    ## ingest ATAC embeddings into RNA UMAP - TARGET DATA
    sc.pp.neighbors(feature_topic_adatas['target_rna'], use_rep='X', n_neighbors=10)
    sc.tl.umap(feature_topic_adatas['target_rna'], min_dist=0.3)
    sc.tl.leiden(feature_topic_adatas['target_rna'], resolution=0.5)
    sc.tl.ingest(feature_topic_adatas['target_atac'], feature_topic_adatas['target_rna'], embedding_method='umap', obs='leiden')
    concat_feature_topic_adatas = sc.concat([feature_topic_adatas['target_rna'], feature_topic_adatas['target_atac']], axis=0)
    umap_axes = sc.pl.umap(
        concat_feature_topic_adatas,
        color=['gene_or_topic', 'leiden'],
        ncols=3,
        wspace=0.1,
        size=concat_feature_topic_adatas.obs['gene_or_topic'].map({'gene':20, 'peak':20, 'topic':350}),
        show=False,
    )
    topic_mask = concat_feature_topic_adatas.obs['gene_or_topic'].eq('topic').to_numpy()
    topic_coords = concat_feature_topic_adatas.obsm['X_umap'][topic_mask]
    n_topics = int(feature_topic_adatas['target_rna'].obs['gene_or_topic'].eq('topic').sum())
    topic_labels = [str(i % n_topics) for i in range(topic_coords.shape[0])]
    for ax in np.atleast_1d(umap_axes):
        for (x_coord, y_coord), topic_label in zip(topic_coords, topic_labels):
            ax.text(x_coord, y_coord, topic_label, fontsize=8, ha='center', va='center')
    plt.tight_layout(); plt.show()

    ## ingest ATAC embeddings into RNA UMAP - SOURCE DATA
    sc.pp.neighbors(feature_topic_adatas['source_rna'], use_rep='X', n_neighbors=10)
    sc.tl.umap(feature_topic_adatas['source_rna'], min_dist=0.3)
    sc.tl.leiden(feature_topic_adatas['source_rna'], resolution=0.5)
    sc.tl.ingest(feature_topic_adatas['source_atac'], feature_topic_adatas['source_rna'], embedding_method='umap', obs='leiden')
    concat_feature_topic_adatas = sc.concat([feature_topic_adatas['source_rna'], feature_topic_adatas['source_atac']], axis=0)
    umap_axes = sc.pl.umap(
        concat_feature_topic_adatas,
        color=['gene_or_topic', 'leiden'],
        ncols=3,
        wspace=0.1,
        size=concat_feature_topic_adatas.obs['gene_or_topic'].map({'gene':20, 'peak':20, 'topic':350}),
        show=False,
    )
    topic_mask = concat_feature_topic_adatas.obs['gene_or_topic'].eq('topic').to_numpy()
    topic_coords = concat_feature_topic_adatas.obsm['X_umap'][topic_mask]
    n_topics = int(feature_topic_adatas['source_rna'].obs['gene_or_topic'].eq('topic').sum())
    topic_labels = [str(i % n_topics) for i in range(topic_coords.shape[0])]
    for ax in np.atleast_1d(umap_axes):
        for (x_coord, y_coord), topic_label in zip(topic_coords, topic_labels):
            ax.text(x_coord, y_coord, topic_label, fontsize=8, ha='center', va='center')
    plt.tight_layout(); plt.show()

    ## compare topic embeddings across datasets and modalities
    topic_embeddings = []
    for dat in feature_topic_adatas.values():
        embs = dat[dat.obs['gene_or_topic'].eq('topic')].X.copy()
        topic_embeddings.append(embs)

    if \
        (topic_embeddings[0] == topic_embeddings[2]).all() and \
        (topic_embeddings[1] == topic_embeddings[3]).all():
        print("Tied topic embeddings confirmed.")

    topic_embeddings = np.concatenate(topic_embeddings, axis=0)
    corr_matrix = np.corrcoef(topic_embeddings)
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(); plt.show()

    ## build shared Hallmark mouse net once
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

    ## extract topic-by-gene weight matrix from source and target data
    source_topic_mat, source_sorted_topics, source_net = _extract_fastopic_topic_gene_weights(
        source_rna, hallmark_mouse_net,
    )
    target_topic_mat, target_sorted_topics, target_net = _extract_fastopic_topic_gene_weights(
        target_rna, hallmark_mouse_net,
    )

    ## build LR-level sender-receiver gene sets
    import liana as li
    import decoupler as dc

    # 1) Load ligand-receptor pairs from LIANA
    lr_pairs = li.rs.select_resource('mouseconsensus')

    # 2) Load pathway gene sets / weights from decoupler
    #avail_resources = dc.op.show_resources()
    #resource = dc.op.resource('HPA_secretome')


    # 3) Generate LR-level sender-receiver gene sets
    lr_hallmark = li.rs.generate_lr_geneset(
        lr_pairs,
        hallmark_mouse_net,
        lr_sep="^",
        weight=None
    )

    ortholog_mapper = map_df.set_index('target_human').to_dict()['target']
    resource = dc.op.progeny(organism='human')
    resource = resource.map(lambda x: ortholog_mapper.get(x, x) if pd.notnull(x) else x)

    lr_resource = li.rs.generate_lr_geneset(
        lr_pairs,
        resource,
        lr_sep="^",
        weight=None
    )

    ## create LR genesets
    lr_hallmark_grouped_source = lr_hallmark.groupby('source')['interaction'].apply(lambda x: np.unique(x.str.split('^').str[0].explode().tolist()))
    lr_hallmark_grouped_target = lr_hallmark.groupby('source')['interaction'].apply(lambda x: np.unique(x.str.split('^').str[1].explode().tolist()))
    lr_hallmark_grouped = pd.concat([lr_hallmark_grouped_source.rename('source_genes'), lr_hallmark_grouped_target.rename('target_genes')], axis=1)
    lr_hallmark_grouped['all_genes'] = lr_hallmark_grouped.apply(lambda x: list(x['source_genes']) + list(x['target_genes']), axis=1)
    lr_hallmark_grouped.index.rename('pathway', inplace=True)

    lr_resource_grouped_source = lr_resource.groupby('source')['interaction'].apply(lambda x: np.unique(x.str.split('^').str[0].explode().tolist()))
    lr_resource_grouped_target = lr_resource.groupby('source')['interaction'].apply(lambda x: np.unique(x.str.split('^').str[1].explode().tolist()))
    lr_resource_grouped = pd.concat([lr_resource_grouped_source.rename('source_genes'), lr_resource_grouped_target.rename('target_genes')], axis=1)
    lr_resource_grouped['all_genes'] = lr_resource_grouped.apply(lambda x: list(x['source_genes']) + list(x['target_genes']), axis=1)
    lr_resource_grouped.index.rename('pathway', inplace=True)

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
