#!/usr/bin/env python
#%%
import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

# Python 3.7 compatibility for muon/mudata (they use typing.Literal in newer versions)
if sys.version_info < (3, 8):
    import typing
    from typing_extensions import Literal

    typing.Literal = Literal

from dotenv import dotenv_values, load_dotenv
ENV_FILE_PATH = "/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"
DEFAULT_SOURCE_LABEL_KEY = "RNA_clusters"
DEFAULT_TARGET_LABEL_KEY = "arc_gex_kmeans_5_clusters_Cluster"
SPLIT_RATIO_TRAIN = 0.7
SPLIT_RATIO_VAL = 0.2
SPLIT_RATIO_TEST = 0.1
SPLIT_ARTIFACT_PATH = os.path.join("splits", "domain_splits.json")
SPLIT_SCHEMA_VERSION = 1

BASE_PATH = None
REPO_ROOT = None
MultiGATE = None
MultiGATETrainer = None


def bootstrap_runtime():
    global BASE_PATH, REPO_ROOT, MultiGATE, MultiGATETrainer

    load_dotenv(dotenv_path=ENV_FILE_PATH)
    print("Loaded environment variables from .env or env:", end="\n\n")
    pprint(dotenv_values(ENV_FILE_PATH))

    datapath = os.getenv("DATAPATH")
    if datapath is None:
        raise EnvironmentError(
            "DATAPATH is not set. Export DATAPATH to the base data directory, e.g. "
            "'/home/mcb/users/dmannk/BAKLAVA_base/data'."
        )

    if shutil.which("bedtools") is None:
        raise EnvironmentError(
            "bedtools is required for Cal_gene_peak_Net_new. Install bedtools and ensure sortBed is available on PATH."
        )

    baklava_base_dir = os.getenv("BAKLAVA_BASE_DIR")
    if baklava_base_dir is None:
        raise EnvironmentError(
            "BAKLAVA_BASE_DIR is not set. Export BAKLAVA_BASE_DIR to the base repo directory, e.g. "
            "'/home/mcb/users/dmannk/BAKLAVA_base'."
        )

    REPO_ROOT = os.path.join(baklava_base_dir, "MultiGATE")
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    baklava_repo_root = os.path.join(baklava_base_dir, "BAKLAVA")
    if os.path.isdir(baklava_repo_root) and baklava_repo_root not in sys.path:
        sys.path.insert(0, baklava_repo_root)

    # Make sure env-local binaries (e.g., bedtools) are discoverable when running
    # with an explicit python path instead of an activated conda shell.
    env_bin = os.path.dirname(sys.executable)
    current_path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if env_bin and env_bin not in current_path_entries:
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")

    if MultiGATE is None or MultiGATETrainer is None:
        import MultiGATE as multigate_module
        from MultiGATE.MultiGATE import MultiGATE as multigate_trainer_cls

        MultiGATE = multigate_module
        MultiGATETrainer = multigate_trainer_cls
        print("Using MultiGATE module:", MultiGATE.__file__)

    BASE_PATH = os.path.join(datapath, "aligned_data")


def require_runtime_bootstrap():
    if BASE_PATH is None or REPO_ROOT is None or MultiGATE is None or MultiGATETrainer is None:
        raise RuntimeError("bootstrap_runtime() must be called before training or data preparation.")


def load_nichecompass_combined_gp_dict_mouse(
    *,
    species: str = "mouse",
    load_from_disk: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Load the same combined_gp_dict as BAKLAVA `mouse_brain_multimodal.py` / `data_utils.build_combined_gp_dict_mouse_brain`.

    Expects `DATAPATH` and `BAKLAVA_BASE_DIR` (and BAKLAVA repo on sys.path from `bootstrap_runtime`).
    Gene-program CSVs live under ``{DATAPATH}/gene_programs/``; orthologs under ``{DATAPATH}/gene_annotations/``.
    """
    datapath = os.getenv("DATAPATH")
    if datapath is None:
        raise EnvironmentError("DATAPATH must be set to load combined_gp_dict.")

    ga_data_folder_path = os.path.join(datapath, "gene_annotations")
    gp_data_folder_path = os.path.join(datapath, "gene_programs")
    omnipath_lr_network_file_path = os.path.join(gp_data_folder_path, "omnipath_lr_network.csv")
    nichenet_lr_network_file_path = os.path.join(
        gp_data_folder_path, "nichenet_lr_network_v2_{}.csv".format(species)
    )
    nichenet_ligand_target_matrix_file_path = os.path.join(
        gp_data_folder_path, "nichenet_ligand_target_matrix_v2_{}.csv".format(species)
    )
    mebocost_enzyme_sensor_interactions_folder_path = os.path.join(
        gp_data_folder_path, "metabolite_enzyme_sensor_gps"
    )
    collectri_tf_network_file_path = os.path.join(
        gp_data_folder_path, "collectri_tf_network_{}.csv".format(species)
    )
    gene_orthologs_mapping_file_path = os.path.join(ga_data_folder_path, "human_mouse_gene_orthologs.csv")

    combined_gp_dict_path = os.path.join(gp_data_folder_path, "combined_gp_dict_{}.pkl".format(species))
    if load_from_disk and os.path.exists(combined_gp_dict_path):
        with open(combined_gp_dict_path, "rb") as f:
            combined = pickle.load(f)
        if verbose:
            print("Loaded NicheCompass combined_gp_dict from disk.")
        return combined

    from data_utils import build_combined_gp_dict_mouse_brain

    combined = build_combined_gp_dict_mouse_brain(
        species=species,
        omnipath_lr_network_file_path=omnipath_lr_network_file_path,
        nichenet_lr_network_file_path=nichenet_lr_network_file_path,
        nichenet_ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
        mebocost_enzyme_sensor_interactions_folder_path=mebocost_enzyme_sensor_interactions_folder_path,
        collectri_tf_network_file_path=collectri_tf_network_file_path,
        gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
        load_from_disk=load_from_disk,
        verbose=verbose,
    )
    if verbose:
        print("Built NicheCompass combined_gp_dict with {} gene programs and saved to disk.".format(len(combined)))
    with open(combined_gp_dict_path, "wb") as f:
        pickle.dump(combined, f)
    return combined


import matplotlib.pyplot as plt
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.preprocessing import Normalizer
from ignite.metrics import MaximumMeanDiscrepancy as MMD
from tqdm import tqdm
import tempfile
import pickle

import warnings
warnings.filterwarnings("ignore")


@dataclass
class DomainData:
    rna: AnnData
    atac: AnnData
    label_key: Optional[str] = None


@dataclass
class DomainSplitBundle:
    full: DomainData
    train: DomainData
    val: DomainData
    test: DomainData
    eval: DomainData
    split_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    split_obs_names: Dict[str, np.ndarray] = field(default_factory=dict)
    split_seed: Optional[int] = None

    @property
    def rna(self):
        return self.train.rna

    @property
    def atac(self):
        return self.train.atac

    @property
    def label_key(self):
        return self.train.label_key


@dataclass
class DataBundle:
    source: DomainSplitBundle
    target: DomainSplitBundle
    split_metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional NicheCompass prior pathways used to build fixed rho pathway masks.
    combined_gp_dict: Optional[Dict[str, Any]] = None


@dataclass
class GraphInputBundle:
    graph_tf: Tuple[Any, Any, Any]
    gp_tf: Tuple[Any, Any, Any]
    x1: pd.DataFrame
    x2: pd.DataFrame

    def __repr__(self) -> str:
        def _sparse_tf_summary(tf: Tuple[Any, Any, Any]) -> str:
            if tf is None or len(tf) < 3:
                return "invalid"
            shape = tf[2]
            n_edges = int(getattr(tf[0], "shape", [0])[0]) if tf[0] is not None else 0
            return "edges={}, shape={}".format(n_edges, shape)

        return (
            "GraphInputBundle(graph_tf=({}), gp_tf=({}), x1.shape={}, x2.shape={})".format(
                _sparse_tf_summary(self.graph_tf),
                _sparse_tf_summary(self.gp_tf),
                tuple(self.x1.shape),
                tuple(self.x2.shape),
            )
        )


@dataclass
class GraphBundle:
    source: GraphInputBundle
    target: GraphInputBundle
    bp_width: int = 400
    graph_type: str = "ATAC"
    protein_value: float = 0.001

    def __repr__(self) -> str:
        return (
            "GraphBundle(bp_width={}, graph_type={}, protein_value={}, source={}, target={})".format(
                self.bp_width,
                self.graph_type,
                self.protein_value,
                self.source,
                self.target,
            )
        )


@dataclass
class PathwayDecoderMaskBundle:
    pathway_names: np.ndarray
    source_pathway_names: np.ndarray
    target_pathway_names: np.ndarray
    rho_rna_mask: np.ndarray
    rho_atac_mask: np.ndarray
    n_zero_source_pathways: int = 0
    n_zero_target_pathways: int = 0


@dataclass
class Stage1CacheConfig:
    use_cache: bool
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    run_params: Dict[str, Any] = field(default_factory=dict)
    dual_source_kd: bool = False
    student_graph_type: str = "identity"
    vgp_anchor_mode: str = "feature"


@dataclass
class Stage1TrainerBundle:
    teacher: Any
    student: Optional[Any]
    nonspatial: Optional[Any]
    source_inputs_tensors: Optional[Tuple[Any, Any, Any, Any, Any]]
    source_student_inputs_tensors: Optional[Tuple[Any, Any, Any, Any, Any]]
    source_student_graph_tf: Optional[Tuple[Any, Any, Any]]
    primary: Any
    primary_model_name: str
    primary_source_graph_tf: Tuple[Any, Any, Any]


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(description="Train MultiGATE on source and run live zero-shot eval on target.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed used to generate deterministic 70/20/10 train/val/test splits for both domains.",
    )
    parser.add_argument(
        "--source-split-train-eval",
        action="store_true",
        default=False,
        help=(
            "If set, use source train/eval split subsets (train vs val+test). "
            "By default, source train/eval both use the full source dataset while target remains split."
        ),
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
        "--top-n-genes",
        type=int,
        default=2000,
        help="Number of top genes to keep for filtering.",
    )
    parser.add_argument(
        "--top-n-peaks",
        type=int,
        default=10000,
        help="Number of top in-cis peaks to keep for filtering.",
    )
    parser.add_argument(
        "--lambda-kd",
        type=float,
        default=1.0,
        help="Scale factor for KD objectives (stage-1 dual-source KD and stage-2 target KD).",
    )
    parser.add_argument(
        "--kd-mix-kl",
        type=float,
        default=0.1,
        help="Weight on the KL distillation term in stage-2 loss: lambda_kd * (kd_mix_kl * KL + kd_mix_ot * OT-CLIP).",
    )
    parser.add_argument(
        "--kd-mix-ot",
        type=float,
        default=0.9,
        help="Weight on the OT-CLIP distillation term in stage-2 loss (see --kd-mix-kl).",
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
        "--vgp-anchor-mode",
        type=str,
        choices=["spot", "feature"],
        default="feature",
        help=(
            "How vgp0/vgp1 are parameterized inside MGATE. "
            "'spot': vgp shape is (n_cells, 1), i.e. spot-anchored (legacy behavior). "
            "'feature': vgp shape is (n_features_total, 1), i.e. feature-anchored."
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
        "--stage1-mlflow-cache-run-name",
        "--stage1-mlflow-cache-dir",
        dest="stage1_mlflow_cache_run_name",
        type=str,
        default=None,
        help=(
            "MLflow run name to load stage-1 model artifacts from and reuse as the parent run. "
            "--stage1-mlflow-cache-dir is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--switcharoo",
        action="store_true",
        default=False,
        help="If set, swap source and target.",
    )
    parser.add_argument(
        "--combined-gp-dict",
        action="store_true",
        default=False,
        help=(
            "If set, do loading the NicheCompass combined_gp_dict from BAKLAVA data_utils "
            "(gene_programs/ CSVs under DATAPATH). Use when files are missing or to save startup time."
        ),
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


def _extract_gp_gene_list(gp_entry: Any, key: str) -> List[str]:
    if not isinstance(gp_entry, dict):
        return []
    genes = gp_entry.get(key, [])
    if isinstance(genes, np.ndarray):
        genes = genes.tolist()
    if genes is None:
        genes = []
    return [str(gene).upper() for gene in genes if gene is not None and str(gene)]


def build_pathway_decoder_masks_from_gp_dict(
    combined_gp_dict: Dict[str, Any],
    gene_names: pd.Index,
    peak_names: pd.Index,
    gene_peak_net: pd.DataFrame,
    drop_zero_overlap_rows: bool = True,
) -> PathwayDecoderMaskBundle:
    if not isinstance(combined_gp_dict, dict) or len(combined_gp_dict) == 0:
        raise ValueError("combined_gp_dict must be a non-empty dictionary to build pathway decoder masks.")
    if gene_peak_net is None:
        raise ValueError("gene_peak_Net is required to map pathway genes to peaks.")
    if not {"Gene", "Peak"}.issubset(set(gene_peak_net.columns)):
        raise ValueError("gene_peak_Net must contain columns {'Gene', 'Peak'}.")

    gene_names = pd.Index(gene_names).astype(str)
    peak_names = pd.Index(peak_names).astype(str)

    gene_to_idx = {gene.upper(): idx for idx, gene in enumerate(gene_names)}
    peak_to_idx = {peak: idx for idx, peak in enumerate(peak_names)}

    gp_links = gene_peak_net.loc[:, ["Gene", "Peak"]].copy()
    gp_links["Gene"] = gp_links["Gene"].astype(str).str.upper()
    gp_links["Peak"] = gp_links["Peak"].astype(str)
    gp_links = gp_links[gp_links["Peak"].isin(peak_to_idx)].drop_duplicates(["Gene", "Peak"])

    gene_to_peak_indices: Dict[str, List[int]] = {}
    for gene, gene_df in gp_links.groupby("Gene", sort=False):
        gene_to_peak_indices[gene] = [peak_to_idx[peak] for peak in gene_df["Peak"].tolist()]

    n_genes = len(gene_names)
    n_peaks = len(peak_names)
    source_pathway_names: List[str] = []
    target_pathway_names: List[str] = []
    source_rna_masks: List[np.ndarray] = []
    source_atac_masks: List[np.ndarray] = []
    target_rna_masks: List[np.ndarray] = []
    target_atac_masks: List[np.ndarray] = []
    n_zero_source = 0
    n_zero_target = 0

    def _build_feature_rows(pathway_genes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        rna_row = np.zeros(n_genes, dtype=np.float32)
        atac_row = np.zeros(n_peaks, dtype=np.float32)
        for gene in pathway_genes:
            gene_idx = gene_to_idx.get(gene)
            if gene_idx is not None:
                rna_row[gene_idx] = 1.0
            for peak_idx in gene_to_peak_indices.get(gene, []):
                atac_row[peak_idx] = 1.0
        return rna_row, atac_row

    for gp_name, gp_entry in combined_gp_dict.items():
        gp_name_str = str(gp_name)
        source_genes = _extract_gp_gene_list(gp_entry, "sources")
        target_genes = _extract_gp_gene_list(gp_entry, "targets")

        source_rna_row, source_atac_row = _build_feature_rows(source_genes)
        target_rna_row, target_atac_row = _build_feature_rows(target_genes)

        zero_source = (source_rna_row.sum() + source_atac_row.sum()) == 0
        zero_target = (target_rna_row.sum() + target_atac_row.sum()) == 0

        if not drop_zero_overlap_rows or not zero_source:
            source_rna_masks.append(source_rna_row)
            source_atac_masks.append(source_atac_row)
            source_pathway_names.append("{}__source".format(gp_name_str))

        if not drop_zero_overlap_rows or not zero_target:
            target_rna_masks.append(target_rna_row)
            target_atac_masks.append(target_atac_row)
            target_pathway_names.append("{}__target".format(gp_name_str))

        if zero_source:
            n_zero_source += 1
        if zero_target:
            n_zero_target += 1

    source_rna_mask = np.stack(source_rna_masks, axis=0)
    source_atac_mask = np.stack(source_atac_masks, axis=0)
    target_rna_mask = np.stack(target_rna_masks, axis=0)
    target_atac_mask = np.stack(target_atac_masks, axis=0)

    rho_rna_mask = np.concatenate([source_rna_mask, target_rna_mask], axis=0).astype(np.float32, copy=False)
    rho_atac_mask = np.concatenate([source_atac_mask, target_atac_mask], axis=0).astype(np.float32, copy=False)
    pathway_names = np.asarray(source_pathway_names + target_pathway_names, dtype=object)

    if float(rho_rna_mask.sum() + rho_atac_mask.sum()) == 0:
        raise ValueError(
            "Constructed pathway decoder masks are entirely zero after feature alignment. "
            "Check that combined_gp_dict genes overlap current RNA features and gene_peak_Net."
        )

    return PathwayDecoderMaskBundle(
        pathway_names=pathway_names,
        source_pathway_names=np.asarray(source_pathway_names, dtype=object),
        target_pathway_names=np.asarray(target_pathway_names, dtype=object),
        rho_rna_mask=rho_rna_mask,
        rho_atac_mask=rho_atac_mask,
        n_zero_source_pathways=int(n_zero_source),
        n_zero_target_pathways=int(n_zero_target),
    )


def maybe_build_pathway_decoder_masks(data_bundle: DataBundle, graph_bundle: GraphBundle) -> Optional[PathwayDecoderMaskBundle]:
    if data_bundle.combined_gp_dict is None:
        return None
    source_rna = data_bundle.source.rna
    gene_peak_net = source_rna.uns.get("gene_peak_Net")
    if gene_peak_net is None:
        raise ValueError(
            "combined_gp_dict was provided but source gene_peak_Net is missing. "
            "Cannot build pathway-by-peak rho masks."
        )

    drop_zero_overlap_rows = getattr(graph_bundle, "drop_zero_overlap_rows", True)
    mask_bundle = build_pathway_decoder_masks_from_gp_dict(
        combined_gp_dict=data_bundle.combined_gp_dict,
        gene_names=graph_bundle.source.x1.columns,
        peak_names=graph_bundle.source.x2.columns,
        gene_peak_net=gene_peak_net,
        drop_zero_overlap_rows=drop_zero_overlap_rows,
    )
    print(
        "[GP Masks] Built {} source + {} target pathways ({} total). "
        "Zero-overlap rows: source={}, target={}. Using fixed rho masks and dense alpha. {}."
        .format(
            len(mask_bundle.source_pathway_names),
            len(mask_bundle.target_pathway_names),
            len(mask_bundle.pathway_names),
            mask_bundle.n_zero_source_pathways,
            mask_bundle.n_zero_target_pathways,
            'DROPPING' if drop_zero_overlap_rows else 'KEEPING',
        )
    )
    return mask_bundle


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


def _extract_linear_decoder_kwargs_from_mgate(mgate) -> Dict[str, Any]:
    if not getattr(mgate, "linear_etm_decoder", False):
        return {}

    kwargs: Dict[str, Any] = {}
    if hasattr(mgate, "alpha"):
        kwargs["etm_emb_dim"] = int(mgate.alpha.shape[1])

    rho_is_fixed = False
    if hasattr(mgate, "rho_is_fixed_mask"):
        rho_is_fixed = bool(int(mgate.rho_is_fixed_mask.detach().cpu().item()))
    if rho_is_fixed:
        kwargs["rho_rna_mask"] = mgate.rho_rna.detach().cpu().numpy()
        kwargs["rho_atac_mask"] = mgate.rho_atac.detach().cpu().numpy()
    return kwargs


def build_zero_shot_target_trainer(source_trainer, target_spot_num, vgp_anchor_mode=None):
    # Rebuild MGATE with target N and load transferable weights.
    # If vgp_anchor_mode == "feature": transfer trained vgp directly.
    # If vgp_anchor_mode == "spot":    zero out vgp0/vgp1 (prior-only GP attention).
    if vgp_anchor_mode is None:
        vgp_anchor_mode = getattr(source_trainer.mgate, "vgp_anchor_mode", "spot")

    target_trainer = MultiGATETrainer(
        hidden_dims1=source_trainer.mgate.hidden_dims1,
        hidden_dims2=source_trainer.mgate.hidden_dims2,
        spot_num=target_spot_num,
        temp=float(source_trainer.mgate.logit_scale.detach().cpu().item()),
        vgp_anchor_mode=vgp_anchor_mode,
        n_epochs=1,
        lr=source_trainer.lr,
        gradient_clipping=source_trainer.gradient_clipping,
        nonlinear=source_trainer.mgate.nonlinear,
        verbose=False,
        random_seed=0,
        config={"device": str(source_trainer.device)},
        skip_gp_attention=source_trainer.mgate.skip_gp_attention,
        **_extract_linear_decoder_kwargs_from_mgate(source_trainer.mgate),
    )

    if vgp_anchor_mode == "feature":
        state_dict = source_trainer.mgate.state_dict()
        target_trainer.mgate.load_state_dict(state_dict, strict=False)
    else:  # vgp_anchor_mode == "spot"
        state_dict = {
            key: value
            for key, value in source_trainer.mgate.state_dict().items()
            if key not in {"vgp0", "vgp1"}
        }
        target_trainer.mgate.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            target_trainer.mgate.vgp0.zero_()
            target_trainer.mgate.vgp1.zero_()

    for attr_name in ("pathway_names", "source_pathway_names", "target_pathway_names"):
        if hasattr(source_trainer.mgate, attr_name):
            attr_value = getattr(source_trainer.mgate, attr_name)
            setattr(
                target_trainer.mgate,
                attr_name,
                attr_value.copy() if hasattr(attr_value, "copy") else attr_value,
            )

    return target_trainer


def _effective_vgp_anchor_mode(trainer, vgp_anchor_mode=None):
    if vgp_anchor_mode is not None:
        return vgp_anchor_mode
    return getattr(trainer.mgate, "vgp_anchor_mode", "spot")


def maybe_resize_trainer_for_inference(source_trainer, spot_num, vgp_anchor_mode=None):
    mode = _effective_vgp_anchor_mode(source_trainer, vgp_anchor_mode=vgp_anchor_mode)
    if mode != "spot":
        return source_trainer

    current_spot_num = int(source_trainer.mgate.vgp0.shape[0])
    if current_spot_num == int(spot_num):
        return source_trainer
    return build_zero_shot_target_trainer(
        source_trainer=source_trainer,
        target_spot_num=int(spot_num),
        vgp_anchor_mode=mode,
    )


def infer_source_embeddings(
    source_trainer,
    source_graph_tf,
    source_gp_tf,
    source_x1,
    source_x2,
    vgp_anchor_mode=None,
):
    source_infer_trainer = maybe_resize_trainer_for_inference(
        source_trainer,
        source_x1.shape[0],
        vgp_anchor_mode=vgp_anchor_mode,
    )
    source_infer_trainer.mgate.eval()
    with torch.no_grad():
        source_embeddings = source_infer_trainer.infer(
            source_graph_tf,
            source_graph_tf,
            source_gp_tf,
            source_x1,
            source_x2,
        )
    return source_embeddings, source_infer_trainer


def infer_target_embeddings_from_source_trainer(
    source_trainer,
    target_graph_tf,
    target_gp_tf,
    target_x1,
    target_x2,
    target_spot_num,
    vgp_anchor_mode=None,
):
    target_trainer = build_zero_shot_target_trainer(
        source_trainer,
        target_spot_num,
        vgp_anchor_mode=vgp_anchor_mode,
    )
    target_trainer.mgate.eval()
    with torch.no_grad():
        target_embeddings = target_trainer.infer(
            target_graph_tf,
            target_graph_tf,
            target_gp_tf,
            target_x1,
            target_x2,
        )
    return target_embeddings, target_trainer


def infer_source_and_zero_shot_target_embeddings(
    source_trainer,
    source_graph_tf,
    source_gp_tf,
    source_x1,
    source_x2,
    target_graph_tf,
    target_gp_tf,
    target_x1,
    target_x2,
    target_spot_num,
    vgp_anchor_mode=None,
):
    source_embeddings, source_infer_trainer = infer_source_embeddings(
        source_trainer=source_trainer,
        source_graph_tf=source_graph_tf,
        source_gp_tf=source_gp_tf,
        source_x1=source_x1,
        source_x2=source_x2,
        vgp_anchor_mode=vgp_anchor_mode,
    )
    target_embeddings, target_trainer = infer_target_embeddings_from_source_trainer(
        source_trainer=source_trainer,
        target_graph_tf=target_graph_tf,
        target_gp_tf=target_gp_tf,
        target_x1=target_x1,
        target_x2=target_x2,
        target_spot_num=target_spot_num,
        vgp_anchor_mode=vgp_anchor_mode,
    )
    del source_infer_trainer
    return source_embeddings, target_embeddings, target_trainer


# Legacy helper kept for compatibility with older co-embed runs that do not
# have split artifacts.
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


def pair_modalities(rna, atac, domain_name):
    shared_obs = rna.obs_names.intersection(atac.obs_names)
    if len(shared_obs) == 0:
        raise ValueError("{} RNA/ATAC share zero cells after preprocessing.".format(domain_name))
    return rna[shared_obs].copy(), atac[shared_obs].copy()


def _build_split_indices(n_obs, seed, domain_name):
    if n_obs <= 0:
        raise ValueError("{} has zero observations and cannot be split.".format(domain_name))

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_obs)
    train_n = int(np.floor(SPLIT_RATIO_TRAIN * n_obs))
    val_n = int(np.floor(SPLIT_RATIO_VAL * n_obs))
    test_n = int(n_obs - train_n - val_n)
    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise ValueError(
            "{} split would create an empty subset with n_obs={} (train={}, val={}, test={}).".format(
                domain_name,
                n_obs,
                train_n,
                val_n,
                test_n,
            )
        )

    idx_train = perm[:train_n]
    idx_val = perm[train_n:train_n + val_n]
    idx_test = perm[train_n + val_n:]
    idx_eval = np.concatenate([idx_val, idx_test], axis=0)
    return {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
        "eval": idx_eval,
    }


def _subset_domain(rna, atac, obs_names, label_key):
    sub_rna = rna[obs_names].copy()
    sub_atac = atac[obs_names].copy()
    sub_rna.uns["label_key"] = label_key
    return DomainData(rna=sub_rna, atac=sub_atac, label_key=label_key)


def build_domain_split_bundle(rna, atac, label_key, split_seed, domain_name):
    split_indices = _build_split_indices(rna.n_obs, split_seed, domain_name)
    split_obs_names = {
        split_name: np.asarray(rna.obs_names)[indices]
        for split_name, indices in split_indices.items()
    }
    full_domain = _subset_domain(rna, atac, rna.obs_names, label_key)
    train_domain = _subset_domain(rna, atac, split_obs_names["train"], label_key)
    val_domain = _subset_domain(rna, atac, split_obs_names["val"], label_key)
    test_domain = _subset_domain(rna, atac, split_obs_names["test"], label_key)
    eval_domain = _subset_domain(rna, atac, split_obs_names["eval"], label_key)

    return DomainSplitBundle(
        full=full_domain,
        train=train_domain,
        val=val_domain,
        test=test_domain,
        eval=eval_domain,
        split_indices=split_indices,
        split_obs_names=split_obs_names,
        split_seed=split_seed,
    )


def configure_source_train_eval_bundle(source_split_bundle, use_source_split_train_eval):
    if use_source_split_train_eval:
        return source_split_bundle

    full_obs_names = np.asarray(source_split_bundle.full.rna.obs_names)
    full_indices = np.arange(source_split_bundle.full.rna.n_obs)
    source_split_bundle.train = source_split_bundle.full
    source_split_bundle.eval = source_split_bundle.full
    source_split_bundle.split_obs_names["train"] = full_obs_names
    source_split_bundle.split_obs_names["eval"] = full_obs_names
    source_split_bundle.split_indices["train"] = full_indices
    source_split_bundle.split_indices["eval"] = full_indices
    return source_split_bundle


def build_split_metadata(source_split_bundle, target_split_bundle):
    def _domain_payload(bundle):
        return {
            "n_obs": int(bundle.full.rna.n_obs),
            "seed": int(bundle.split_seed),
            "splits": {
                split_name: {
                    "indices": [int(v) for v in bundle.split_indices[split_name].tolist()],
                    "obs_names": [str(v) for v in bundle.split_obs_names[split_name].tolist()],
                    "n_obs": int(len(bundle.split_indices[split_name])),
                }
                for split_name in ("train", "val", "test", "eval")
            },
        }

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "evaluation_split": "val_plus_test",
        "ratios": {
            "train": SPLIT_RATIO_TRAIN,
            "val": SPLIT_RATIO_VAL,
            "test": SPLIT_RATIO_TEST,
        },
        "domains": {
            "source": _domain_payload(source_split_bundle),
            "target": _domain_payload(target_split_bundle),
        },
    }


def log_split_metadata_artifact(split_metadata):
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = os.path.dirname(SPLIT_ARTIFACT_PATH)
        artifact_name = os.path.basename(SPLIT_ARTIFACT_PATH)
        local_path = os.path.join(tmpdir, artifact_name)
        with open(local_path, "w") as f:
            json.dump(split_metadata, f, indent=2)
        mlflow.log_artifact(local_path, artifact_path=artifact_dir)


def apply_hvg_and_gp_filtering(
    source_rna,
    source_atac,
    target_rna,
    target_atac,
    gp_net,
    top_n_genes,
    top_n_peaks,
    rank_type="fused",
):
    """Shared source/target feature filtering logic used by training and co-embed scripts."""
    assert rank_type in ["fused", "source", "target"], "rank_type must be 'fused' or 'source' or 'target'"

    gp_net_genes = gp_net["Gene"].unique()
    gp_net_peaks = gp_net["Peak"].unique()

    source_rna = source_rna[:, source_rna.var_names.isin(gp_net_genes)].copy()
    source_atac = source_atac[:, source_atac.var_names.isin(gp_net_peaks)].copy()
    target_rna = target_rna[:, target_rna.var_names.isin(gp_net_genes)].copy()
    target_atac = target_atac[:, target_atac.var_names.isin(gp_net_peaks)].copy()

    source_rna.var["highly_variable"] = False
    source_atac.var["highly_variable"] = False
    target_rna.var["highly_variable"] = False
    target_atac.var["highly_variable"] = False

    source_rna.var["highly_variable_rank"] = source_rna.var["dispersions_norm"].rank(ascending=False)
    target_rna.var["highly_variable_rank"] = target_rna.var["dispersions_norm"].rank(ascending=False)
    source_atac.var["highly_variable_rank"] = source_atac.var["dispersions_norm"].rank(ascending=False)
    target_atac.var["highly_variable_rank"] = target_atac.var["dispersions_norm"].rank(ascending=False)

    # Compute combined rank. Note that may have more than n_top_genes/peaks due to rank ties.
    def _rank_fused(source_adata, target_adata, local_rank_type):
        if local_rank_type == "fused":
            return pd.concat(
                [
                    source_adata.var["highly_variable_rank"],
                    target_adata.var["highly_variable_rank"],
                ],
                axis=1,
            ).mean(axis=1).rank(ascending=True, method="min")
        elif local_rank_type == "source":
            return source_adata.var["highly_variable_rank"]
        elif local_rank_type == "target":
            return target_adata.var["highly_variable_rank"]

    # Compute combined rank for genes.
    rna_combined_rank = _rank_fused(source_rna, target_rna, rank_type)
    gene_filt = rna_combined_rank.le(top_n_genes)
    source_rna.var.loc[gene_filt, "highly_variable"] = True
    target_rna.var.loc[gene_filt, "highly_variable"] = True

    # Filter peaks in-cis with filtered genes.
    peak_filt = gp_net.loc[
        gp_net["Gene"].isin(gene_filt.loc[gene_filt].index),
        "Peak",
    ].unique()

    # Compute combined rank for peaks.
    atac_combined_rank = _rank_fused(source_atac, target_atac, rank_type)
    atac_combined_rank_filt = atac_combined_rank.loc[source_atac.var_names.isin(peak_filt)]
    atac_combined_rank_filt = atac_combined_rank_filt.rank(ascending=True, method="min")
    peak_filt = atac_combined_rank_filt.le(top_n_peaks)
    peak_filt = peak_filt.loc[peak_filt].index

    source_atac.var.loc[source_atac.var_names.isin(peak_filt), "highly_variable"] = True
    target_atac.var.loc[target_atac.var_names.isin(peak_filt), "highly_variable"] = True

    # Re-introduce gp-net based on filtered genes and peaks.
    gp_net = gp_net[
        gp_net["Gene"].isin(gene_filt.loc[gene_filt].index)
        & gp_net["Peak"].isin(peak_filt)
    ]
    source_rna.uns["gene_peak_Net"] = gp_net.copy()
    target_rna.uns["gene_peak_Net"] = gp_net.copy()

    return source_rna, source_atac, target_rna, target_atac, gp_net


def prepare_target_for_spatial_graph_type(
    target_rna,
    target_atac,
    source_rna,
    source_atac,
    spatial_graph_type,
    gtf_path,
):
    """Shared target graph preparation used by training and co-embed scripts."""
    if spatial_graph_type == "spatial":
        MultiGATE.Cal_Spatial_Net(target_rna, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Cal_Spatial_Net(target_atac, rad_cutoff=40)
        MultiGATE.Stats_Spatial_Net(target_atac)
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        MultiGATE.Cal_gene_peak_Net_new(target_rna, target_atac, 150000, file=gtf_path)
        target_rna.uns["gene_peak_Net"] = target_atac.uns["gene_peak_Net"]

    elif spatial_graph_type == "tangram":
        target_rna = target_rna[:, target_rna.var_names.isin(source_rna.var_names)].copy()
        target_atac = target_atac[:, target_atac.var_names.isin(source_atac.var_names)].copy()
        tangram_net = pd.read_csv(os.path.join(os.getenv("OUTPATH"), "tangram", "tangram_spatial_net_affinity.csv"))
        target_rna.uns["Spatial_Net"] = tangram_net.copy()
        target_atac.uns["Spatial_Net"] = tangram_net.copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"].copy()
        target_atac.uns["gene_peak_Net"] = source_atac.uns["gene_peak_Net"].copy()

    elif spatial_graph_type == "knn":
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        build_knn_graph_as_spatial_net(target_rna, n_neighbors=15)
        target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
        MultiGATE.Stats_Spatial_Net(target_rna)
        MultiGATE.Stats_Spatial_Net(target_atac)

    elif spatial_graph_type == "identity":
        target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
        target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
        target_rna.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_atac.uns["gene_peak_Net"] = source_rna.uns["gene_peak_Net"]
        target_rna.uns["Spatial_Net"] = pd.DataFrame(columns=["Cell1", "Cell2", "Distance"])
        target_atac.uns["Spatial_Net"] = target_rna.uns["Spatial_Net"].copy()
    else:
        raise ValueError("Unknown spatial_graph_type '{}'".format(spatial_graph_type))

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
    embedding_key="MultiGATE",
):
    scib_backend = require_scib_backend()
    Benchmarker = scib_backend["Benchmarker"]
    BioConservation = scib_backend["BioConservation"]
    BatchCorrection = scib_backend["BatchCorrection"]

    concat_adata = build_concat_adata_for_umap(rna_adata, atac_adata, embedding_key=embedding_key)
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


def compute_concat_umap(
    concat_adata,
    n_neighbors=10,
    resolution=1.5,
    deterministic=False,
    random_state=0,
):
    if deterministic:
        sc.pp.neighbors(
            concat_adata,
            n_neighbors=n_neighbors,
            use_rep="X",
            knn=True,
            method="umap",
            metric="euclidean",
            random_state=random_state,
        )
        sc.tl.umap(concat_adata, random_state=random_state, init_pos="spectral")
        sc.tl.leiden(concat_adata, resolution=resolution, random_state=random_state)
    else:
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
    embedding_key="MultiGATE",
):
    source_concat_adata = build_concat_adata_for_umap(source_rna, source_atac, embedding_key=embedding_key)
    target_concat_adata = build_concat_adata_for_umap(target_rna, target_atac, embedding_key=embedding_key)

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
            embedding_key=embedding_key,
            n_neighbors=10,
            resolution=1.5,
        )
        target_mdata = build_mudata_with_umap(
            target_rna,
            target_atac,
            embedding_key=embedding_key,
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


def compute_balanced_source_target_mmd(source_rna_embed, source_atac_embed, target_rna_embed, target_atac_embed):
    x = np.concatenate([source_rna_embed, source_atac_embed], axis=0)
    y = np.concatenate([target_rna_embed, target_atac_embed], axis=0)
    min_n = min(x.shape[0], y.shape[0])
    x = x[:min_n, :]
    y = y[:min_n, :]
    mmd = MMD(var=1.0)
    mmd.reset()
    mmd.update((torch.from_numpy(x), torch.from_numpy(y)))
    return float(mmd.compute())



def resolve_run_id_from_name(client, run_name, experiment_name="multigate_mouse_brain_live_zeroshot"):
    """
    Look up the run ID for a given run name within the MultiGATE experiment.
    Raises ValueError if no run or more than one run is found.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            "MLflow experiment '{}' not found. Ensure tracking URI is configured correctly.".format(
                experiment_name
            )
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.run_name = '{}'".format(run_name),
    )
    if len(runs) == 0:
        raise ValueError(
            "No run named '{}' found in experiment '{}'.".format(run_name, experiment_name)
        )
    if len(runs) > 1:
        run_ids = [run.info.run_id for run in runs]
        raise ValueError(
            "Multiple runs named '{}' found in experiment '{}': {}.".format(
                run_name, experiment_name, run_ids
            )
        )

    run_id = runs[0].info.run_id
    print("Resolved run name '{}' -> run ID: {}".format(run_name, run_id))
    return run_id


def load_run_params(client, run_id, run_name):
    run = client.get_run(run_id)
    params = run.data.params
    print("\nRun params for '{}' (ID: {}):".format(run_name, run_id))
    pprint(params)
    return params


def artifact_exists(client, run_id, artifact_path):
    parent_dir = os.path.dirname(artifact_path)
    try:
        artifacts = client.list_artifacts(run_id, parent_dir)
        return any(artifact.path == artifact_path for artifact in artifacts)
    except Exception:
        return False


def download_model_artifact(client, run_id, artifact_name, dst_dir):
    artifact_path = "models/{}".format(artifact_name)
    if not artifact_exists(client, run_id, artifact_path):
        raise FileNotFoundError(
            "Artifact '{}' not found for run {}.".format(artifact_path, run_id)
        )
    print("  Downloading {} ...".format(artifact_path))
    return client.download_artifacts(run_id, artifact_path, dst_dir)


def hidden_dims_from_state_dict(state_dict, w_prefix):
    keys = sorted(
        [key for key in state_dict if key.startswith(w_prefix + ".")],
        key=lambda key: int(key.split(".")[1]),
    )
    if not keys:
        raise ValueError("No '{}' weights found in state dict.".format(w_prefix))

    dims = [state_dict[keys[0]].shape[0]]
    for key in keys:
        dims.append(state_dict[key].shape[1])
    return dims


def _extract_linear_decoder_kwargs_from_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    alpha = state_dict.get("alpha")
    if alpha is not None:
        kwargs["etm_emb_dim"] = int(alpha.shape[1])

    rho_is_fixed_tensor = state_dict.get("rho_is_fixed_mask")
    rho_is_fixed = bool(int(torch.as_tensor(rho_is_fixed_tensor).item())) if rho_is_fixed_tensor is not None else False
    if rho_is_fixed and ("rho_rna" in state_dict) and ("rho_atac" in state_dict):
        kwargs["rho_rna_mask"] = state_dict["rho_rna"].detach().cpu().numpy()
        kwargs["rho_atac_mask"] = state_dict["rho_atac"].detach().cpu().numpy()
    return kwargs


def build_stage1_trainer_from_state_dict(
    state_dict,
    spot_num,
    n_epochs,
    lr=0.0001,
    gradient_clipping=5,
    weight_decay=0.0001,
    random_seed=2020,
):
    hidden_dims1 = hidden_dims_from_state_dict(state_dict, "W1")
    hidden_dims2 = hidden_dims_from_state_dict(state_dict, "W2")
    temp = float(state_dict.get("logit_scale", torch.tensor(1.0)).item())

    if "vgp0" not in state_dict:
        raise KeyError("Stage-1 state dict missing required key 'vgp0'.")

    feat_num = hidden_dims1[0] + hidden_dims2[0]
    vgp_len = int(state_dict["vgp0"].shape[0])
    if vgp_len == feat_num:
        vgp_anchor_mode = "feature"
    else:
        vgp_anchor_mode = "spot"
        if vgp_len != int(spot_num):
            raise ValueError(
                "Spot-anchored checkpoint expects {} source cells, but current source has {}."
                .format(vgp_len, spot_num)
            )

    trainer = MultiGATETrainer(
        hidden_dims1=hidden_dims1,
        hidden_dims2=hidden_dims2,
        spot_num=int(spot_num),
        temp=temp,
        vgp_anchor_mode=vgp_anchor_mode,
        n_epochs=max(1, int(n_epochs)),
        lr=lr,
        gradient_clipping=gradient_clipping,
        nonlinear=True,
        weight_decay=weight_decay,
        verbose=False,
        random_seed=random_seed,
        **_extract_linear_decoder_kwargs_from_state_dict(state_dict),
    )
    trainer.mgate.load_state_dict(state_dict, strict=False)
    trainer.mgate.eval()
    return trainer, hidden_dims1, hidden_dims2, vgp_anchor_mode

def run_stage2_distillation(
    source_trainer,
    target_train_rna,
    target_train_atac,
    target_train_graph_tf,
    target_train_gp_tf,
    target_train_x1,
    target_train_x2,
    source_eval_rna,
    source_eval_atac,
    source_eval_graph_tf,
    source_eval_gp_tf,
    source_eval_x1,
    source_eval_x2,
    target_eval_rna,
    target_eval_atac,
    target_eval_graph_tf,
    target_eval_gp_tf,
    target_eval_x1,
    target_eval_x2,
    stage2_epochs,
    lambda_kd,
    kd_mix_kl,
    kd_mix_ot,
    target_label_key,
    scib_n_jobs,
    vgp_anchor_mode=None,
):
    if stage2_epochs <= 0:
        return None, None

    emd = require_ot_backend()

    teacher_trainer = build_zero_shot_target_trainer(
        source_trainer,
        target_train_rna.n_obs,
        vgp_anchor_mode=vgp_anchor_mode,
    )
    teacher_model = teacher_trainer.mgate
    teacher_model.eval()
    for teacher_param in teacher_model.parameters():
        teacher_param.requires_grad = False

    student_trainer = MultiGATETrainer(
        hidden_dims1=source_trainer.mgate.hidden_dims1,
        hidden_dims2=source_trainer.mgate.hidden_dims2,
        spot_num=target_train_rna.n_obs,
        temp=float(source_trainer.mgate.logit_scale.detach().cpu().item()),
        vgp_anchor_mode=(vgp_anchor_mode or getattr(source_trainer.mgate, "vgp_anchor_mode", "spot")),
        n_epochs=stage2_epochs,
        lr=source_trainer.lr,
        gradient_clipping=source_trainer.gradient_clipping,
        nonlinear=source_trainer.mgate.nonlinear,
        verbose=False,
        random_seed=2021,
        config={"device": str(source_trainer.device)},
        **_extract_linear_decoder_kwargs_from_mgate(source_trainer.mgate),
    )
    for attr_name in ("pathway_names", "source_pathway_names", "target_pathway_names"):
        if hasattr(source_trainer.mgate, attr_name):
            attr_value = getattr(source_trainer.mgate, attr_name)
            setattr(
                student_trainer.mgate,
                attr_name,
                attr_value.copy() if hasattr(attr_value, "copy") else attr_value,
            )

    target_train_a_t, target_train_prune_t, target_train_gp_t, target_train_x1_t, target_train_x2_t = student_trainer._prepare_inputs(
        target_train_graph_tf,
        target_train_graph_tf,
        target_train_gp_tf,
        target_train_x1,
        target_train_x2,
    )

    parent_run = mlflow.active_run()
    parent_run_name = None
    if parent_run is not None:
        parent_run_name = parent_run.data.tags.get("mlflow.runName")
    stage2_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if parent_run_name:
        stage2_run_name = "{}_stage2_{}".format(parent_run_name, stage2_timestamp)
    else:
        stage2_run_name = "stage2_distillation_{}".format(stage2_timestamp)

    stage2_run_id = None
    with mlflow.start_run(run_name=stage2_run_name, nested=True) as stage2_run:
        stage2_run_id = stage2_run.info.run_id
        mlflow.set_tag("training_stage", "stage2_distillation")
        mlflow.set_tag("teacher_student_distillation", "true")
        mlflow.log_param("hidden_dims", json.dumps(source_trainer.mgate.hidden_dims1[1:])) # skip first dimension (input size)
        mlflow.log_param("stage2_epochs", stage2_epochs)
        mlflow.log_param("lambda_kd", lambda_kd)
        mlflow.log_param("kd_mix_kl", kd_mix_kl)
        mlflow.log_param("kd_mix_ot", kd_mix_ot)
        mlflow.log_param("student_init", "random")
        teacher_init_mode = (
            "source_to_target_zero_shot_copy_vgp"
            if (vgp_anchor_mode or getattr(source_trainer.mgate, "vgp_anchor_mode", "spot")) == "feature"
            else "source_to_target_zero_shot_zero_vgp"
        )
        mlflow.log_param("teacher_init", teacher_init_mode)
        mlflow.log_param("stage2_target_label_key", target_label_key if target_label_key is not None else "None")
        mlflow.log_param("stage2_target_train_cells", int(target_train_rna.n_obs))
        mlflow.log_param("stage2_target_eval_cells", int(target_eval_rna.n_obs))
        target_scib_label_mode_logged = False
        target_scib_effective_label_key_logged = False

        pbar = tqdm(range(1, stage2_epochs + 1), desc="Stage 2 distillation", unit="epoch")
        for epoch in pbar:
            student_trainer.mgate.train()
            student_trainer.optimizer.zero_grad(set_to_none=True)

            student_outputs = student_trainer.mgate(
                target_train_a_t,
                target_train_prune_t,
                target_train_gp_t,
                target_train_x1_t,
                target_train_x2_t,
            )
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    target_train_a_t,
                    target_train_prune_t,
                    target_train_gp_t,
                    target_train_x1_t,
                    target_train_x2_t,
                )

            student_clip_rna, student_clip_atac = student_outputs[5], student_outputs[6]
            teacher_clip_rna, teacher_clip_atac = teacher_outputs[5], teacher_outputs[6]

            student_logits = compute_clip_logits(student_clip_rna, student_clip_atac, student_trainer.mgate.logit_scale)
            teacher_logits = compute_clip_logits(teacher_clip_rna, teacher_clip_atac, teacher_model.logit_scale)

            kd_ot_loss = compute_ot_clip_loss(student_logits, teacher_logits, emd=emd)
            kd_kl_loss = compute_kd_kl_loss(student_logits, teacher_logits)
            kd_kl_loss = kd_kl_loss * 50 # TMP - bring KL loss to same scale as OT loss
            distill_loss = lambda_kd * (kd_mix_kl * kd_kl_loss + kd_mix_ot * kd_ot_loss)

            distill_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_trainer.mgate.parameters(), student_trainer.gradient_clipping)
            student_trainer.optimizer.step()

            mlflow.log_metric("stage2_distill_loss", float(distill_loss.detach().cpu().item()), step=epoch)
            mlflow.log_metric("stage2_kd_kl_loss", float(kd_kl_loss.detach().cpu().item()), step=epoch)
            mlflow.log_metric("stage2_kd_ot_clip_loss", float(kd_ot_loss.detach().cpu().item()), step=epoch)

            loss_val = float(distill_loss.detach().cpu().item())
            pbar.set_postfix({"distill_loss": "{:.4f}".format(loss_val)})

            # Drop training forward tensors before periodic eval; otherwise eval runs
            # with student_outputs/teacher_outputs still resident and OOMs on peak epochs.
            del student_outputs, teacher_outputs
            del student_clip_rna, student_clip_atac, teacher_clip_rna, teacher_clip_atac
            del student_logits, teacher_logits, kd_ot_loss, kd_kl_loss, distill_loss

            if (epoch == 1) or (epoch % 500 == 0) or (epoch == stage2_epochs):
                # Last epoch: teacher is not needed again; infer_* builds extra trainers and peaks VRAM.
                if epoch == stage2_epochs:
                    del teacher_model, teacher_trainer
                    teacher_model = None
                    teacher_trainer = None
                    if str(student_trainer.device).startswith("cuda"):
                        torch.cuda.empty_cache()

                source_eval_embeddings, target_eval_embeddings, target_eval_trainer = infer_source_and_zero_shot_target_embeddings(
                    source_trainer=student_trainer,
                    source_graph_tf=source_eval_graph_tf,
                    source_gp_tf=source_eval_gp_tf,
                    source_x1=source_eval_x1,
                    source_x2=source_eval_x2,
                    target_graph_tf=target_eval_graph_tf,
                    target_gp_tf=target_eval_gp_tf,
                    target_x1=target_eval_x1,
                    target_x2=target_eval_x2,
                    target_spot_num=target_eval_rna.n_obs,
                    vgp_anchor_mode=vgp_anchor_mode,
                )
                source_eval_rna_embeddings = source_eval_embeddings[0]
                source_eval_atac_embeddings = source_eval_embeddings[1]
                target_eval_rna_embeddings = target_eval_embeddings[0]
                target_eval_atac_embeddings = target_eval_embeddings[1]

                set_multigate_embeddings(
                    target_eval_rna,
                    target_eval_atac,
                    target_eval_rna_embeddings,
                    target_eval_atac_embeddings,
                    key_added="MultiGATE",
                )

                # compute and log scib metrics for target data
                target_scib_metrics = compute_scib_metrics_for_domain(
                    rna_adata=target_eval_rna,
                    atac_adata=target_eval_atac,
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

                stage2_mmd_value = compute_balanced_source_target_mmd(
                    source_eval_rna_embeddings,
                    source_eval_atac_embeddings,
                    target_eval_rna_embeddings,
                    target_eval_atac_embeddings,
                )
                mlflow.log_metric("stage2_source_target_balanced_mmd", stage2_mmd_value, step=epoch)
                del source_eval_embeddings, target_eval_embeddings, target_eval_trainer
                del source_eval_rna_embeddings, source_eval_atac_embeddings, target_eval_rna_embeddings, target_eval_atac_embeddings, stage2_mmd_value

    final_target_embeddings, final_target_trainer = infer_target_embeddings_from_source_trainer(
        source_trainer=student_trainer,
        target_graph_tf=target_eval_graph_tf,
        target_gp_tf=target_eval_gp_tf,
        target_x1=target_eval_x1,
        target_x2=target_eval_x2,
        target_spot_num=target_eval_rna.n_obs,
        vgp_anchor_mode=vgp_anchor_mode,
    )
    set_multigate_embeddings(
        target_eval_rna,
        target_eval_atac,
        final_target_embeddings[0],
        final_target_embeddings[1],
        key_added="MultiGATE",
    )
    del final_target_trainer

    return student_trainer, stage2_run_id


def log_torch_state_dict_artifacts(tmpdir, state_dicts):
    for artifact_name, state_dict in state_dicts.items():
        if state_dict is None:
            continue
        local_path = os.path.join(tmpdir, artifact_name)
        torch.save(state_dict, local_path)
        mlflow.log_artifact(local_path, artifact_path="models")


def log_sparse_matrix_artifacts(tmpdir, matrices):
    for artifact_name, matrix in matrices.items():
        if matrix is None:
            continue
        local_path = os.path.join(tmpdir, artifact_name)
        sp.save_npz(local_path, matrix)
        mlflow.log_artifact(local_path, artifact_path="matrices")


def log_stage2_artifacts_for_run(
    stage2_run_id,
    stage2_trainer,
    source_rna,
    source_atac,
    target_rna,
    target_atac,
    target_graph_tf,
    target_gp_tf,
    target_x1,
    target_x2,
    log_mudata_umaps,
):
    if stage2_run_id is None:
        raise ValueError("stage2_run_id is required to log stage-2 artifacts.")

    with mlflow.start_run(run_id=stage2_run_id, nested=True):
        log_stage_umap_artifacts(
            source_rna=source_rna,
            source_atac=source_atac,
            target_rna=target_rna,
            target_atac=target_atac,
            stage_label="stage2",
            log_mudata_umaps=log_mudata_umaps,
        )

        stage2_target_embeddings, stage2_target_infer_trainer = infer_target_embeddings_from_source_trainer(
            source_trainer=stage2_trainer,
            target_graph_tf=target_graph_tf,
            target_gp_tf=target_gp_tf,
            target_x1=target_x1,
            target_x2=target_x2,
            target_spot_num=target_x1.shape[0],
            vgp_anchor_mode=getattr(stage2_trainer.mgate, "vgp_anchor_mode", "spot"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_torch_state_dict_artifacts(
                tmpdir,
                {
                    "model_stage2.pth": stage2_trainer.mgate.state_dict(),
                },
            )
            if not stage2_trainer.mgate.skip_gp_attention:
                log_sparse_matrix_artifacts(
                    tmpdir,
                    {
                        "target_peak_gene_attention.npz": stage2_target_embeddings[4][0],
                    },
                )
        del stage2_target_infer_trainer


def _mlflow_namespace_dir():
    """MultiGATE artifact / local-DB root under MLFLOW_BASE_DIR."""
    env_mlflow_base_dir = os.environ.get("MLFLOW_BASE_DIR", os.path.expanduser("~/services/mlflow/artifacts"))
    mlflow_base_dir = os.path.abspath(env_mlflow_base_dir)
    if os.path.basename(mlflow_base_dir.rstrip(os.sep)) != "MultiGATE":
        mlflow_base_dir = os.path.join(mlflow_base_dir, "MultiGATE")
    return mlflow_base_dir


def _mlflow_tracking_connection_failed(exc):
    msg = str(exc).lower()
    return any(
        fragment in msg
        for fragment in (
            "connection refused",
            "failed to establish",
            "connection error",
            "max retries exceeded",
            "newconnectionerror",
        )
    )


def setup_mlflow():
    require_runtime_bootstrap()

    mlflow_base_dir = _mlflow_namespace_dir()
    os.makedirs(mlflow_base_dir, exist_ok=True)
    print("MLflow artifact dir:", mlflow_base_dir)

    # Optional: log to a local SQLite file instead of the HTTP tracking server (no postgres/mlflow
    # process needed).  export MLFLOW_OFFLINE_SQLITE=1
    offline = os.environ.get("MLFLOW_OFFLINE_SQLITE", "").lower() in ("1", "true", "yes")
    if offline:
        mlflow_db_path = os.path.join(mlflow_base_dir, "mlflow.db")
        tracking_uri = "sqlite:///{}".format(mlflow_db_path)
        print("MLFLOW_OFFLINE_SQLITE: using local backend", tracking_uri)
    else:
        # Point to the MLflow tracking server (PostgreSQL-backed).
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    mlflow.set_tracking_uri(tracking_uri)
    print("MLflow tracking URI:", tracking_uri)

    experiment_name = "multigate_mouse_brain_live_zeroshot"
    artifact_dir = os.path.join(mlflow_base_dir, experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    try:
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
    except MlflowException as e:
        if offline or not _mlflow_tracking_connection_failed(e):
            raise
        baklava_root = os.path.join(os.path.dirname(REPO_ROOT), "BAKLAVA")
        starter = os.path.join(baklava_root, "scripts", "start_mlflow_services.sh")
        raise RuntimeError(
            "Cannot reach the MLflow tracking server at {!r} (connection refused or unreachable). "
            "Start PostgreSQL + the MLflow server first, e.g. in tmux:\n"
            "  conda activate nichecompass  # or your project env\n"
            "  bash {}\n"
            "Or run without a server by setting:\n"
            "  export MLFLOW_OFFLINE_SQLITE=1\n"
            "(uses a SQLite DB under the MultiGATE artifact directory; not for multi-node jobs.)"
            .format(tracking_uri, starter)
        ) from e

    return experiment_id


def is_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        if shell == "TerminalInteractiveShell":
            return False
        return False
    except Exception:
        return False


def validate_args(args):
    if args.split_seed < 0:
        raise ValueError("--split-seed must be a non-negative integer.")
    if args.stage1_epochs <= 0:
        raise ValueError("--stage1-epochs must be a positive integer.")
    if args.stage2_epochs < 0:
        raise ValueError("--stage2-epochs must be a non-negative integer.")
    if args.lambda_kd < 0:
        raise ValueError("--lambda-kd must be non-negative.")
    if args.kd_mix_kl < 0 or args.kd_mix_ot < 0:
        raise ValueError("--kd-mix-kl and --kd-mix-ot must be non-negative.")
    if args.kd_mix_kl + args.kd_mix_ot <= 0:
        raise ValueError("--kd-mix-kl and --kd-mix-ot must sum to a positive value.")
    if args.scib_n_jobs <= 0:
        raise ValueError("--scib-n-jobs must be a positive integer.")


def resolve_stage1_cache_config(args, mlflow_client):
    cache_config = Stage1CacheConfig(
        use_cache=args.stage1_mlflow_cache_run_name is not None,
        run_name=args.stage1_mlflow_cache_run_name,
        dual_source_kd=bool(args.stage1_dual_source_kd),
        student_graph_type=args.spatial_graph_type,
        vgp_anchor_mode=args.vgp_anchor_mode,
    )

    if cache_config.use_cache:
        cache_config.run_id = resolve_run_id_from_name(mlflow_client, cache_config.run_name)
        cache_config.run_params = load_run_params(mlflow_client, cache_config.run_id, cache_config.run_name)

        cache_config.dual_source_kd = (
            str(cache_config.run_params.get("stage1_dual_source_kd", "False")).lower() == "true"
        )
        cache_config.student_graph_type = cache_config.run_params.get("stage1_student_graph", "identity")
        if cache_config.student_graph_type in {"NA", "na", "None", "none", "", None}:
            cache_config.student_graph_type = "identity"

        cached_vgp_anchor_mode = cache_config.run_params.get("vgp_anchor_mode")
        if cached_vgp_anchor_mode in {"spot", "feature"}:
            cache_config.vgp_anchor_mode = cached_vgp_anchor_mode

        print(
            "[Stage1 Cache] run='{}' (id={}), dual_source_kd={}, student_graph={}, vgp_anchor_mode={}".format(
                cache_config.run_name,
                cache_config.run_id,
                cache_config.dual_source_kd,
                cache_config.student_graph_type,
                cache_config.vgp_anchor_mode,
            )
        )

    if cache_config.dual_source_kd and cache_config.student_graph_type == "tangram":
        raise ValueError(
            "Stage-1 dual-source KD with tangram student graph is not supported for source training. "
            "Use spatial, knn, or identity."
        )

    return cache_config


def resolve_domain_label_key(adata, requested_key, default_key, domain_name):
    if requested_key is not None:
        if requested_key in adata.obs.columns:
            adata.obs[requested_key] = adata.obs[requested_key].astype("category")
            return requested_key
        warnings.warn(
            "Requested {} label key '{}' not found. Falling back to default key '{}' if available.".format(
                domain_name,
                requested_key,
                default_key,
            )
        )

    if default_key in adata.obs.columns:
        adata.obs[default_key] = adata.obs[default_key].astype("category")
        return default_key

    warnings.warn(
        "No usable {} label key found (requested='{}', default='{}'). scIB will fall back to pseudo labels.".format(
            domain_name,
            requested_key,
            default_key,
        )
    )
    return None


def load_and_prepare_data_bundle(args):
    require_runtime_bootstrap()

    combined_gp_dict = None
    if getattr(args, "combined_gp_dict", True):
        combined_gp_dict = load_nichecompass_combined_gp_dict_mouse(verbose=True, load_from_disk=True)

    source_rna = sc.read_h5ad(os.path.join(BASE_PATH, "source_rna_aligned.h5ad"))
    source_atac = sc.read_h5ad(os.path.join(BASE_PATH, "source_atac_aligned.h5ad"))
    source_rna.obsm["spatial"] = source_rna.obsm["spatial"] * -1
    source_atac.obsm["spatial"] = source_atac.obsm["spatial"] * -1

    target_rna = sc.read_h5ad(os.path.join(BASE_PATH, "target_rna_aligned.h5ad"))
    target_atac = sc.read_h5ad(os.path.join(BASE_PATH, "target_atac_aligned.h5ad"))
    assert target_rna.obs_names.equals(target_atac.obs_names), "Target RNA and ATAC must have matching obs_names"

    source_label_key = resolve_domain_label_key(
        source_rna,
        requested_key=args.source_label_key,
        default_key=DEFAULT_SOURCE_LABEL_KEY,
        domain_name="source",
    )
    target_label_key = resolve_domain_label_key(
        target_rna,
        requested_key=args.target_label_key,
        default_key=DEFAULT_TARGET_LABEL_KEY,
        domain_name="target",
    )
    source_rna.uns["label_key"] = source_label_key
    target_rna.uns["label_key"] = target_label_key

    gtf_path = os.path.join(
        os.getenv("DATAPATH"),
        "gene_annotations",
        "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    )
    if not os.path.exists(gtf_path):
        raise FileNotFoundError("GTF annotation file not found: {}".format(gtf_path))

    MultiGATE.Cal_gene_peak_Net_new(source_rna, source_atac, 150000, file=gtf_path)
    gp_net = source_atac.uns["gene_peak_Net"].copy()
    del source_atac.uns["gene_peak_Net"]

    source_rna, source_atac, target_rna, target_atac, gp_net = apply_hvg_and_gp_filtering(
        source_rna=source_rna,
        source_atac=source_atac,
        target_rna=target_rna,
        target_atac=target_atac,
        gp_net=gp_net,
        top_n_genes=args.top_n_genes,
        top_n_peaks=args.top_n_peaks,
        rank_type="fused",
    )
    print("Filtered {} genes and {} peaks from gene-peak net".format(len(target_rna.var_names), len(target_atac.var_names)))
    del gp_net

    MultiGATE.Cal_Spatial_Net(source_rna, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_rna)
    MultiGATE.Cal_Spatial_Net(source_atac, rad_cutoff=40)
    MultiGATE.Stats_Spatial_Net(source_atac)
    source_rna = source_rna[:, source_rna.var["highly_variable"]].copy()
    source_atac = source_atac[:, source_atac.var["highly_variable"]].copy()

    target_rna, target_atac = prepare_target_for_spatial_graph_type(
        target_rna=target_rna,
        target_atac=target_atac,
        source_rna=source_rna,
        source_atac=source_atac,
        spatial_graph_type=args.spatial_graph_type,
        gtf_path=gtf_path,
    )
    source_rna, source_atac = pair_modalities(source_rna, source_atac, domain_name="Source")
    target_rna, target_atac = pair_modalities(target_rna, target_atac, domain_name="Target")

    source_split_bundle = build_domain_split_bundle(
        source_rna,
        source_atac,
        label_key=source_label_key,
        split_seed=int(args.split_seed),
        domain_name="Source",
    )
    target_split_bundle = build_domain_split_bundle(
        target_rna,
        target_atac,
        label_key=target_label_key,
        split_seed=int(args.split_seed + 1),
        domain_name="Target",
    )

    data_bundle = DataBundle(
        source=source_split_bundle,
        target=target_split_bundle,
        split_metadata={},
        combined_gp_dict=combined_gp_dict,
    )

    if args.switcharoo:
        data_bundle = DataBundle(
            source=data_bundle.target,
            target=data_bundle.source,
            split_metadata={},
            combined_gp_dict=combined_gp_dict,
        )

    data_bundle.source = configure_source_train_eval_bundle(
        data_bundle.source,
        use_source_split_train_eval=bool(args.source_split_train_eval),
    )
    data_bundle.split_metadata = build_split_metadata(data_bundle.source, data_bundle.target)

    return data_bundle


def build_graph_bundle_from_domains(source_domain, target_domain, bp_width=400, graph_type="ATAC", protein_value=0.001):
    source_graph_tf, source_gp_tf, source_x1, source_x2 = build_graph_inputs(
        source_domain.rna,
        source_domain.atac,
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )
    target_graph_tf, target_gp_tf, target_x1, target_x2 = build_graph_inputs(
        target_domain.rna,
        target_domain.atac,
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

    return GraphBundle(
        source=GraphInputBundle(graph_tf=source_graph_tf, gp_tf=source_gp_tf, x1=source_x1, x2=source_x2),
        target=GraphInputBundle(graph_tf=target_graph_tf, gp_tf=target_gp_tf, x1=target_x1, x2=target_x2),
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )


def build_graph_bundle(data_bundle, bp_width=400, graph_type="ATAC", protein_value=0.001):
    return build_graph_bundle_from_domains(
        source_domain=data_bundle.source.train,
        target_domain=data_bundle.target.train,
        bp_width=bp_width,
        graph_type=graph_type,
        protein_value=protein_value,
    )


def create_multigate_trainer(
    graph_inputs,
    hidden_dims,
    n_epochs,
    vgp_anchor_mode,
    random_seed,
    skip_gp_attention=True,
    device=None,
    pathway_decoder_masks: Optional[PathwayDecoderMaskBundle]=None,
):
    trainer_kwargs = {
        "hidden_dims1": [graph_inputs.x1.shape[1]] + hidden_dims,
        "hidden_dims2": [graph_inputs.x2.shape[1]] + hidden_dims,
        "spot_num": graph_inputs.x1.shape[0],
        "temp": 1,
        "vgp_anchor_mode": vgp_anchor_mode,
        "n_epochs": n_epochs,
        "lr": 0.0001,
        "gradient_clipping": 5,
        "nonlinear": True,
        "weight_decay": 0.0001,
        "verbose": False,
        "random_seed": random_seed,
        "skip_gp_attention": skip_gp_attention,
    }
    if pathway_decoder_masks is not None:
        trainer_kwargs["etm_emb_dim"] = int(pathway_decoder_masks.rho_rna_mask.shape[0])
        trainer_kwargs["rho_rna_mask"] = pathway_decoder_masks.rho_rna_mask
        trainer_kwargs["rho_atac_mask"] = pathway_decoder_masks.rho_atac_mask
    if device is not None:
        trainer_kwargs["config"] = {"device": str(device)}
    trainer = MultiGATETrainer(**trainer_kwargs)
    if pathway_decoder_masks is not None:
        trainer.mgate.pathway_names = pathway_decoder_masks.pathway_names.copy()
        trainer.mgate.source_pathway_names = pathway_decoder_masks.source_pathway_names.copy()
        trainer.mgate.target_pathway_names = pathway_decoder_masks.target_pathway_names.copy()
    return trainer


def initialize_stage1_trainers_for_training(data_bundle, graph_bundle, cache_config, num_epochs):
    hidden_dims = [512, 30]
    pathway_decoder_masks = maybe_build_pathway_decoder_masks(
        data_bundle=data_bundle,
        graph_bundle=graph_bundle,
    )
    teacher_trainer = create_multigate_trainer(
        graph_bundle.source,
        hidden_dims=hidden_dims,
        n_epochs=num_epochs,
        vgp_anchor_mode=cache_config.vgp_anchor_mode,
        random_seed=2020,
        skip_gp_attention=True,
        pathway_decoder_masks=pathway_decoder_masks,
    )
    source_inputs_tensors = teacher_trainer._prepare_inputs(
        graph_bundle.source.graph_tf,
        graph_bundle.source.graph_tf,
        graph_bundle.source.gp_tf,
        graph_bundle.source.x1,
        graph_bundle.source.x2,
    )

    student_trainer = None
    nonspatial_trainer = None
    source_student_graph_tf = None
    source_student_inputs_tensors = None

    if cache_config.dual_source_kd:
        source_student_graph_tf = build_source_student_graph_tf(
            source_rna=data_bundle.source.rna,
            spatial_graph_type=cache_config.student_graph_type,
        )
        student_trainer = create_multigate_trainer(
            graph_bundle.source,
            hidden_dims=hidden_dims,
            n_epochs=num_epochs,
            vgp_anchor_mode=cache_config.vgp_anchor_mode,
            random_seed=2021,
            device=teacher_trainer.device,
            pathway_decoder_masks=pathway_decoder_masks,
        )
        source_student_inputs_tensors = student_trainer._prepare_inputs(
            source_student_graph_tf,
            source_student_graph_tf,
            graph_bundle.source.gp_tf,
            graph_bundle.source.x1,
            graph_bundle.source.x2,
        )
        nonspatial_trainer = create_multigate_trainer(
            graph_bundle.source,
            hidden_dims=hidden_dims,
            n_epochs=num_epochs,
            vgp_anchor_mode=cache_config.vgp_anchor_mode,
            random_seed=2022,
            device=teacher_trainer.device,
            pathway_decoder_masks=pathway_decoder_masks,
        )

    primary_trainer = student_trainer if cache_config.dual_source_kd else teacher_trainer
    primary_source_graph_tf = source_student_graph_tf if cache_config.dual_source_kd else graph_bundle.source.graph_tf
    primary_model_name = "student" if cache_config.dual_source_kd else "teacher"

    return Stage1TrainerBundle(
        teacher=teacher_trainer,
        student=student_trainer,
        nonspatial=nonspatial_trainer,
        source_inputs_tensors=source_inputs_tensors,
        source_student_inputs_tensors=source_student_inputs_tensors,
        source_student_graph_tf=source_student_graph_tf,
        primary=primary_trainer,
        primary_model_name=primary_model_name,
        primary_source_graph_tf=primary_source_graph_tf,
    )


def load_cached_stage1_primary_trainer(data_bundle, graph_bundle, cache_config, mlflow_client, num_epochs):
    source_student_graph_tf = None
    if cache_config.dual_source_kd:
        source_student_graph_tf = build_source_student_graph_tf(
            source_rna=data_bundle.source.rna,
            spatial_graph_type=cache_config.student_graph_type,
        )
    primary_source_graph_tf = source_student_graph_tf if cache_config.dual_source_kd else graph_bundle.source.graph_tf

    with tempfile.TemporaryDirectory() as tmpdir:
        local_stage1_path = download_model_artifact(
            mlflow_client,
            cache_config.run_id,
            "model_stage1.pth",
            tmpdir,
        )
        stage1_state_dict = torch.load(local_stage1_path, map_location="cpu", weights_only=False)

    stage1_primary_trainer, hidden_dims1_loaded, hidden_dims2_loaded, inferred_vgp_anchor_mode = build_stage1_trainer_from_state_dict(
        stage1_state_dict,
        spot_num=graph_bundle.source.x1.shape[0],
        n_epochs=num_epochs,
        lr=0.0001,
        gradient_clipping=5,
        weight_decay=0.0001,
        random_seed=2020,
    )

    if hidden_dims1_loaded[0] != graph_bundle.source.x1.shape[1] or hidden_dims2_loaded[0] != graph_bundle.source.x2.shape[1]:
        raise ValueError(
            "Feature dimension mismatch for cached stage-1 model: checkpoint expects "
            "RNA {} / ATAC {}, current data is RNA {} / ATAC {}."
            .format(
                hidden_dims1_loaded[0],
                hidden_dims2_loaded[0],
                graph_bundle.source.x1.shape[1],
                graph_bundle.source.x2.shape[1],
            )
        )

    cache_config.vgp_anchor_mode = inferred_vgp_anchor_mode
    if "skip_gp_attention" in cache_config.run_params:
        stage1_primary_trainer.mgate.skip_gp_attention = (
            str(cache_config.run_params["skip_gp_attention"]).lower() == "true"
        )
    primary_model_name = cache_config.run_params.get(
        "stage1_primary_model",
        "student" if cache_config.dual_source_kd else "teacher",
    )
    print(
        "[Stage1 Cache] Loaded model_stage1.pth with hidden_dims1={}, hidden_dims2={}, vgp_anchor_mode={}".format(
            hidden_dims1_loaded,
            hidden_dims2_loaded,
            inferred_vgp_anchor_mode,
        )
    )

    return Stage1TrainerBundle(
        teacher=None,
        student=None,
        nonspatial=None,
        source_inputs_tensors=None,
        source_student_inputs_tensors=None,
        source_student_graph_tf=source_student_graph_tf,
        primary=stage1_primary_trainer,
        primary_model_name=primary_model_name,
        primary_source_graph_tf=primary_source_graph_tf,
    )


def maybe_run_stage2_and_log(
    args,
    data_bundle,
    train_graph_bundle,
    eval_graph_bundle,
    source_eval_graph_tf,
    trainer_bundle,
    source_eval_embeddings,
    vgp_anchor_mode,
):
    if args.stage2_epochs <= 0:
        print("[Stage2 KD] Skipped because --stage2-epochs is 0.")
        return None, None

    print(
        "[Stage2 KD] Starting target distillation for {} epochs (lambda_kd={}, kd_mix_kl={}, kd_mix_ot={})".format(
            args.stage2_epochs,
            args.lambda_kd,
            args.kd_mix_kl,
            args.kd_mix_ot,
        )
    )

    stage2_trainer, stage2_run_id = run_stage2_distillation(
        source_trainer=trainer_bundle.primary,
        target_train_rna=data_bundle.target.train.rna,
        target_train_atac=data_bundle.target.train.atac,
        target_train_graph_tf=train_graph_bundle.target.graph_tf,
        target_train_gp_tf=train_graph_bundle.target.gp_tf,
        target_train_x1=train_graph_bundle.target.x1,
        target_train_x2=train_graph_bundle.target.x2,
        source_eval_rna=data_bundle.source.eval.rna,
        source_eval_atac=data_bundle.source.eval.atac,
        source_eval_graph_tf=source_eval_graph_tf,
        source_eval_gp_tf=eval_graph_bundle.source.gp_tf,
        source_eval_x1=eval_graph_bundle.source.x1,
        source_eval_x2=eval_graph_bundle.source.x2,
        target_eval_rna=data_bundle.target.eval.rna,
        target_eval_atac=data_bundle.target.eval.atac,
        target_eval_graph_tf=eval_graph_bundle.target.graph_tf,
        target_eval_gp_tf=eval_graph_bundle.target.gp_tf,
        target_eval_x1=eval_graph_bundle.target.x1,
        target_eval_x2=eval_graph_bundle.target.x2,
        stage2_epochs=args.stage2_epochs,
        lambda_kd=args.lambda_kd,
        kd_mix_kl=args.kd_mix_kl,
        kd_mix_ot=args.kd_mix_ot,
        target_label_key=data_bundle.target.eval.label_key,
        scib_n_jobs=args.scib_n_jobs,
        vgp_anchor_mode=vgp_anchor_mode,
    )
    if stage2_trainer is None or stage2_run_id is None:
        raise RuntimeError("Stage-2 trainer/run-id was not returned despite stage2_epochs > 0.")

    set_multigate_embeddings(
        data_bundle.source.eval.rna,
        data_bundle.source.eval.atac,
        source_eval_embeddings[0],
        source_eval_embeddings[1],
        key_added="MultiGATE",
    )
    log_stage2_artifacts_for_run(
        stage2_run_id=stage2_run_id,
        stage2_trainer=stage2_trainer,
        source_rna=data_bundle.source.eval.rna,
        source_atac=data_bundle.source.eval.atac,
        target_rna=data_bundle.target.eval.rna,
        target_atac=data_bundle.target.eval.atac,
        target_graph_tf=eval_graph_bundle.target.graph_tf,
        target_gp_tf=eval_graph_bundle.target.gp_tf,
        target_x1=eval_graph_bundle.target.x1,
        target_x2=eval_graph_bundle.target.x2,
        log_mudata_umaps=args.log_mudata_umaps,
    )
    return stage2_trainer, stage2_run_id


def run_stage1_training_and_log(
    args,
    experiment_id,
    run_name,
    eval_every,
    num_epochs,
    data_bundle,
    train_graph_bundle,
    eval_graph_bundle,
    trainer_bundle,
    cache_config,
):
    source_train_rna = data_bundle.source.train.rna
    source_train_atac = data_bundle.source.train.atac
    target_train_rna = data_bundle.target.train.rna
    target_train_atac = data_bundle.target.train.atac
    source_eval_rna = data_bundle.source.eval.rna
    source_eval_atac = data_bundle.source.eval.atac
    target_eval_rna = data_bundle.target.eval.rna
    target_eval_atac = data_bundle.target.eval.atac
    source_eval_student_graph_tf = None
    if cache_config.dual_source_kd:
        source_eval_student_graph_tf = build_source_student_graph_tf(
            source_rna=source_eval_rna,
            spatial_graph_type=cache_config.student_graph_type,
        )
    source_eval_primary_graph_tf = (
        source_eval_student_graph_tf if cache_config.dual_source_kd else eval_graph_bundle.source.graph_tf
    )

    trainer = trainer_bundle.teacher
    student_trainer = trainer_bundle.student
    nonspatial_trainer = trainer_bundle.nonspatial

    if trainer is None:
        raise RuntimeError("Stage-1 teacher trainer is required for non-cached training.")
    if trainer_bundle.source_inputs_tensors is None:
        raise RuntimeError("source_inputs_tensors is required for non-cached stage-1 training.")
    if cache_config.dual_source_kd and trainer_bundle.source_student_inputs_tensors is None:
        raise RuntimeError("source_student_inputs_tensors is required for dual-source stage-1 training.")

    source_a_t, source_prune_t, source_gp_t, source_x1_t, source_x2_t = trainer_bundle.source_inputs_tensors
    source_student_a_t = source_student_prune_t = source_student_gp_t = source_student_x1_t = source_student_x2_t = None
    if trainer_bundle.source_student_inputs_tensors is not None:
        source_student_a_t, source_student_prune_t, source_student_gp_t, source_student_x1_t, source_student_x2_t = (
            trainer_bundle.source_student_inputs_tensors
        )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("mlflow_experiment_id", experiment_id)
        mlflow.log_param("hidden_dims", json.dumps([512, 30]))
        mlflow.log_param("n_epochs", num_epochs)
        mlflow.log_param("stage2_epochs", args.stage2_epochs)
        mlflow.log_param("lambda_kd", args.lambda_kd)
        mlflow.log_param("kd_mix_kl", args.kd_mix_kl)
        mlflow.log_param("kd_mix_ot", args.kd_mix_ot)
        mlflow.log_param("bp_width", train_graph_bundle.bp_width)
        mlflow.log_param("split_seed", args.split_seed)
        mlflow.log_param("split_ratio_train", SPLIT_RATIO_TRAIN)
        mlflow.log_param("split_ratio_val", SPLIT_RATIO_VAL)
        mlflow.log_param("split_ratio_test", SPLIT_RATIO_TEST)
        mlflow.log_param("evaluation_split", "val_plus_test")
        mlflow.log_param("log_mudata_umaps", args.log_mudata_umaps)
        mlflow.log_param("source_label_key_requested", args.source_label_key if args.source_label_key is not None else "None")
        mlflow.log_param("target_label_key_requested", args.target_label_key if args.target_label_key is not None else "None")
        mlflow.log_param("source_label_key", data_bundle.source.train.label_key if data_bundle.source.train.label_key is not None else "None")
        mlflow.log_param("target_label_key", data_bundle.target.train.label_key if data_bundle.target.train.label_key is not None else "None")
        mlflow.log_param("eval_every", eval_every)
        mlflow.log_param("source_cells_train", int(source_train_rna.n_obs))
        mlflow.log_param("source_cells_eval", int(source_eval_rna.n_obs))
        mlflow.log_param("target_cells_train", int(target_train_rna.n_obs))
        mlflow.log_param("target_cells_eval", int(target_eval_rna.n_obs))
        mlflow.log_param("n_genes", int(source_train_rna.n_vars))
        mlflow.log_param("n_peaks", int(source_train_atac.n_vars))
        mlflow.log_param("graph_type", train_graph_bundle.graph_type)
        mlflow.log_param("source_split_train_n", int(data_bundle.split_metadata["domains"]["source"]["splits"]["train"]["n_obs"]))
        mlflow.log_param("source_split_val_n", int(data_bundle.split_metadata["domains"]["source"]["splits"]["val"]["n_obs"]))
        mlflow.log_param("source_split_test_n", int(data_bundle.split_metadata["domains"]["source"]["splits"]["test"]["n_obs"]))
        mlflow.log_param("source_split_eval_n", int(data_bundle.split_metadata["domains"]["source"]["splits"]["eval"]["n_obs"]))
        mlflow.log_param("target_split_train_n", int(data_bundle.split_metadata["domains"]["target"]["splits"]["train"]["n_obs"]))
        mlflow.log_param("target_split_val_n", int(data_bundle.split_metadata["domains"]["target"]["splits"]["val"]["n_obs"]))
        mlflow.log_param("target_split_test_n", int(data_bundle.split_metadata["domains"]["target"]["splits"]["test"]["n_obs"]))
        mlflow.log_param("target_split_eval_n", int(data_bundle.split_metadata["domains"]["target"]["splits"]["eval"]["n_obs"]))
        mlflow.log_param("stage1_dual_source_kd", bool(cache_config.dual_source_kd))
        mlflow.log_param("stage1_primary_model", trainer_bundle.primary_model_name)
        mlflow.log_param("stage1_teacher_graph", "spatial")
        mlflow.log_param("stage1_student_graph", cache_config.student_graph_type if cache_config.dual_source_kd else "NA")
        mlflow.log_param("stage1_nonspatial_enabled", bool(nonspatial_trainer is not None))
        mlflow.log_param(
            "stage1_nonspatial_graph",
            cache_config.student_graph_type if nonspatial_trainer is not None else "NA",
        )
        mlflow.log_param("vgp_anchor_mode", cache_config.vgp_anchor_mode)
        mlflow.log_param("skip_gp_attention", trainer.mgate.skip_gp_attention)
        log_split_metadata_artifact(data_bundle.split_metadata)

        source_scib_label_mode_logged = False
        target_scib_label_mode_logged = False
        source_scib_effective_label_key_logged = False
        target_scib_effective_label_key_logged = False
        source_embeddings = None

        for epoch in tqdm(range(1, num_epochs + 1), desc="Stage 1 training", unit="epoch"):
            trainer.mgate.train()
            teacher_loss = trainer.run_epoch(epoch, source_a_t, source_prune_t, source_gp_t, source_x1_t, source_x2_t)
            mlflow.log_metric("source_train_loss", float(teacher_loss), step=epoch)

            mlflow.log_metric("source_train_loss_atac", float(trainer.loss_list_atac[-1]), step=epoch)
            mlflow.log_metric("source_train_loss_rna", float(trainer.loss_list_rna[-1]), step=epoch)
            mlflow.log_metric("source_train_loss_clip", float(trainer.loss_list_clip[-1]), step=epoch)
            mlflow.log_metric("source_train_loss_decorr", float(trainer.loss_list_deco[-1]), step=epoch)

            if cache_config.dual_source_kd:
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
                stage1_distill_loss = F.mse_loss(student_clip_rna, teacher_clip_rna) + F.mse_loss(student_clip_atac, teacher_clip_atac)

                stage1_distill_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_trainer.mgate.parameters(), student_trainer.gradient_clipping)
                student_trainer.optimizer.step()

                mlflow.log_metric(
                    "stage1_student_distill_loss",
                    float(stage1_distill_loss.detach().cpu().item()),
                    step=epoch,
                )
                del teacher_outputs, student_outputs
                del teacher_clip_rna, teacher_clip_atac, student_clip_rna, student_clip_atac
                del stage1_distill_loss

            if nonspatial_trainer is not None:
                nonspatial_trainer.mgate.train()
                nonspatial_loss = nonspatial_trainer.run_epoch(
                    epoch,
                    source_student_a_t,
                    source_student_prune_t,
                    source_student_gp_t,
                    source_student_x1_t,
                    source_student_x2_t,
                )
                mlflow.log_metric("stage1_nonspatial_train_loss", float(nonspatial_loss), step=epoch)
                mlflow.log_metric("stage1_nonspatial_train_loss_atac", float(nonspatial_trainer.loss_list_atac[-1]), step=epoch)
                mlflow.log_metric("stage1_nonspatial_train_loss_rna", float(nonspatial_trainer.loss_list_rna[-1]), step=epoch)
                mlflow.log_metric("stage1_nonspatial_train_loss_clip", float(nonspatial_trainer.loss_list_clip[-1]), step=epoch)
                mlflow.log_metric("stage1_nonspatial_train_loss_decorr", float(nonspatial_trainer.loss_list_deco[-1]), step=epoch)

            should_eval = ((epoch == 1) or (epoch % eval_every == 0) or (epoch == num_epochs)) and eval_every > 0
            if not should_eval:
                continue

            source_embeddings, target_embeddings, trainer_target = infer_source_and_zero_shot_target_embeddings(
                source_trainer=trainer_bundle.primary,
                source_graph_tf=source_eval_primary_graph_tf,
                source_gp_tf=eval_graph_bundle.source.gp_tf,
                source_x1=eval_graph_bundle.source.x1,
                source_x2=eval_graph_bundle.source.x2,
                target_graph_tf=eval_graph_bundle.target.graph_tf,
                target_gp_tf=eval_graph_bundle.target.gp_tf,
                target_x1=eval_graph_bundle.target.x1,
                target_x2=eval_graph_bundle.target.x2,
                target_spot_num=target_eval_rna.n_obs,
                vgp_anchor_mode=cache_config.vgp_anchor_mode,
            )
            set_multigate_embeddings(
                source_eval_rna,
                source_eval_atac,
                source_embeddings[0],
                source_embeddings[1],
                key_added="MultiGATE",
            )
            set_multigate_embeddings(
                target_eval_rna,
                target_eval_atac,
                target_embeddings[0],
                target_embeddings[1],
                key_added="MultiGATE",
            )

            source_scib_metrics = compute_scib_metrics_for_domain(
                rna_adata=source_eval_rna,
                atac_adata=source_eval_atac,
                domain_name="source",
                label_key=data_bundle.source.eval.label_key,
                scib_n_jobs=args.scib_n_jobs,
            )
            log_scib_metrics(prefix="source", metrics=source_scib_metrics, step=epoch)
            if not source_scib_label_mode_logged:
                mlflow.log_param("source_scib_label_mode", source_scib_metrics["label_mode"])
                source_scib_label_mode_logged = True
            if not source_scib_effective_label_key_logged:
                mlflow.log_param("source_scib_effective_label_key", source_scib_metrics["effective_label_key"])
                source_scib_effective_label_key_logged = True

            target_scib_metrics = compute_scib_metrics_for_domain(
                rna_adata=target_eval_rna,
                atac_adata=target_eval_atac,
                domain_name="target",
                label_key=data_bundle.target.eval.label_key,
                scib_n_jobs=args.scib_n_jobs,
            )
            log_scib_metrics(prefix="target", metrics=target_scib_metrics, step=epoch)
            if not target_scib_label_mode_logged:
                mlflow.log_param("target_scib_label_mode", target_scib_metrics["label_mode"])
                target_scib_label_mode_logged = True
            if not target_scib_effective_label_key_logged:
                mlflow.log_param("target_scib_effective_label_key", target_scib_metrics["effective_label_key"])
                target_scib_effective_label_key_logged = True

            mmd_value = compute_balanced_source_target_mmd(
                source_eval_rna.obsm["MultiGATE"],
                source_eval_atac.obsm["MultiGATE"],
                target_eval_rna.obsm["MultiGATE"],
                target_eval_atac.obsm["MultiGATE"],
            )
            mlflow.log_metric("stage1_source_target_balanced_mmd", mmd_value, step=epoch)

            if nonspatial_trainer is not None:
                nonspatial_source_embeddings, nonspatial_target_embeddings, nonspatial_target_trainer = (
                    infer_source_and_zero_shot_target_embeddings(
                        source_trainer=nonspatial_trainer,
                        source_graph_tf=source_eval_student_graph_tf,
                        source_gp_tf=eval_graph_bundle.source.gp_tf,
                        source_x1=eval_graph_bundle.source.x1,
                        source_x2=eval_graph_bundle.source.x2,
                        target_graph_tf=eval_graph_bundle.target.graph_tf,
                        target_gp_tf=eval_graph_bundle.target.gp_tf,
                        target_x1=eval_graph_bundle.target.x1,
                        target_x2=eval_graph_bundle.target.x2,
                        target_spot_num=target_eval_rna.n_obs,
                        vgp_anchor_mode=cache_config.vgp_anchor_mode,
                    )
                )
                set_multigate_embeddings(
                    source_eval_rna,
                    source_eval_atac,
                    nonspatial_source_embeddings[0],
                    nonspatial_source_embeddings[1],
                    key_added="MultiGATE_nonspatial",
                )
                set_multigate_embeddings(
                    target_eval_rna,
                    target_eval_atac,
                    nonspatial_target_embeddings[0],
                    nonspatial_target_embeddings[1],
                    key_added="MultiGATE_nonspatial",
                )
                nonspatial_mmd_value = compute_balanced_source_target_mmd(
                    source_eval_rna.obsm["MultiGATE_nonspatial"],
                    source_eval_atac.obsm["MultiGATE_nonspatial"],
                    target_eval_rna.obsm["MultiGATE_nonspatial"],
                    target_eval_atac.obsm["MultiGATE_nonspatial"],
                )
                mlflow.log_metric("stage1_nonspatial_source_target_balanced_mmd", nonspatial_mmd_value, step=epoch)
                if epoch != num_epochs:
                    del nonspatial_target_trainer, nonspatial_target_embeddings, nonspatial_source_embeddings

            if epoch != num_epochs:
                del trainer_target, target_embeddings, source_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if source_embeddings is None:
            raise RuntimeError("Stage-1 source embeddings were not computed before artifact logging.")

        log_stage_umap_artifacts(
            source_rna=source_eval_rna,
            source_atac=source_eval_atac,
            target_rna=target_eval_rna,
            target_atac=target_eval_atac,
            stage_label="stage1",
            log_mudata_umaps=args.log_mudata_umaps,
        )

        if cache_config.dual_source_kd:
            teacher_source_embeddings, teacher_target_embeddings, teacher_target_trainer = (
                infer_source_and_zero_shot_target_embeddings(
                    source_trainer=trainer,
                    source_graph_tf=eval_graph_bundle.source.graph_tf,
                    source_gp_tf=eval_graph_bundle.source.gp_tf,
                    source_x1=eval_graph_bundle.source.x1,
                    source_x2=eval_graph_bundle.source.x2,
                    target_graph_tf=eval_graph_bundle.target.graph_tf,
                    target_gp_tf=eval_graph_bundle.target.gp_tf,
                    target_x1=eval_graph_bundle.target.x1,
                    target_x2=eval_graph_bundle.target.x2,
                    target_spot_num=target_eval_rna.n_obs,
                    vgp_anchor_mode=cache_config.vgp_anchor_mode,
                )
            )
            set_multigate_embeddings(
                source_eval_rna,
                source_eval_atac,
                teacher_source_embeddings[0],
                teacher_source_embeddings[1],
                key_added="MultiGATE_teacher",
            )
            set_multigate_embeddings(
                target_eval_rna,
                target_eval_atac,
                teacher_target_embeddings[0],
                teacher_target_embeddings[1],
                key_added="MultiGATE_teacher",
            )
            log_stage_umap_artifacts(
                source_rna=source_eval_rna,
                source_atac=source_eval_atac,
                target_rna=target_eval_rna,
                target_atac=target_eval_atac,
                stage_label="stage1_teacher",
                log_mudata_umaps=args.log_mudata_umaps,
                embedding_key="MultiGATE_teacher",
            )
            del teacher_target_trainer

        nonspatial_source_embeddings_final = None
        if nonspatial_trainer is not None:
            nonspatial_source_embeddings_final, nonspatial_target_embeddings_final, nonspatial_target_trainer_final = (
                infer_source_and_zero_shot_target_embeddings(
                    source_trainer=nonspatial_trainer,
                    source_graph_tf=source_eval_student_graph_tf,
                    source_gp_tf=eval_graph_bundle.source.gp_tf,
                    source_x1=eval_graph_bundle.source.x1,
                    source_x2=eval_graph_bundle.source.x2,
                    target_graph_tf=eval_graph_bundle.target.graph_tf,
                    target_gp_tf=eval_graph_bundle.target.gp_tf,
                    target_x1=eval_graph_bundle.target.x1,
                    target_x2=eval_graph_bundle.target.x2,
                    target_spot_num=target_eval_rna.n_obs,
                    vgp_anchor_mode=cache_config.vgp_anchor_mode,
                )
            )
            set_multigate_embeddings(
                source_eval_rna,
                source_eval_atac,
                nonspatial_source_embeddings_final[0],
                nonspatial_source_embeddings_final[1],
                key_added="MultiGATE_nonspatial",
            )
            set_multigate_embeddings(
                target_eval_rna,
                target_eval_atac,
                nonspatial_target_embeddings_final[0],
                nonspatial_target_embeddings_final[1],
                key_added="MultiGATE_nonspatial",
            )
            log_stage_umap_artifacts(
                source_rna=source_eval_rna,
                source_atac=source_eval_atac,
                target_rna=target_eval_rna,
                target_atac=target_eval_atac,
                stage_label="stage1_nonspatial",
                log_mudata_umaps=args.log_mudata_umaps,
                embedding_key="MultiGATE_nonspatial",
            )
            del nonspatial_target_trainer_final

        with tempfile.TemporaryDirectory() as tmpdir:
            log_torch_state_dict_artifacts(
                tmpdir,
                {
                    "model_stage1.pth": trainer_bundle.primary.mgate.state_dict(),
                    "model_stage1_teacher.pth": trainer.mgate.state_dict() if cache_config.dual_source_kd else None,
                    "model_stage1_student.pth": student_trainer.mgate.state_dict() if cache_config.dual_source_kd else None,
                    "model_stage1_nonspatial.pth": nonspatial_trainer.mgate.state_dict() if nonspatial_trainer is not None else None,
                },
            )

            sparse_matrices = {}
            if not trainer_bundle.primary.mgate.skip_gp_attention:
                sparse_matrices["source_peak_gene_attention.npz"] = source_embeddings[4][0]
            if (
                nonspatial_trainer is not None
                and not nonspatial_trainer.mgate.skip_gp_attention
                and nonspatial_source_embeddings_final is not None
            ):
                sparse_matrices["source_peak_gene_attention_nonspatial.npz"] = nonspatial_source_embeddings_final[4][0]
            log_sparse_matrix_artifacts(tmpdir, sparse_matrices)

        maybe_run_stage2_and_log(
            args=args,
            data_bundle=data_bundle,
            train_graph_bundle=train_graph_bundle,
            eval_graph_bundle=eval_graph_bundle,
            source_eval_graph_tf=source_eval_primary_graph_tf,
            trainer_bundle=trainer_bundle,
            source_eval_embeddings=source_embeddings,
            vgp_anchor_mode=cache_config.vgp_anchor_mode,
        )


def summarize_stage1_setup(num_epochs, data_bundle, cache_config, trainer_bundle):
    print("Training epochs for stage 1:", num_epochs)
    source_split_mode = "train/eval splits" if data_bundle.source.train.rna.n_obs != data_bundle.source.full.rna.n_obs else "full source train+eval"
    print(
        "Source mode: {} (train/val/test/eval/full = {}/{}/{}/{}/{})".format(
            source_split_mode,
            data_bundle.source.train.rna.n_obs,
            data_bundle.source.val.rna.n_obs,
            data_bundle.source.test.rna.n_obs,
            data_bundle.source.eval.rna.n_obs,
            data_bundle.source.full.rna.n_obs,
        )
    )
    print(
        "Target split sizes (train/val/test/eval): {}/{}/{}/{}".format(
            data_bundle.target.train.rna.n_obs,
            data_bundle.target.val.rna.n_obs,
            data_bundle.target.test.rna.n_obs,
            data_bundle.target.eval.rna.n_obs,
        )
    )
    if cache_config.dual_source_kd:
        print(
            "[Stage1 Dual KD] Enabled: teacher graph=spatial, student graph={}".format(
                cache_config.student_graph_type
            )
        )
        if trainer_bundle.nonspatial is not None:
            print(
                "[Stage1 Non-Spatial] Enabled: auxiliary non-spatial model trains on the student graph "
                "with teacher-style losses."
            )


def main():
    bootstrap_runtime()

    notebook_mode = is_notebook()
    args = parse_args(notebook=notebook_mode)
    validate_args(args)

    # Fail fast if scib-metrics backend is not available.
    require_scib_backend()

    experiment_id = setup_mlflow()
    eval_every = 3000  # set to -1 for very basic debugging only.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_epochs = args.stage1_epochs

    mlflow_client = MlflowClient()
    cache_config = resolve_stage1_cache_config(args, mlflow_client)
    data_bundle = load_and_prepare_data_bundle(args)
    train_graph_bundle = build_graph_bundle(data_bundle)
    eval_graph_bundle = build_graph_bundle_from_domains(
        source_domain=data_bundle.source.eval,
        target_domain=data_bundle.target.eval,
        bp_width=train_graph_bundle.bp_width,
        graph_type=train_graph_bundle.graph_type,
        protein_value=train_graph_bundle.protein_value,
    )

    if cache_config.use_cache:
        trainer_bundle = load_cached_stage1_primary_trainer(
            data_bundle=data_bundle,
            graph_bundle=train_graph_bundle,
            cache_config=cache_config,
            mlflow_client=mlflow_client,
            num_epochs=num_epochs,
        )
        summarize_stage1_setup(num_epochs, data_bundle, cache_config, trainer_bundle)
        print(
            "[Stage1 Cache] Reusing parent run '{}' (ID: {}). Stage-1 training will be skipped.".format(
                cache_config.run_name,
                cache_config.run_id,
            )
        )
        with mlflow.start_run(run_id=cache_config.run_id):
            source_eval_student_graph_tf = None
            if cache_config.dual_source_kd:
                source_eval_student_graph_tf = build_source_student_graph_tf(
                    source_rna=data_bundle.source.eval.rna,
                    spatial_graph_type=cache_config.student_graph_type,
                )
            source_eval_primary_graph_tf = (
                source_eval_student_graph_tf if cache_config.dual_source_kd else eval_graph_bundle.source.graph_tf
            )
            source_embeddings, source_infer_trainer = infer_source_embeddings(
                source_trainer=trainer_bundle.primary,
                source_graph_tf=source_eval_primary_graph_tf,
                source_gp_tf=eval_graph_bundle.source.gp_tf,
                source_x1=eval_graph_bundle.source.x1,
                source_x2=eval_graph_bundle.source.x2,
                vgp_anchor_mode=cache_config.vgp_anchor_mode,
            )
            del source_infer_trainer
            maybe_run_stage2_and_log(
                args=args,
                data_bundle=data_bundle,
                train_graph_bundle=train_graph_bundle,
                eval_graph_bundle=eval_graph_bundle,
                source_eval_graph_tf=source_eval_primary_graph_tf,
                trainer_bundle=trainer_bundle,
                source_eval_embeddings=source_embeddings,
                vgp_anchor_mode=cache_config.vgp_anchor_mode,
            )
        return

    trainer_bundle = initialize_stage1_trainers_for_training(
        data_bundle=data_bundle,
        graph_bundle=train_graph_bundle,
        cache_config=cache_config,
        num_epochs=num_epochs,
    )
    summarize_stage1_setup(num_epochs, data_bundle, cache_config, trainer_bundle)
    run_stage1_training_and_log(
        args=args,
        experiment_id=experiment_id,
        run_name=run_name,
        eval_every=eval_every,
        num_epochs=num_epochs,
        data_bundle=data_bundle,
        train_graph_bundle=train_graph_bundle,
        eval_graph_bundle=eval_graph_bundle,
        trainer_bundle=trainer_bundle,
        cache_config=cache_config,
    )


if __name__ == "__main__":
    main()

# %%
