try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#!/usr/bin/env python
#%%
import argparse
import os
import sys
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, Optional

# Python 3.7 compatibility for muon/mudata (they use typing.Literal in newer versions)
if sys.version_info < (3, 8):
    import typing
    from typing_extensions import Literal

    typing.Literal = Literal

import scvi

from dotenv import dotenv_values, load_dotenv
ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
DEFAULT_SOURCE_LABEL_KEY = "RNA_clusters"
DEFAULT_TARGET_LABEL_KEY = "arc_gex_kmeans_5_clusters_Cluster"
SPLIT_RATIO_TRAIN = 0.5
SPLIT_RATIO_VAL = 0.3
SPLIT_RATIO_TEST = 0.2
SPLIT_SCHEMA_VERSION = 1

BASE_PATH = None
REPO_ROOT = None
MultiGATE = None


def bootstrap_runtime():
    global BASE_PATH, REPO_ROOT, MultiGATE

    load_dotenv(dotenv_path=ENV_FILE_PATH)
    print("Loaded environment variables from .env or env:", end="\n\n")
    pprint(dotenv_values(ENV_FILE_PATH))

    datapath = os.getenv("SPATIALJEPA_DATA")
    if datapath is None:
        raise EnvironmentError(
            "SPATIALJEPA_DATA is not set. Export SPATIALJEPA_DATA to the base data directory, e.g. "
            "'$SPATIALJEPA_HOME/data'."
        )

    baklava_base_dir = os.getenv("SPATIALJEPA_HOME")
    if baklava_base_dir is None:
        raise EnvironmentError(
            "SPATIALJEPA_HOME is not set. Export SPATIALJEPA_HOME to the base repo directory, e.g. "
            "'$SPATIALJEPA_HOME'."
        )

    REPO_ROOT = os.environ["SPATIALJEPA_REPO"]
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    baklava_repo_root = os.environ["SPATIALJEPA_SHARED"]
    if os.path.isdir(baklava_repo_root) and baklava_repo_root not in sys.path:
        sys.path.insert(0, baklava_repo_root)

    BASE_PATH = os.path.join(datapath, "aligned_data")

    import MultiGATE
    import scvi


import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

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
        "--top-n-genes",
        type=int,
        default=0,
        help="Number of top genes to keep for filtering. If 0, use SCT genes.",
    )
    parser.add_argument(
        "--top-n-peaks",
        type=int,
        default=0,
        help="Number of top in-cis peaks to keep for filtering. If 0, use peaks linked to SCT genes.",
    )
    parser.add_argument(
        "--source-label-key",
        type=str,
        default=None,
        help="Optional source label key in .obs. Falls back to default key if missing.",
    )
    parser.add_argument(
        "--target-label-key",
        type=str,
        default=None,
        help="Optional target label key in .obs. Falls back to default key if missing.",
    )
    parser.add_argument(
        "--spatial-graph-type",
        type=str,
        choices=["spatial", "knn", "identity", "tangram"],
        default="identity",
        help="Type of graph to use for MultiGATE.",
    )
    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


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

def get_sct_genes_and_gp_filtering(
    source_rna,
    source_atac,
    target_rna,
    target_atac,
    gp_net
    ):

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

    where_overlap_sct_genes = source_rna.var['SCT_gene'] & target_rna.var['SCT_gene']
    overlap_sct_genes = where_overlap_sct_genes.index[where_overlap_sct_genes]

    source_rna.var.loc[overlap_sct_genes, "highly_variable"] = True
    target_rna.var.loc[overlap_sct_genes, "highly_variable"] = True

    peak_filt = gp_net.loc[
        gp_net["Gene"].isin(overlap_sct_genes),
        "Peak",
    ].unique()
    source_atac.var.loc[source_atac.var_names.isin(peak_filt), "highly_variable"] = True
    target_atac.var.loc[target_atac.var_names.isin(peak_filt), "highly_variable"] = True

    gp_net = gp_net[
        gp_net["Gene"].isin(overlap_sct_genes)
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
        tangram_net = pd.read_csv(os.path.join(os.getenv("SPATIALJEPA_OUT"), "tangram", "tangram_spatial_net_affinity.csv"))
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
        "No usable {} label key found (requested='{}', default='{}').".format(
            domain_name,
            requested_key,
            default_key,
        )
    )
    return None


def load_and_prepare_data_bundle(args):

    source_rna = sc.read_h5ad(os.path.join(BASE_PATH, "source_rna_aligned_SCT.h5ad"))
    source_atac = sc.read_h5ad(os.path.join(BASE_PATH, "source_atac_aligned.h5ad"))
    source_rna.obsm["spatial"] = source_rna.obsm["spatial"] * -1
    source_atac.obsm["spatial"] = source_atac.obsm["spatial"] * -1

    target_rna = sc.read_h5ad(os.path.join(BASE_PATH, "target_rna_aligned_SCT.h5ad"))
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
        os.getenv("SPATIALJEPA_DATA"),
        "gene_annotations",
        "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    )
    if not os.path.exists(gtf_path):
        raise FileNotFoundError("GTF annotation file not found: {}".format(gtf_path))

    MultiGATE.Cal_gene_peak_Net_new(source_rna, source_atac, 150000, file=gtf_path)
    gp_net = source_atac.uns["gene_peak_Net"].copy()
    del source_atac.uns["gene_peak_Net"]

    use_sct_genes = (args.top_n_genes==0 and args.top_n_peaks==0)

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
        combined_gp_dict=None,
    )

    data_bundle.source = configure_source_train_eval_bundle(
        data_bundle.source,
        use_source_split_train_eval=bool(args.source_split_train_eval),
    )
    data_bundle.split_metadata = build_split_metadata(data_bundle.source, data_bundle.target)

    return data_bundle


#%%
def main():

    # setup runtime
    bootstrap_runtime()
    notebook_mode = is_notebook()
    args = parse_args(notebook=notebook_mode)
    validate_args(args)

    # load and prepare data
    data_bundle = load_and_prepare_data_bundle(args)
    mdata = mu.MuData({
        "rna": data_bundle.source.train.rna,
        "atac": data_bundle.source.train.atac
    })
    target_mdata = mu.MuData({
        "rna": data_bundle.target.train.rna,
        "atac": data_bundle.target.train.atac
    })

    # setup mudata
    scvi.model.MULTIVI.setup_mudata(
        mdata,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        },
    )
    scvi.model.MULTIVI.setup_mudata(
        target_mdata,
        modalities={
            "rna_layer": "rna",
            "atac_layer": "atac",
        },
    )

    # setup model
    model = scvi.model.MULTIVI(
        mdata,
        n_genes=len(mdata.mod["rna"].var),
        n_regions=len(mdata.mod["atac"].var),
    )
    target_model = scvi.model.MULTIVI(
        target_mdata,
        n_genes=len(target_mdata.mod["rna"].var),
        n_regions=len(target_mdata.mod["atac"].var),
    )

    model.view_anndata_setup()
    target_model.view_anndata_setup()

    # set csr counts data
    mdata.mod["rna"].X = mdata.mod["rna"].layers['counts'].tocsr()
    mdata.mod["atac"].X = mdata.mod["atac"].layers['counts'].tocsr()
    target_mdata.mod["rna"].X = target_mdata.mod["rna"].layers['counts'].tocsr()
    target_mdata.mod["atac"].X = target_mdata.mod["atac"].layers['counts'].tocsr()
    mdata.update()
    target_mdata.update()

    # train model
    num_epochs = args.stage1_epochs
    model.train(max_epochs=num_epochs)
    target_model.train(max_epochs=num_epochs)

    # save model
    model_dir = os.path.join(os.getenv("SPATIALJEPA_OUT"), "multivi_mouse_brain_spatial_rna_atac")
    target_model_dir = os.path.join(os.getenv("SPATIALJEPA_OUT"), "multivi_mouse_brain_spatial_rna_atac_target")
    model.save(model_dir, overwrite=True)
    target_model.save(target_model_dir, overwrite=True)
    print(f"Model saved to {model_dir}")
    print(f"Target model saved to {target_model_dir}")

    #model = scvi.model.MULTIVI.load(model_dir, adata=mdata)
    #target_model = scvi.model.MULTIVI.load(target_model_dir, adata=target_mdata)
    
    # extracting and visualizing the latent space
    MULTIVI_LATENT_KEY = "X_multivi"
    MULTIVI_RNA_LATENT_KEY = "X_multivi_rna"
    MULTIVI_ATAC_LATENT_KEY = "X_multivi_atac"

    # joint embedding (used for UMAP/clustering)
    mdata.obsm[MULTIVI_LATENT_KEY] = model.get_latent_representation(modality="joint")
    target_mdata.obsm[MULTIVI_LATENT_KEY] = target_model.get_latent_representation(modality="joint")
    target_mdata.obsm[MULTIVI_LATENT_KEY+'_zero_shot'] = model.get_latent_representation(adata=target_mdata, modality="joint")

    # modality-specific embeddings stored per-modality
    mdata.mod["rna"].obsm[MULTIVI_RNA_LATENT_KEY] = model.get_latent_representation(modality="expression")
    mdata.mod["atac"].obsm[MULTIVI_ATAC_LATENT_KEY] = model.get_latent_representation(modality="accessibility")
    target_mdata.mod["rna"].obsm[MULTIVI_RNA_LATENT_KEY] = target_model.get_latent_representation(modality="expression")
    target_mdata.mod["atac"].obsm[MULTIVI_ATAC_LATENT_KEY] = target_model.get_latent_representation(modality="accessibility")
    target_mdata.mod["rna"].obsm[MULTIVI_RNA_LATENT_KEY+'_zero_shot'] = model.get_latent_representation(adata=target_mdata, modality="expression")
    target_mdata.mod["atac"].obsm[MULTIVI_ATAC_LATENT_KEY+'_zero_shot'] = model.get_latent_representation(adata=target_mdata, modality="accessibility")

    def compute_umap_and_leiden(
        mdata,
        rna_ct_key="RNA_clusters",
        atac_ct_key="ATAC_clusters",
        mudata_fn=None,
        zero_shot=False
        ):

        latent_key = MULTIVI_LATENT_KEY+'_zero_shot' if zero_shot else MULTIVI_LATENT_KEY
        rna_latent_key = MULTIVI_RNA_LATENT_KEY+'_zero_shot' if zero_shot else MULTIVI_RNA_LATENT_KEY
        atac_latent_key = MULTIVI_ATAC_LATENT_KEY+'_zero_shot' if zero_shot else MULTIVI_ATAC_LATENT_KEY

        # compute UMAP and Leiden clustering
        sc.pp.neighbors(mdata, use_rep=latent_key)
        sc.tl.umap(mdata, min_dist=0.2)
        sc.tl.leiden(mdata, resolution=0.25)

        sc.pp.neighbors(mdata.mod['rna'], use_rep=rna_latent_key)
        sc.tl.umap(mdata.mod['rna'], min_dist=0.2)
        sc.tl.leiden(mdata.mod['rna'], resolution=0.2)

        sc.pp.neighbors(mdata.mod['atac'], use_rep=atac_latent_key)
        sc.tl.umap(mdata.mod['atac'], min_dist=0.2)
        sc.tl.leiden(mdata.mod['atac'], resolution=0.2)

        # assign clusters to the mudata
        mdata.mod['rna'].obs[rna_ct_key] = 'c_' + mdata.mod['rna'].obs[rna_ct_key].astype(str)
        mdata.mod['atac'].obs[atac_ct_key] = 'c_' + mdata.mod['atac'].obs[atac_ct_key].astype(str)
        mdata.obs[rna_ct_key] = mdata.mod['rna'].obs[rna_ct_key]
        mdata.obs[atac_ct_key] = mdata.mod['atac'].obs[atac_ct_key]

        # visualize the clusters
        sc.pl.umap(mdata, color=[rna_ct_key, atac_ct_key, "leiden"])
        sc.pl.umap(mdata.mod['rna'], color=[rna_ct_key, "leiden"])
        sc.pl.umap(mdata.mod['atac'], color=[atac_ct_key, "leiden"])

        if ('spatial' in mdata.mod['rna'].obsm) and ('spatial' in mdata.mod['atac'].obsm) or ('spatial' in mdata.obsm):
            assert np.all(mdata.mod['rna'].obsm['spatial'] == mdata.mod['atac'].obsm['spatial']), "Spatial coordinates must have the same shape"
            mdata.obsm['spatial'] = mdata.mod['rna'].obsm['spatial']
            sc.pl.embedding(mdata, color=[rna_ct_key, atac_ct_key, "leiden"], basis="spatial", s=60)
            sc.pl.embedding(mdata.mod['rna'], color=[rna_ct_key, "leiden"], basis="spatial", s=60)
            sc.pl.embedding(mdata.mod['atac'], color=[atac_ct_key, "leiden"], basis="spatial", s=60)

        if mudata_fn is not None:
            # save mudata to disk
            mudata_path = os.path.join(BASE_PATH, mudata_fn)
            mdata.write_h5mu(mudata_path)
            print(f"Mudata saved to {mudata_path}")

    compute_umap_and_leiden(mdata, mudata_fn='multivi_mdata.h5mu')
    compute_umap_and_leiden(target_mdata, rna_ct_key="arc_gex_graphclust_Cluster", atac_ct_key="arc_atac_graphclust_Cluster", mudata_fn='multivi_mdata_target.h5mu')
    compute_umap_and_leiden(target_mdata, rna_ct_key="arc_gex_graphclust_Cluster", atac_ct_key="arc_atac_graphclust_Cluster", mudata_fn='multivi_mdata_zero_shot.h5mu', zero_shot=True)

if __name__ == "__main__":
    main()

# %%
