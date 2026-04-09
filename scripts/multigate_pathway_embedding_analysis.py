#!/usr/bin/env python
"""
Post-hoc interpretation: relate NicheCompass-style combined_gp_dict pathways to MultiGATE embeddings.

Designed for small, explicit extension points (PathwayEmbeddingConfig and optional gene extractors).

Typical usage (after embeddings exist on RNA AnnData):

    from multigate_pathway_embedding_analysis import (
        PathwayEmbeddingConfig,
        run_pathway_embedding_analysis,
        save_pathway_embedding_results,
    )

    cfg = PathwayEmbeddingConfig(embedding_key="MultiGATE", z_score_pathways=True)
    result = run_pathway_embedding_analysis(source_rna, combined_gp_dict, cfg)
    save_pathway_embedding_results(result, "pathway_emb_out")

CLI:

    python multigate_pathway_embedding_analysis.py --adata source_rna_co_embed.h5ad \\
        --embedding-key MultiGATE --out-dir ./pathway_analysis --load-nichecompass-gp-dict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration (extend here when you want new behavior)
# ---------------------------------------------------------------------------


GenePoolMode = Literal["union", "targets", "sources"]

PathwayGeneExtractor = Callable[[str, Any, "PathwayEmbeddingConfig"], List[str]]


def default_extract_pathway_genes(gp_name: str, gp_value: Any, cfg: "PathwayEmbeddingConfig") -> List[str]:
    """
    Map one combined_gp_dict entry to a list of gene symbols.

    NicheCompass gp_dict values are typically dict-like with 'sources' and 'targets'
    (lists or ndarray of str). Override via PathwayEmbeddingConfig.extract_pathway_genes.
    """
    del gp_name  # available for custom extractors / logging

    if gp_value is None:
        return []

    if isinstance(gp_value, (str, bytes)):
        return []

    if isinstance(gp_value, Mapping):
        sources = gp_value.get("sources", gp_value.get("source_genes", []))
        targets = gp_value.get("targets", gp_value.get("target_genes", []))
        if isinstance(sources, np.ndarray):
            sources = sources.tolist()
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()
        sources = [str(g) for g in sources if g is not None and str(g)]
        targets = [str(g) for g in targets if g is not None and str(g)]

        if cfg.gene_pool == "targets":
            genes = targets
        elif cfg.gene_pool == "sources":
            genes = sources
        else:
            genes = list(dict.fromkeys(sources + targets))
        return genes

    if isinstance(gp_value, (list, tuple, set)):
        return [str(g) for g in gp_value if g is not None and str(g)]

    return []


@dataclass
class PathwayEmbeddingConfig:
    """Knobs for pathway scoring and correlation; add fields here as analyses grow."""

    embedding_key: str = "MultiGATE"
    gene_pool: GenePoolMode = "union"
    min_genes_per_pathway: int = 3
    apply_log1p: bool = True
    z_score_pathways: bool = True
    layer: Optional[str] = None
    correlation_method: Literal["spearman"] = "spearman"
    extract_pathway_genes: PathwayGeneExtractor = default_extract_pathway_genes
    # Mean pathway score per cluster; set to None to disable. Skipped if column missing on AnnData.
    cluster_obs_key: Optional[str] = "leiden"


@dataclass
class PathwayEmbeddingResult:
    pathway_scores: pd.DataFrame
    correlation: pd.DataFrame
    p_values: pd.DataFrame
    n_genes_per_pathway: pd.Series
    skipped_pathways: List[str]
    config: PathwayEmbeddingConfig = field(repr=False)
    pathway_mean_by_cluster: Optional[pd.DataFrame] = None
    cluster_obs_key_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def _dense_expression(adata: AnnData, layer: Optional[str]) -> np.ndarray:
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def compute_pathway_scores(
    adata: AnnData,
    combined_gp_dict: Mapping[str, Any],
    config: PathwayEmbeddingConfig,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Per spot: mean log1p expression over genes in each pathway (after intersecting var_names).
    """
    X = _dense_expression(adata, config.layer)
    if config.apply_log1p:
        X = np.log1p(X)

    var_names = np.array(adata.var_names.astype(str))
    name_to_col = {g: i for i, g in enumerate(var_names)}

    columns: Dict[str, np.ndarray] = {}
    n_genes: Dict[str, int] = {}
    skipped: List[str] = []

    for gp_name, gp_value in combined_gp_dict.items():
        genes = config.extract_pathway_genes(str(gp_name), gp_value, config)
        cols = sorted({name_to_col[g] for g in genes if g in name_to_col})
        n_genes[str(gp_name)] = len(cols)
        if len(cols) < config.min_genes_per_pathway:
            skipped.append(str(gp_name))
            continue
        mean_expr = X[:, cols].mean(axis=1)
        columns[str(gp_name)] = mean_expr

    if not columns:
        raise ValueError(
            "No pathways passed min_genes_per_pathway={}. Check gene symbols vs adata.var_names.".format(
                config.min_genes_per_pathway
            )
        )

    scores = pd.DataFrame(columns, index=adata.obs_names.copy())
    if config.z_score_pathways:
        scores = scores.apply(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12), axis=0)

    n_genes_series = pd.Series(n_genes, dtype=int)
    return scores, n_genes_series, skipped


def mean_pathway_scores_by_cluster(
    adata_rna: AnnData,
    pathway_scores: pd.DataFrame,
    obs_key: str,
) -> Optional[pd.DataFrame]:
    """
    Mean per-cell pathway scores within each ``obs[obs_key]`` group (e.g. Leiden).

    Returns a DataFrame (clusters × pathways), or None if ``obs_key`` is absent.
    """
    if obs_key not in adata_rna.obs.columns:
        return None

    scores = pathway_scores.reindex(adata_rna.obs_names)
    labels_raw = adata_rna.obs[obs_key]
    mask = labels_raw.notna()
    labels = labels_raw.astype(str)
    mask = mask & ~labels.str.lower().isin({"nan", "none", ""})
    if not mask.any():
        return None

    grouped = scores.loc[mask].groupby(labels.loc[mask], observed=True).mean()
    grouped.index.name = obs_key
    return grouped


def correlate_pathways_with_embedding(
    embedding: np.ndarray,
    pathway_scores: pd.DataFrame,
    method: Literal["spearman"] = "spearman",
    embedding_column_prefix: str = "emb",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """pathways x embedding dimensions: Spearman rho and two-sided p-values."""
    if embedding.shape[0] != pathway_scores.shape[0]:
        raise ValueError(
            "embedding rows ({}) != pathway_scores rows ({})".format(
                embedding.shape[0], pathway_scores.shape[0]
            )
        )

    if method != "spearman":
        raise ValueError("Only 'spearman' is implemented; extend correlate_pathways_with_embedding for others.")

    n_obs, n_emb = embedding.shape
    n_pw = pathway_scores.shape[1]
    rho = np.zeros((n_pw, n_emb), dtype=np.float64)
    pval = np.zeros((n_pw, n_emb), dtype=np.float64)

    pw_mat = pathway_scores.to_numpy(dtype=np.float64)
    for i in range(n_pw):
        for j in range(n_emb):
            r, p = stats.spearmanr(pw_mat[:, i], embedding[:, j], nan_policy="omit")
            rho[i, j] = 0.0 if np.isnan(r) else float(r)
            pval[i, j] = 1.0 if np.isnan(p) else float(p)

    pw_names = list(pathway_scores.columns)
    emb_cols = ["{}_{:d}".format(embedding_column_prefix, j) for j in range(n_emb)]
    return (
        pd.DataFrame(rho, index=pw_names, columns=emb_cols),
        pd.DataFrame(pval, index=pw_names, columns=emb_cols),
    )


def summarize_top_pathway_dimensions(
    correlation: pd.DataFrame,
    p_values: Optional[pd.DataFrame] = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Per pathway: strongest |rho| embedding dimension."""
    rows = []
    for pw in correlation.index:
        s = correlation.loc[pw]
        j = int(np.nanargmax(np.abs(s.values)))
        col = s.index[j]
        entry = {
            "pathway": pw,
            "best_dim": col,
            "spearman_rho": float(s.iloc[j]),
        }
        if p_values is not None and pw in p_values.index:
            entry["p_value"] = float(p_values.loc[pw, col])
        rows.append(entry)
    out = pd.DataFrame(rows).set_index("pathway")
    out["abs_rho"] = out["spearman_rho"].abs()
    out = out.sort_values("abs_rho", ascending=False).head(top_k)
    return out.drop(columns=["abs_rho"])


def run_pathway_embedding_analysis(
    adata_rna: AnnData,
    combined_gp_dict: Optional[Mapping[str, Any]],
    config: Optional[PathwayEmbeddingConfig] = None,
) -> PathwayEmbeddingResult:
    """
    Run full pipeline: pathway scores from RNA, Spearman vs obsm[embedding_key].

    adata_rna must contain config.embedding_key in .obsm with shape (n_obs, d).
    """
    if combined_gp_dict is None:
        raise ValueError("combined_gp_dict is None; load it or pass --load-nichecompass-gp-dict in CLI.")

    cfg = config or PathwayEmbeddingConfig()
    if cfg.embedding_key not in adata_rna.obsm:
        raise KeyError(
            "obsm['{}'] missing; available: {}".format(cfg.embedding_key, list(adata_rna.obsm.keys()))
        )

    emb = np.asarray(adata_rna.obsm[cfg.embedding_key], dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError("Embedding must be 2D, got shape {}".format(emb.shape))

    scores, n_genes_series, skipped = compute_pathway_scores(adata_rna, combined_gp_dict, cfg)
    scores = scores.reindex(adata_rna.obs_names)
    if scores.shape[0] != emb.shape[0]:
        raise ValueError(
            "pathway score rows {} != embedding rows {}".format(scores.shape[0], emb.shape[0])
        )

    rho, pval = correlate_pathways_with_embedding(
        emb,
        scores,
        method=cfg.correlation_method,
    )

    cluster_means: Optional[pd.DataFrame] = None
    cluster_key_used: Optional[str] = None
    if cfg.cluster_obs_key is not None:
        cluster_means = mean_pathway_scores_by_cluster(adata_rna, scores, cfg.cluster_obs_key)
        if cluster_means is not None:
            cluster_key_used = cfg.cluster_obs_key

    return PathwayEmbeddingResult(
        pathway_scores=scores,
        correlation=rho,
        p_values=pval,
        n_genes_per_pathway=n_genes_series,
        skipped_pathways=skipped,
        config=cfg,
        pathway_mean_by_cluster=cluster_means,
        cluster_obs_key_used=cluster_key_used,
    )


def save_pathway_embedding_results(result: PathwayEmbeddingResult, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "pathway_scores": os.path.join(out_dir, "pathway_scores.csv"),
        "correlation": os.path.join(out_dir, "pathway_embedding_spearman.csv"),
        "p_values": os.path.join(out_dir, "pathway_embedding_spearman_pvalues.csv"),
        "n_genes_per_pathway": os.path.join(out_dir, "pathway_n_genes_in_adata.csv"),
        "skipped_pathways": os.path.join(out_dir, "pathways_skipped.json"),
        "summary_top": os.path.join(out_dir, "pathway_embedding_top_by_abs_rho.csv"),
        "config": os.path.join(out_dir, "pathway_embedding_config.json"),
    }
    result.pathway_scores.to_csv(paths["pathway_scores"])
    result.correlation.to_csv(paths["correlation"])
    result.p_values.to_csv(paths["p_values"])
    result.n_genes_per_pathway.to_csv(paths["n_genes_per_pathway"], header=["n_genes"])
    with open(paths["skipped_pathways"], "w") as f:
        json.dump(result.skipped_pathways, f, indent=2)

    summary = summarize_top_pathway_dimensions(
        result.correlation,
        p_values=result.p_values,
        top_k=min(50, len(result.correlation)),
    )
    summary.to_csv(paths["summary_top"])

    if result.pathway_mean_by_cluster is not None:
        p_cluster = os.path.join(out_dir, "pathway_mean_by_cluster.csv")
        result.pathway_mean_by_cluster.to_csv(p_cluster)
        paths["pathway_mean_by_cluster"] = p_cluster

    cfg = result.config
    cfg_dump = {
        "embedding_key": cfg.embedding_key,
        "gene_pool": cfg.gene_pool,
        "min_genes_per_pathway": cfg.min_genes_per_pathway,
        "apply_log1p": cfg.apply_log1p,
        "z_score_pathways": cfg.z_score_pathways,
        "layer": cfg.layer,
        "correlation_method": cfg.correlation_method,
        "extract_pathway_genes": getattr(cfg.extract_pathway_genes, "__name__", "custom"),
        "cluster_obs_key": cfg.cluster_obs_key,
        "cluster_obs_key_used": result.cluster_obs_key_used,
    }
    with open(paths["config"], "w") as f:
        json.dump(cfg_dump, f, indent=2)

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pathway vs MultiGATE embedding Spearman analysis.")
    p.add_argument("--adata", required=True, help="RNA AnnData (.h5ad) with embeddings in obsm.")
    p.add_argument("--embedding-key", default="MultiGATE", help="obsm key for embedding matrix.")
    p.add_argument("--out-dir", required=True, help="Directory for CSV outputs.")
    p.add_argument(
        "--gp-dict-pickle",
        default=None,
        help="Path to a pickle file containing combined_gp_dict (mapping pathway name -> entry).",
    )
    p.add_argument(
        "--load-nichecompass-gp-dict",
        action="store_true",
        help="Load combined_gp_dict via mouse_brain_spatial_rna_atac (needs DATAPATH, BAKLAVA on path).",
    )
    p.add_argument("--gene-pool", choices=["union", "targets", "sources"], default="union")
    p.add_argument("--min-genes", type=int, default=3)
    p.add_argument("--no-log1p", action="store_true", help="Use expression as-is (already log1p, etc.).")
    p.add_argument("--no-zscore-pathways", action="store_true")
    p.add_argument("--layer", default=None, help="Use adata.layers[layer] instead of X.")
    return p.parse_args(argv)


def main_cli(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    import scanpy as sc

    adata = sc.read_h5ad(args.adata)

    combined: Optional[Mapping[str, Any]] = None
    if args.gp_dict_pickle:
        import pickle

        with open(args.gp_dict_pickle, "rb") as f:
            combined = pickle.load(f)
        if not isinstance(combined, Mapping):
            print("--gp-dict-pickle must contain a mapping (dict-like).", file=sys.stderr)
            sys.exit(1)
    elif args.load_nichecompass_gp_dict:
        _repo = os.environ.get("BAKLAVA_BASE_DIR")
        if not _repo:
            print("BAKLAVA_BASE_DIR must be set to load the gp dict.", file=sys.stderr)
            sys.exit(1)
        _baklava = os.path.join(_repo, "BAKLAVA")
        if os.path.isdir(_baklava) and _baklava not in sys.path:
            sys.path.insert(0, _baklava)
        _mg = os.path.join(_repo, "MultiGATE")
        if os.path.isdir(_mg) and _mg not in sys.path:
            sys.path.insert(0, _mg)

        from mouse_brain_spatial_rna_atac import load_nichecompass_combined_gp_dict_mouse

        combined = load_nichecompass_combined_gp_dict_mouse(verbose=True)
    else:
        print(
            "Provide --gp-dict-pickle or --load-nichecompass-gp-dict, or call run_pathway_embedding_analysis() "
            "from Python.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = PathwayEmbeddingConfig(
        embedding_key=args.embedding_key,
        gene_pool=args.gene_pool,
        min_genes_per_pathway=args.min_genes,
        apply_log1p=not args.no_log1p,
        z_score_pathways=not args.no_zscore_pathways,
        layer=args.layer,
    )
    result = run_pathway_embedding_analysis(adata, combined, cfg)
    paths = save_pathway_embedding_results(result, args.out_dir)
    print("Wrote:", json.dumps(paths, indent=2))


if __name__ == "__main__":
    main_cli()
