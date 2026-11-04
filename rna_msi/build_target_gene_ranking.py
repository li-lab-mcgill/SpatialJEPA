#!/usr/bin/env python
"""Build a ranked target-gene panel for spatialMETA/SpatialJEPA ST feature selection.

Run this in the SEA-AD env (``envs/spatialjepa-analysis.yml``). It precomputes,
from the SEA-AD target data, a single ordered gene ranking that the spatialMETA training
script consumes (via ``SMA_TARGET_PANEL``) to restrict the model's ST features to genes
the transfer target actually measures:

  1. All measured spatial panel genes are forced to the TOP (whitelist) so the single
     trained model still covers the targeted MERFISH/Xenium panel.
  2. The remaining slots are snRNA-seq genes ranked by variance (most variable first).

The output is a small CSV (``gene``, ``rank``, ``whitelist``, ``variance_score``); the
training script never has to load the full target data.

Example:
    python build_target_gene_ranking.py \
        --snrna-h5ad /path/SEAAD_MTG_snRNA.h5ad \
        --spatial-h5ad /path/SEAAD_MTG_MERFISH.h5ad \
        --out /path/rankings --n-top 4000

Protein-coding-only panel:
    python build_target_gene_ranking.py \
        --snrna-h5ad /path/CaH_final_nuclei.h5ad \
        --spatial-h5ad /path/CaH_Xenium_final.h5ad \
        --protein-coding-only \
        --out /path/rankings --n-top 4000
"""

from __future__ import annotations
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402

import argparse
import re
from importlib import util as importlib_util
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

BIOMART_ATTRS = ["ensembl_gene_id", "external_gene_name", "gene_biotype"]


def _norm(name) -> str:
    """Bare, case-folded gene symbol (mirrors the training-side consumer)."""
    text = str(name).strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text.upper()


def safe_path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def target_name_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".h5ad", ".h5", ".zarr"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def panel_name_for_output(target_name: str, *, biotype_filter: str | None) -> str:
    if biotype_filter is None:
        return target_name
    token = safe_path_token(biotype_filter)
    if token.lower() in target_name.lower():
        return target_name
    return f"{target_name}_{token}"


def resolve_output_path(out: Path, *, target_name: str) -> Path:
    """Use a target-specific filename when --out is a directory-like path."""
    target_token = safe_path_token(target_name)
    filename = f"target_gene_ranking_{target_token}.csv"
    if out.exists() and out.is_dir():
        return out / filename
    if str(out).endswith("/"):
        return out / filename
    if out.suffix == "":
        return out / filename
    if target_token.lower() not in out.stem.lower():
        return out.with_name(f"{out.stem}_{target_token}{out.suffix}")
    return out


def _clean_string_series(values, *, index=None) -> pd.Series:
    series = pd.Series(values, index=index, dtype="string").str.strip()
    return series.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _strip_ensembl_version(values, *, index=None) -> pd.Series:
    series = _clean_string_series(values, index=index)
    return series.str.replace(r"\.\d+$", "", regex=True)


def _dedupe_annotation_map(annotations: pd.DataFrame, key: str) -> pd.Series:
    table = annotations[[key, "gene_biotype"]].dropna(subset=[key, "gene_biotype"])
    table = table.drop_duplicates(key, keep="first")
    return table.set_index(key)["gene_biotype"]


def normalize_biotype_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in BIOMART_ATTRS if col not in annotations.columns]
    if missing:
        raise ValueError(
            f"Biotype annotation table is missing required columns: {missing}. "
            f"Expected at least {BIOMART_ATTRS}."
        )
    annot = annotations.loc[:, BIOMART_ATTRS].copy()
    annot["ensembl_gene_id"] = _strip_ensembl_version(annot["ensembl_gene_id"])
    annot["external_gene_name"] = _clean_string_series(annot["external_gene_name"])
    annot["gene_biotype"] = _clean_string_series(annot["gene_biotype"])
    return annot.dropna(subset=["gene_biotype"])


def load_biotype_annotations(
    *,
    annotation_csv: Path | None,
    species: str,
    use_cache: bool,
) -> pd.DataFrame:
    if annotation_csv is not None:
        print(f"[generator] gene biotypes from {annotation_csv}")
        annotations = pd.read_csv(annotation_csv)
    else:
        print(f"[generator] gene biotypes from Ensembl BioMart species={species!r}")
        annotations = sc.queries.biomart_annotations(
            species,
            attrs=BIOMART_ATTRS,
            use_cache=use_cache,
        )
    annot = normalize_biotype_annotations(annotations)
    print(f"[generator] loaded {len(annot):,} gene biotype annotation rows")
    return annot


def infer_gene_biotypes(
    var: pd.DataFrame,
    *,
    annotations: pd.DataFrame | None,
    gene_id_col: str,
    symbol_col: str | None,
) -> pd.Series:
    if "gene_biotype" in var.columns:
        return _clean_string_series(var["gene_biotype"], index=var.index)
    if annotations is None:
        raise ValueError("Biotype filtering needs annotations or an existing var['gene_biotype'] column.")

    biotypes = pd.Series(pd.NA, index=var.index, dtype="object")

    annot_by_id = _dedupe_annotation_map(annotations, "ensembl_gene_id")
    if gene_id_col in var.columns:
        gene_ids = _strip_ensembl_version(var[gene_id_col], index=var.index)
        biotypes = gene_ids.map(annot_by_id)
    else:
        index_ids = _strip_ensembl_version(var.index, index=var.index)
        index_id_mask = index_ids.str.startswith("ENS", na=False)
        if index_id_mask.any():
            biotypes.loc[index_id_mask] = index_ids.loc[index_id_mask].map(annot_by_id)

    missing = biotypes.isna()
    if missing.any():
        annot_by_symbol = _dedupe_annotation_map(annotations, "external_gene_name")
        if symbol_col is not None and symbol_col in var.columns:
            symbols = _clean_string_series(var[symbol_col], index=var.index)
        else:
            symbols = _clean_string_series(var.index, index=var.index)
        biotypes.loc[missing] = symbols.loc[missing].map(annot_by_symbol)

    return _clean_string_series(biotypes, index=var.index)


def biotype_mask_for_var(
    var: pd.DataFrame,
    *,
    annotations: pd.DataFrame | None,
    biotype_filter: str | None,
    gene_id_col: str,
    symbol_col: str | None,
    label: str,
) -> pd.Series:
    if biotype_filter is None:
        return pd.Series(True, index=var.index)
    biotypes = infer_gene_biotypes(
        var,
        annotations=annotations,
        gene_id_col=gene_id_col,
        symbol_col=symbol_col,
    )
    mask = biotypes.eq(biotype_filter).fillna(False).astype(bool)
    matched = int(biotypes.notna().sum())
    kept = int(mask.sum())
    print(
        f"[generator] {label}: {kept:,}/{len(var):,} genes match "
        f"gene_biotype={biotype_filter!r} ({matched:,} annotated)"
    )
    if kept == 0:
        raise ValueError(f"{label}: no genes matched gene_biotype={biotype_filter!r}.")
    return mask


def load_spatial_panel(
    path: Path,
    *,
    blank_prefix: str = "Blank-",
    include_control_probes: bool = False,
    annotations: pd.DataFrame | None = None,
    biotype_filter: str | None = None,
    gene_id_col: str = "gene_ids",
    symbol_col: str | None = None,
) -> list[str]:
    """Spatial panel gene symbols from var_names, excluding controls by default."""
    adata = sc.read_h5ad(path, backed="r")
    try:
        var = adata.var.copy()
        var.index = pd.Index(adata.var_names.astype(str))
        if "feature_types" in var:
            var = var.loc[var["feature_types"].astype(str).eq("Gene Expression")].copy()
        if not include_control_probes and blank_prefix:
            var = var.loc[~var.index.astype(str).str.startswith(blank_prefix)].copy()
        mask = biotype_mask_for_var(
            var,
            annotations=annotations,
            biotype_filter=biotype_filter,
            gene_id_col=gene_id_col,
            symbol_col=symbol_col,
            label="spatial panel",
        )
        panel = var.index[mask.to_numpy()].astype(str).tolist()
    finally:
        adata.file.close()
    if not panel:
        raise ValueError(f"No spatial panel genes found in {path}.")
    return panel


def _load_snrna_for_ranking(
    path: Path,
    *,
    gene_col: str | None,
    layer: str | None,
    subsample_cells: int | None,
) -> "sc.AnnData":
    """Load snRNA into memory, subsampling first when the object is large."""
    backed = sc.read_h5ad(path, backed="r")
    try:
        if gene_col is not None:
            if gene_col not in backed.var.columns:
                raise ValueError(f"--gene-col {gene_col!r} not in snRNA var columns.")
            backed.var_names = backed.var[gene_col].astype(str)
        backed.var_names_make_unique()
        if layer is not None and layer not in backed.layers:
            raise ValueError(f"--snrna-layer {layer!r} not in snRNA layers: {list(backed.layers.keys())}")

        n_obs = backed.n_obs
        if subsample_cells is not None and n_obs > subsample_cells:
            rng = np.random.default_rng(0)
            obs_idx = np.sort(rng.choice(n_obs, size=subsample_cells, replace=False))
            print(f"[generator] subsampling {n_obs:,} snRNA cells -> {subsample_cells:,} for variance ranking")
            adata = backed[obs_idx].to_memory()
        else:
            print(f"[generator] loading all {n_obs:,} snRNA cells into memory")
            adata = backed.to_memory()
    finally:
        backed.file.close()
    if layer is not None:
        print(f"[generator] ranking from snRNA layer {layer!r}")
        adata.X = adata.layers[layer].copy()
    return adata


def sample_nonzero_values(matrix, max_values: int = 100_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    if sparse.issparse(matrix):
        values = np.asarray(matrix.data)
    else:
        arr = np.asarray(matrix)
        if arr.size > max_values:
            rows = rng.integers(0, arr.shape[0], size=max_values)
            cols = rng.integers(0, arr.shape[1], size=max_values)
            values = arr[rows, cols]
        else:
            values = arr.ravel()
        values = values[values != 0]
    if values.size > max_values:
        values = values[rng.choice(values.size, size=max_values, replace=False)]
    return values.astype(float, copy=False)


def describe_hvg_matrix_state(
    adata: "sc.AnnData",
    *,
    trust_uns_log1p: bool = True,
) -> dict[str, float | bool | str]:
    """Infer whether X is raw count-like or already processed for HVG selection."""
    values = sample_nonzero_values(adata.X)
    if values.size == 0:
        return {
            "state": "empty",
            "count_like": False,
            "has_log1p_uns": trust_uns_log1p and "log1p" in adata.uns,
            "max": np.nan,
            "integer_fraction": np.nan,
        }
    nonnegative = bool(np.nanmin(values) >= 0)
    integer_fraction = float(np.mean(np.isclose(values, np.round(values), atol=1e-6)))
    max_value = float(np.nanmax(values))
    has_log1p_uns = trust_uns_log1p and "log1p" in adata.uns
    count_like = nonnegative and integer_fraction > 0.999 and not has_log1p_uns
    if count_like:
        state = "raw_counts"
    elif has_log1p_uns or (nonnegative and max_value <= 30):
        state = "processed_log_like"
    else:
        state = "processed_unknown"
    return {
        "state": state,
        "count_like": count_like,
        "has_log1p_uns": has_log1p_uns,
        "max": max_value,
        "integer_fraction": integer_fraction,
    }


def prepare_for_hvg(
    adata: "sc.AnnData",
    *,
    flavor: str,
    trust_uns_log1p: bool = True,
) -> None:
    state = describe_hvg_matrix_state(adata, trust_uns_log1p=trust_uns_log1p)
    print(
        "[generator] snRNA X preprocessing check: "
        f"state={state['state']}, max={state['max']:.3g}, "
        f"integer_fraction={state['integer_fraction']:.3f}, "
        f"uns_log1p={state['has_log1p_uns']}"
    )
    if flavor == "seurat_v3":
        if not state["count_like"]:
            print(
                "[generator] flavor=seurat_v3 usually expects raw counts; "
                "using existing processed values because they do not look count-like."
            )
        return
    if state["count_like"]:
        print("[generator] normalizing total counts and applying log1p before HVG ranking")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("[generator] using existing snRNA values for HVG ranking; skipping normalize_total/log1p")


def rank_snrna_by_variance(
    path: Path,
    *,
    flavor: str,
    n_top: int | None,
    gene_col: str | None,
    layer: str | None,
    subsample_cells: int | None,
    annotations: pd.DataFrame | None,
    biotype_filter: str | None,
    gene_id_col: str,
    symbol_col: str | None,
) -> pd.DataFrame:
    """Return snRNA genes ordered most->least variable with a variance score."""
    adata = _load_snrna_for_ranking(
        path,
        gene_col=gene_col,
        layer=layer,
        subsample_cells=subsample_cells,
    )
    mask = biotype_mask_for_var(
        adata.var,
        annotations=annotations,
        biotype_filter=biotype_filter,
        gene_id_col=gene_id_col,
        symbol_col=symbol_col,
        label="snRNA",
    )
    if biotype_filter is not None:
        adata._inplace_subset_var(mask.to_numpy(dtype=bool))
    prepare_for_hvg(adata, flavor=flavor, trust_uns_log1p=layer is None)

    if flavor == "seurat_v3":
        if importlib_util.find_spec("skmisc") is None:
            raise RuntimeError(
                "flavor='seurat_v3' requires scikit-misc (skmisc), which is not installed "
                "in this env. Use the default --flavor seurat."
            )
        sc.pp.highly_variable_genes(
            adata, flavor="seurat_v3", n_top_genes=(n_top or adata.n_vars)
        )
        score = adata.var["variances_norm"]
        # seurat_v3 gives an explicit rank (1 = most variable); lower first, NaN last.
        order = adata.var["highly_variable_rank"].sort_values(na_position="last").index
    else:
        sc.pp.highly_variable_genes(adata, flavor=flavor)
        score = adata.var["dispersions_norm"]
        order = score.sort_values(ascending=False, na_position="last").index

    ranked = pd.DataFrame(
        {"gene": pd.Index(order).astype(str), "variance_score": score.reindex(order).to_numpy()}
    )
    if n_top is not None:
        ranked = ranked.head(n_top)
    return ranked.reset_index(drop=True)


def build_ranking(
    spatial_panel: list[str],
    snrna_ranked: pd.DataFrame,
) -> pd.DataFrame:
    """Whitelist spatial panel genes on top, then snRNA by variance, deduped by symbol."""
    rows = []
    seen: set[str] = set()
    for gene in spatial_panel:
        key = _norm(gene)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"gene": gene, "whitelist": True, "variance_score": np.nan})
    for gene, score in zip(snrna_ranked["gene"], snrna_ranked["variance_score"]):
        key = _norm(gene)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"gene": gene, "whitelist": False, "variance_score": score})

    df = pd.DataFrame(rows)
    df.insert(1, "rank", np.arange(len(df), dtype=int))
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snrna-h5ad", required=True, type=Path)
    parser.add_argument(
        "--spatial-h5ad",
        "--merfish-h5ad",
        dest="spatial_h5ad",
        required=True,
        type=Path,
        help="Spatial target h5ad whose measured panel genes should be whitelisted.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--target-name",
        default=None,
        help="Name to use in the default output filename. Defaults to the spatial h5ad filename stem.",
    )
    parser.add_argument("--n-top", type=int, default=None,
                        help="Cap on number of snRNA genes written after the whitelist.")
    parser.add_argument("--flavor", default="seurat", choices=["seurat", "cell_ranger", "seurat_v3"],
                        help="HVG flavor. Default 'seurat' (no extra deps). 'seurat_v3' needs skmisc.")
    parser.add_argument("--gene-col", default=None,
                        help="snRNA var column to use as gene symbol (default: var_names).")
    parser.add_argument("--snrna-layer", default=None,
                        help="Optional snRNA layer to rank instead of X, e.g. UMIs.")
    parser.add_argument(
        "--protein-coding-only",
        action="store_true",
        help="Restrict both the spatial whitelist and snRNA-ranked tail to protein-coding genes.",
    )
    parser.add_argument(
        "--biotype-annotation-csv",
        type=Path,
        default=None,
        help=(
            "Optional annotation CSV with ensembl_gene_id, external_gene_name, gene_biotype. "
            "If omitted with --protein-coding-only, Ensembl BioMart is queried via Scanpy."
        ),
    )
    parser.add_argument(
        "--biomart-species",
        default="hsapiens",
        help="BioMart species name for gene biotypes when querying Scanpy (default: hsapiens).",
    )
    parser.add_argument(
        "--no-biomart-cache",
        action="store_true",
        help="Disable Scanpy's BioMart annotation cache.",
    )
    parser.add_argument(
        "--gene-id-col",
        default="gene_ids",
        help="var column containing Ensembl gene IDs for biotype matching (default: gene_ids).",
    )
    parser.add_argument(
        "--symbol-col",
        default=None,
        help="Optional var column containing gene symbols for biotype matching fallback.",
    )
    parser.add_argument(
        "--subsample-cells",
        type=int,
        default=200_000,
        help=(
            "Subsample snRNA cells before variance ranking (default: 50000). "
            "Required for large cohort h5ads: loading all ~1.4M MTG nuclei can exceed RAM."
        ),
    )
    parser.add_argument(
        "--no-subsample",
        action="store_true",
        help="Load all snRNA cells (only use if you have enough RAM).",
    )
    parser.add_argument("--blank-prefix", default="Blank-",
                        help="Spatial control-probe prefix to drop when feature_types is unavailable.")
    parser.add_argument(
        "--include-control-probes",
        action="store_true",
        help="Whitelist control probes too. By default only spatial panel genes are whitelisted.",
    )
    args = parser.parse_args(argv)

    biotype_filter = "protein_coding" if args.protein_coding_only else None
    annotations = None
    if biotype_filter is not None:
        annotations = load_biotype_annotations(
            annotation_csv=args.biotype_annotation_csv,
            species=args.biomart_species,
            use_cache=not args.no_biomart_cache,
        )

    target_name = panel_name_for_output(
        args.target_name or target_name_from_path(args.spatial_h5ad),
        biotype_filter=biotype_filter,
    )

    print(f"[generator] spatial panel from {args.spatial_h5ad}")
    spatial_panel = load_spatial_panel(
        args.spatial_h5ad,
        blank_prefix=args.blank_prefix,
        include_control_probes=args.include_control_probes,
        annotations=annotations,
        biotype_filter=biotype_filter,
        gene_id_col=args.gene_id_col,
        symbol_col=args.symbol_col,
    )
    print(f"[generator] spatial panel genes (whitelist): {len(spatial_panel)}")

    subsample_cells = None if args.no_subsample else args.subsample_cells

    print(f"[generator] ranking snRNA genes from {args.snrna_h5ad} (flavor={args.flavor})")
    snrna_ranked = rank_snrna_by_variance(
        args.snrna_h5ad,
        flavor=args.flavor,
        n_top=args.n_top,
        gene_col=args.gene_col,
        layer=args.snrna_layer,
        subsample_cells=subsample_cells,
        annotations=annotations,
        biotype_filter=biotype_filter,
        gene_id_col=args.gene_id_col,
        symbol_col=args.symbol_col,
    )
    print(f"[generator] snRNA ranked genes: {len(snrna_ranked)}")

    ranking = build_ranking(spatial_panel, snrna_ranked)
    out_path = resolve_output_path(args.out, target_name=target_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_path, index=False)
    n_white = int(ranking["whitelist"].sum())
    print(
        f"[generator] wrote {len(ranking)} genes to {out_path} "
        f"({n_white} whitelist at top, {len(ranking) - n_white} snRNA-ranked)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
