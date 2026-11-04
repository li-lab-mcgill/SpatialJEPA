"""Target-aware ST feature-panel restriction for spatialMETA/SpatialJEPA training.

At transfer time the SpatialJEPA RNA encoder is frozen to the ST genes chosen at
training time. When the model is applied to a different target (e.g. human SEA-AD
snRNA-seq / MERFISH), genes the target does not measure are zero-filled, which
degrades the encoder input -- badly for small targeted panels like MERFISH.

This module restricts the joint AnnData's ST features to a ranked target panel
*before* spatial-variability selection, so the trained model only keeps ST genes the
target also measures. The companion generator (``human_cah/build_target_gene_ranking.py``)
produces the ranked panel CSV; here we only consume it.

Matching is case-insensitive on the bare gene symbol (the SMA source is mouse, e.g.
``Snca``; SEA-AD targets are human, e.g. ``SNCA``), mirroring the uppercase tokenization
used by the inference worker (``embed_with_spatialmeta.feature_tokens``).
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

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _norm(name) -> str:
    """Normalize a feature name to a bare, case-folded gene symbol for matching.

    Strips a leading modality prefix (``rna:``/``msi:``) and uppercases, matching the
    inference worker so the panel selected at training aligns with what can be matched
    at inference (including mouse->human symbol case differences).
    """
    text = str(name).strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text.upper()


def load_target_panel(path) -> list[str]:
    """Load the ranked target gene panel, returning symbols in priority order.

    Accepts the generator's CSV (a ``gene`` column in rank order) or a plain
    newline-delimited list (treated as an unordered allowlist). Order is preserved as
    written -- whitelisted MERFISH genes are expected at the top.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target panel file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception:
        df = None

    if df is not None and "gene" in df.columns:
        genes = df["gene"].astype(str).tolist()
    else:
        # Fallback: plain one-symbol-per-line file (no header), unordered allowlist.
        with path.open("r", encoding="utf-8") as handle:
            genes = [line.strip() for line in handle if line.strip()]

    if not genes:
        raise ValueError(f"Target panel file {path} contained no gene symbols.")
    return genes


def save_target_panel(genes, path, *, source: str | None = None) -> Path:
    """Write a ranked target gene panel CSV compatible with :func:`load_target_panel`.

    Stores a ``gene`` column in priority order plus an integer ``rank``. Optional
    ``source`` records where the panel was derived from (e.g. input h5ad path).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gene_list = pd.Index(genes).astype(str).tolist()
    if not gene_list:
        raise ValueError("Cannot save an empty target gene panel.")

    payload = pd.DataFrame({"gene": gene_list, "rank": range(len(gene_list))})
    if source is not None:
        payload["source"] = source
    payload.to_csv(path, index=False)
    return path


def restrict_st_to_target_panel(
    joint_adata,
    ranked_genes: list[str],
    n_target_top: Optional[int] = None,
    *,
    min_st_after: int = 50,
):
    """Subset ST features of ``joint_adata`` to the top of a ranked target panel.

    SM features are always retained. The first ``n_target_top`` ranked genes (all if
    ``None``) form the allowlist; an ST feature is kept iff its case-folded symbol is in
    that allowlist. Returns ``(restricted_adata, diagnostics)``.
    """
    var_types = joint_adata.var["type"].astype(str).to_numpy()
    is_st = var_types == "ST"
    is_sm = var_types == "SM"

    selected = ranked_genes if n_target_top is None else ranked_genes[:n_target_top]
    allow = {_norm(g) for g in selected}

    st_norm = pd.Index([_norm(v) for v in joint_adata.var_names])
    keep_st = is_st & np.asarray(st_norm.isin(allow))
    keep = keep_st | is_sm

    n_st_before = int(is_st.sum())
    n_st_after = int(keep_st.sum())
    n_sm = int(is_sm.sum())

    matched_panel = allow & set(st_norm[is_st])
    matched_fraction_of_panel = len(matched_panel) / max(len(allow), 1)
    unmatched_panel_examples = sorted(allow - set(st_norm[is_st]))[:5]

    if n_st_after < min_st_after:
        warnings.warn(
            f"restrict_st_to_target_panel kept only {n_st_after} ST features "
            f"(from {n_st_before}); target panel may not match the source gene names. "
            f"matched_fraction_of_panel={matched_fraction_of_panel:.3f}.",
            stacklevel=2,
        )

    restricted = joint_adata[:, keep].copy()

    diagnostics = {
        "n_st_before": n_st_before,
        "n_st_after": n_st_after,
        "n_sm": n_sm,
        "n_panel": len(allow),
        "n_target_top": n_target_top,
        "matched_fraction_of_panel": float(matched_fraction_of_panel),
        "unmatched_panel_examples": unmatched_panel_examples,
    }
    return restricted, diagnostics
