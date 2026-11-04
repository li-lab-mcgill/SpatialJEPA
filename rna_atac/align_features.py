#!/usr/bin/env python3
#%%
"""Align source/target RNA+ATAC feature spaces by overlap.

This script performs only overlap-based feature alignment via ``DataAligner``
and intentionally omits NicheCompass-specific processing.
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402

import argparse
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import anndata as ad
import pandas as pd
from muon import MuData

#%%
import dotenv
dotenv.load_dotenv(os.environ["SPATIALJEPA_CONFIG"])

BAKLAVA_ROOT = os.environ["SPATIALJEPA_SHARED"]
if BAKLAVA_ROOT not in sys.path:
    sys.path.insert(0, BAKLAVA_ROOT)

from data_aligner import DataAligner
from data_utils import load_10x_mouse_brain_data, basic_feature_processing_for_alignment

REQUIRED_ATAC_COORD_COLS = ("chrom", "chromStart", "chromEnd")


#%%
def parse_args(notebook: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align source and target h5ad datasets by feature overlap."
    )
    parser.add_argument("--source-rna-h5ad", help="Path to source RNA h5ad.",
        default=f"{os.environ['SPATIALJEPA_HOME']}/data/Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain.h5ad",
    )
    parser.add_argument("--source-atac-h5ad", help="Path to source ATAC h5ad.",
        default=f"{os.environ['SPATIALJEPA_HOME']}/data/Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain_atac.h5ad",
    )
    parser.add_argument("--target-rna-h5ad", help="Path to target RNA h5ad.",
        default=None,
    )
    parser.add_argument("--target-atac-h5ad", help="Path to target ATAC h5ad.",
        default=None,
    )
    parser.add_argument("--source-assembly", help="Genome assembly for source data.",
        default="mm10",
    )
    parser.add_argument("--target-assembly", help="Genome assembly for target data.",
        default="mm10",
    )
    parser.add_argument("--source-name", help="Source dataset name.",
        default="Spatial_ATAC_RNA",
    )
    parser.add_argument("--target-name", help="Target dataset name.",
        default="10x_mouse_brain",
    )
    parser.add_argument("--aligned-outdir", default=os.path.join(os.getenv("SPATIALJEPA_DATA"), "aligned_data"), help="Output directory for aligned h5ad files.",)
    parser.add_argument("--mappers-out", default=None, help="Output pickle path for alignment mappers and metadata.",)
    parser.add_argument("--source-atac-peak-col", default=None, help="Column in source ATAC .var to parse peak coordinates from when needed.",)
    parser.add_argument("--target-atac-peak-col", default=None, help="Column in target ATAC .var to parse peak coordinates from when needed.",)

    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


def _ensure_outputs_selected(args: argparse.Namespace) -> None:
    if not args.aligned_outdir and not args.mappers_out:
        raise ValueError(
            "No output selected. Provide at least one of --aligned-outdir or --mappers-out."
        )


def _check_runtime_dependencies() -> None:
    if shutil.which("bedtools") is None:
        raise EnvironmentError(
            "bedtools is required for peak overlap alignment (pybedtools). "
            "Install bedtools and ensure it is on PATH."
        )


def _parse_peak_series_to_coords(peak_values: Iterable[str], label: str) -> pd.DataFrame:
    rows = []
    failed = []
    for idx, raw in enumerate(peak_values):
        value = "" if raw is None else str(raw).strip()
        chrom = None
        start = None
        end = None

        m_colon = re.match(r"^([^:]+):(\d+)-(\d+)$", value)
        if m_colon is not None:
            chrom, start, end = m_colon.group(1), m_colon.group(2), m_colon.group(3)
        else:
            m_dash = re.match(r"^(.+)-(\d+)-(\d+)$", value)
            if m_dash is not None:
                chrom, start, end = m_dash.group(1), m_dash.group(2), m_dash.group(3)

        if chrom is None:
            failed.append((idx, value))
            rows.append((None, None, None))
            continue

        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            failed.append((idx, value))
            rows.append((None, None, None))
            continue
        rows.append((chrom, start_i, end_i))

    if failed:
        examples = ", ".join([f"#{i}='{v}'" for i, v in failed[:5]])
        raise ValueError(
            f"{label} could not parse ATAC peaks for {len(failed)} features. "
            "Expected format 'chr:start-end' or 'chr-start-end'. "
            f"Examples of invalid entries: {examples}"
        )

    return pd.DataFrame(rows, columns=["chrom", "chromStart", "chromEnd"])


def _ensure_atac_coords(
    adata_atac: ad.AnnData,
    dataset_label: str,
    peak_col: str = None,
) -> str:
    has_cols = all(col in adata_atac.var.columns for col in REQUIRED_ATAC_COORD_COLS)
    if has_cols:
        adata_atac.var["chrom"] = adata_atac.var["chrom"].astype(str)
        adata_atac.var["chromStart"] = pd.to_numeric(
            adata_atac.var["chromStart"], errors="raise"
        ).astype(int)
        adata_atac.var["chromEnd"] = pd.to_numeric(
            adata_atac.var["chromEnd"], errors="raise"
        ).astype(int)
        return "existing_columns"

    parse_source = None
    parse_source_name = None
    if peak_col is not None:
        if peak_col not in adata_atac.var.columns:
            raise ValueError(
                f"{dataset_label} requested peak column '{peak_col}' is missing from ATAC .var."
            )
        parse_source = adata_atac.var[peak_col]
        parse_source_name = f"column '{peak_col}'"
    elif "peak" in adata_atac.var.columns:
        parse_source = adata_atac.var["peak"]
        parse_source_name = "column 'peak'"
    else:
        parse_source = pd.Series(adata_atac.var_names, index=adata_atac.var_names)
        parse_source_name = "var_names"

    coords = _parse_peak_series_to_coords(parse_source, f"{dataset_label} ({parse_source_name})")
    coords.index = adata_atac.var.index
    adata_atac.var["chrom"] = coords["chrom"].astype(str)
    adata_atac.var["chromStart"] = coords["chromStart"].astype(int)
    adata_atac.var["chromEnd"] = coords["chromEnd"].astype(int)
    return parse_source_name


def _compute_peak_mapper_used(data_aligner: DataAligner) -> pd.Series:
    mapper = data_aligner.peak_name_mapper
    if isinstance(mapper, dict):
        mapper = pd.Series(mapper)

    source_atac = data_aligner.source_data["atac"]
    target_atac = data_aligner.target_data["atac"]
    mapper = mapper.loc[
        mapper.index.isin(source_atac.var_names) & mapper.isin(target_atac.var_names)
    ]
    mapper = mapper[~mapper.index.duplicated(keep="first")]
    mapper = mapper[~mapper.duplicated(keep="first")]
    return mapper


def _int_indexer(index: pd.Index, values: Iterable[str], label: str) -> list:
    idx = index.get_indexer(list(values))
    if (idx < 0).any():
        missing_count = int((idx < 0).sum())
        raise ValueError(
            f"{label}: failed to compute indexer for {missing_count} selected features."
        )
    return idx.astype(int).tolist()


def _save_aligned_h5ads(
    aligned_outdir: str,
    source_rna: ad.AnnData,
    source_atac: ad.AnnData,
    target_rna: ad.AnnData,
    target_atac: ad.AnnData,
) -> Dict[str, str]:
    outdir = Path(aligned_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "source_rna": outdir / "source_rna_aligned.h5ad",
        "source_atac": outdir / "source_atac_aligned.h5ad",
        "target_rna": outdir / "target_rna_aligned.h5ad",
        "target_atac": outdir / "target_atac_aligned.h5ad",
    }
    source_rna.write_h5ad(paths["source_rna"])
    source_atac.write_h5ad(paths["source_atac"])
    target_rna.write_h5ad(paths["target_rna"])
    target_atac.write_h5ad(paths["target_atac"])
    return {k: str(v) for k, v in paths.items()}


def _build_mapper_payload(
    *,
    args: argparse.Namespace,
    gene_order: list,
    peak_mapper_used: pd.Series,
    orig_source_rna_var_names: pd.Index,
    orig_target_rna_var_names: pd.Index,
    source_atac_bed_var_names_prealign: pd.Index,
    target_atac_bed_var_names_prealign: pd.Index,
    source_shapes_before: Dict[str, Tuple[int, int]],
    target_shapes_before: Dict[str, Tuple[int, int]],
    source_shapes_after: Dict[str, Tuple[int, int]],
    target_shapes_after: Dict[str, Tuple[int, int]],
) -> dict:
    source_rna_var_idx = _int_indexer(
        orig_source_rna_var_names, gene_order, "source_rna_var_idx"
    )
    target_rna_var_idx = _int_indexer(
        orig_target_rna_var_names, gene_order, "target_rna_var_idx"
    )
    source_atac_var_idx = _int_indexer(
        source_atac_bed_var_names_prealign,
        peak_mapper_used.index.tolist(),
        "source_atac_var_idx",
    )
    target_atac_var_idx = _int_indexer(
        target_atac_bed_var_names_prealign,
        peak_mapper_used.values.tolist(),
        "target_atac_var_idx",
    )

    return {
        "gene_order": list(gene_order),
        "peak_source_to_target": peak_mapper_used.to_dict(),
        "source_rna_var_names_selected": list(gene_order),
        "target_rna_var_names_selected": list(gene_order),
        "source_atac_var_names_selected": peak_mapper_used.index.tolist(),
        "target_atac_var_names_selected": peak_mapper_used.values.tolist(),
        "source_rna_var_idx": source_rna_var_idx,
        "target_rna_var_idx": target_rna_var_idx,
        "source_atac_var_idx": source_atac_var_idx,
        "target_atac_var_idx": target_atac_var_idx,
        "metadata": {
            "source_name": args.source_name,
            "target_name": args.target_name,
            "source_assembly": args.source_assembly,
            "target_assembly": args.target_assembly,
            "source_shapes_before": source_shapes_before,
            "target_shapes_before": target_shapes_before,
            "source_shapes_after": source_shapes_after,
            "target_shapes_after": target_shapes_after,
            "n_genes_aligned": len(gene_order),
            "n_peaks_aligned": int(len(peak_mapper_used)),
            "inputs": {
                "source_rna_h5ad": args.source_rna_h5ad,
                "source_atac_h5ad": args.source_atac_h5ad,
                "target_rna_h5ad": args.target_rna_h5ad,
                "target_atac_h5ad": args.target_atac_h5ad,
            },
        },
    }


def _write_pickle(path: str, payload: dict) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return str(out_path)

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

#%%
def main() -> None:
#%%
    NOTEBOOK = is_notebook()
    args = parse_args(notebook=NOTEBOOK)

    _ensure_outputs_selected(args)
    _check_runtime_dependencies()

    print("[INFO] Loading input h5ad files...")
    source_rna = ad.read_h5ad(args.source_rna_h5ad)
    source_atac = ad.read_h5ad(args.source_atac_h5ad)

    if (args.target_rna_h5ad is not None) and (args.target_atac_h5ad is not None):
        target_rna = ad.read_h5ad(args.target_rna_h5ad)
        target_atac = ad.read_h5ad(args.target_atac_h5ad)
    elif args.target_name == "10x_mouse_brain":
        target_rna, target_atac, target_assembly, target_name = load_10x_mouse_brain_data()

    print(f"[INFO] Source RNA   X range: min={source_rna.X.min()}, max={source_rna.X.max()}")
    print(f"[INFO] Source ATAC  X range: min={source_atac.X.min()}, max={source_atac.X.max()}")
    print(f"[INFO] Target RNA   X range: min={target_rna.X.min()}, max={target_rna.X.max()}")
    print(f"[INFO] Target ATAC  X range: min={target_atac.X.min()}, max={target_atac.X.max()}")

    # preprocess data
    print("[INFO] Preprocessing data...")
    source_rna, source_atac, target_rna, target_atac = basic_feature_processing_for_alignment(
        source_rna, source_atac, target_rna, target_atac
    )
    
    source_shapes_before = {
        "rna": source_rna.shape,
        "atac": source_atac.shape,
    }
    target_shapes_before = {
        "rna": target_rna.shape,
        "atac": target_atac.shape,
    }
    print(
        "[INFO] Loaded source RNA/ATAC and target RNA/ATAC: "
        f"source_rna={source_rna.shape}, source_atac={source_atac.shape}, "
        f"target_rna={target_rna.shape}, target_atac={target_atac.shape}"
    )

    target_rna = target_rna[:, ~target_rna.var_names.duplicated(keep="first")].copy()

    src_atac_coord_source = _ensure_atac_coords(
        source_atac, "source ATAC", peak_col=args.source_atac_peak_col
    )
    tgt_atac_coord_source = _ensure_atac_coords(
        target_atac, "target ATAC", peak_col=args.target_atac_peak_col
    )
    print(
        "[INFO] ATAC coordinate columns ready: "
        f"source={src_atac_coord_source}, target={tgt_atac_coord_source}"
    )

    orig_source_rna_var_names = source_rna.var_names.copy()
    orig_target_rna_var_names = target_rna.var_names.copy()

    source_data = MuData({"rna": source_rna, "atac": source_atac})
    target_data = MuData({"rna": target_rna, "atac": target_atac})

    data_aligner = DataAligner(
        source_data=source_data,
        target_data=target_data,
        source_assembly=args.source_assembly,
        target_assembly=args.target_assembly,
        source_name=args.source_name,
        target_name=args.target_name,
        paired_target=True,
    )
    print("[INFO] Running overlap-based alignment...")
    data_aligner.find_gene_overlap()
    data_aligner.find_peak_overlap()

    gene_order = [
        g
        for g in data_aligner.gene_overlap
        if (g in data_aligner.source_data["rna"].var_names)
        and (g in data_aligner.target_data["rna"].var_names)
    ]
    peak_mapper_used = _compute_peak_mapper_used(data_aligner)
    source_atac_bed_var_names_prealign = data_aligner.source_data["atac"].var_names.copy()
    target_atac_bed_var_names_prealign = data_aligner.target_data["atac"].var_names.copy()
    print(
        f"[INFO] Overlap counts before final subsetting: genes={len(gene_order)}, "
        f"peaks={len(peak_mapper_used)}"
    )

    data_aligner.align_features_by_overlap()
    source_rna_aligned = data_aligner.source_data["rna"]
    source_atac_aligned = data_aligner.source_data["atac"]
    target_rna_aligned = data_aligner.target_data["rna"]
    target_atac_aligned = data_aligner.target_data["atac"]

    if not source_rna_aligned.var_names.equals(target_rna_aligned.var_names):
        raise RuntimeError("Aligned RNA features are not identical between source and target.")
    if not source_atac_aligned.var_names.equals(target_atac_aligned.var_names):
        raise RuntimeError("Aligned ATAC features are not identical between source and target.")

    source_shapes_after = {
        "rna": source_rna_aligned.shape,
        "atac": source_atac_aligned.shape,
    }
    target_shapes_after = {
        "rna": target_rna_aligned.shape,
        "atac": target_atac_aligned.shape,
    }
    print(
        "[INFO] Alignment complete:\n "
        f"source_rna={source_rna_aligned.shape}\n source_atac={source_atac_aligned.shape}\n "
        f"target_rna={target_rna_aligned.shape}\n target_atac={target_atac_aligned.shape}"
    )

    #%%
    if args.aligned_outdir:
        saved_paths = _save_aligned_h5ads(
            aligned_outdir=args.aligned_outdir,
            source_rna=source_rna_aligned,
            source_atac=source_atac_aligned,
            target_rna=target_rna_aligned,
            target_atac=target_atac_aligned,
        )
        print("[INFO] Saved aligned datasets:")
        for key, value in saved_paths.items():
            print(f"  - {key}: {value}")

    if args.mappers_out:
        mapper_payload = _build_mapper_payload(
            args=args,
            gene_order=gene_order,
            peak_mapper_used=peak_mapper_used,
            orig_source_rna_var_names=orig_source_rna_var_names,
            orig_target_rna_var_names=orig_target_rna_var_names,
            source_atac_bed_var_names_prealign=source_atac_bed_var_names_prealign,
            target_atac_bed_var_names_prealign=target_atac_bed_var_names_prealign,
            source_shapes_before=source_shapes_before,
            target_shapes_before=target_shapes_before,
            source_shapes_after=source_shapes_after,
            target_shapes_after=target_shapes_after,
        )
        mapper_path = _write_pickle(args.mappers_out, mapper_payload)
        print(f"[INFO] Saved alignment mappers: {mapper_path}")

    print("[INFO] Done.")

#%%
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
