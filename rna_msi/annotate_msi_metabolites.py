"""
Unified, matrix-aware annotation for SMA MALDI-MSI features.

Why this exists
---------------
Across the SMA sections, the *right* primary annotation source depends on the
MALDI matrix (verified against metadata.csv and the METASPACE datasets):

  * FMP-10 sections (V11L12-109, V11T16-085, V11T17-102) -- "neurotransmitters".
    The analytes are FMP-10 *derivatized*, so native-mass METASPACE databases
    cannot find them (e.g. dopamine sits at m/z 421.19 / 674.28, not ~154).
    The authoritative annotations are the authors' MS/MS-validated FMP-10
    panel (Vicari et al., Extended Data Fig. 9). METASPACE is only a
    complement here, for incidental *underivatized* species.

  * DHB sections (V11L12-038 A1/B1) -- "lipids", positive mode. No curated
    panel exists; METASPACE lipid databases (LipidMaps/SwissLipids) are primary.

  * 9-AA sections (V11L12-038 D1) -- "metabolites", negative mode. METASPACE
    metabolite databases (HMDB/CoreMetabolome/ChEBI/KEGG) are primary, with the
    lipid databases as a secondary complement (anionic lipids ionize in neg mode).

This module assigns, per MSI feature, a single primary annotation following that
matrix-keyed hierarchy, while keeping full provenance so the tiers never
silently overwrite each other:

  var["annotation"]             # primary label
  var["annotation_source"]      # "fmp_panel" or "metaspace:<db>"
  var["annotation_confidence"]  # "validated" | "high" | "medium" | "low" | "unranked"
  var["annotation_fdr"]         # METASPACE FDR (NaN for panel / unranked DBs)
  var["annotation_ppm"]         # mass match error of the chosen annotation
  var["annotation_all"]         # every candidate (all sources) for transparency

Confidence tiers: MS/MS-validated panel > METASPACE FDR<=0.05 (high) >
<=0.10 (medium) > <=0.20 (low). A low-FDR METASPACE hit never displaces a
panel call on the same peak.

The core (`annotate_var_table`) is pure pandas/numpy and operates on a `var`
DataFrame + a METASPACE results DataFrame, so it can be unit-tested and reused
for either a standalone MSI .h5ad or the "msi" modality of an aligned .h5mu.
"""
#%%
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
import ast
import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Matrix routing
# --------------------------------------------------------------------------- #

# Fallback when metadata.csv is unavailable. Plate prefixes that used FMP-10.
FMP10_PLATES = {"v11l12-109", "v11t16-085", "v11t17-102"}
DHB_STEMS = {"v11l12-038-a1", "v11l12-038-b1"}
NINEAA_STEMS = {"v11l12-038-d1"}


def sample_stem(sample_id: str) -> str:
    """V11L12-109_B1 -> v11l12-109-b1."""
    return sample_id.lower().replace("_", "-")


def matrix_for_sample(sample_id: str, metadata: pd.DataFrame | None = None) -> str:
    """Return the MALDI matrix ('FMP-10' | 'DHB' | '9-AA') for a sample.

    Prefers the authoritative `metadata.csv` (columns Sample.ID, Matrix);
    falls back to the plate/section mapping above.
    """
    row = metadata_for_sample(sample_id, metadata)
    if row is not None and "Matrix" in row.index:
        return str(row["Matrix"]).strip()

    stem = sample_stem(sample_id)
    plate = stem.rsplit("-", 1)[0]
    if plate in FMP10_PLATES:
        return "FMP-10"
    if stem in NINEAA_STEMS:
        return "9-AA"
    if stem in DHB_STEMS:
        return "DHB"
    return "unknown"


# --------------------------------------------------------------------------- #
# 2. FMP-10 MS/MS-validated neurotransmitter panel
# --------------------------------------------------------------------------- #

# Derivatized m/z are sample-independent (analyte + n x FMP-10 tag), so one
# panel applies to ALL FMP-10 sections. These nine come from the authors'
# V11L12-109_B1 annotated object; supply a fuller reference h5ad via
# load_fmp_panel(reference_h5ad=...) if you have one.
FMP10_PANEL = pd.DataFrame(
    {
        "name": [
            "GABA", "Taurine", "Dopamine (single)", "Histidine", "3-MT",
            "Serotonin", "Dopamine", "Norepinephrine", "Tocopherol",
        ],
        "mz": [
            371.17565, 393.12703, 421.19136, 423.18201, 435.20692,
            444.20715, 674.28050, 690.27454, 698.49196,
        ],
    }
)


def load_fmp_panel(reference_h5ad: str | Path | None = None) -> pd.DataFrame:
    """Build the FMP-10 panel from a reference annotated MSI .h5ad, or fall back
    to the built-in table. Expects var to carry `mz_raw`/`mz` and `annotated`.
    """
    if reference_h5ad and Path(reference_h5ad).exists():
        import anndata as ad

        var = ad.read_h5ad(reference_h5ad, backed="r").var
        mz_col = "mz_raw" if "mz_raw" in var.columns else "mz"
        names = var["annotated"] if "annotated" in var.columns else var.index
        panel = pd.DataFrame(
            {"name": list(map(str, names)), "mz": pd.to_numeric(var[mz_col], errors="coerce")}
        ).dropna(subset=["mz"])
        panel = panel[panel["name"].astype(str).str.strip() != ""].reset_index(drop=True)
        if len(panel):
            return panel
    return FMP10_PANEL.copy()


# --------------------------------------------------------------------------- #
# 3. METASPACE database priority per matrix
# --------------------------------------------------------------------------- #

DB_PRIORITY = {
    "DHB": ["lipidmaps", "swisslipids", "hmdb", "coremetabolome", "chebi", "kegg"],
    "9-AA": ["hmdb", "coremetabolome", "chebi", "kegg", "lipidmaps", "swisslipids"],
    # FMP-10: METASPACE is only the complement (native species). The panel
    # handles the derivatized neurotransmitters.
    "FMP-10": ["hmdb", "coremetabolome", "chebi", "kegg"],
}

# danielReceptorDB stores *native* neurotransmitter masses, which do NOT match
# the FMP-derivatized peaks -- exclude it from the FMP complement so it can't
# produce spurious native-mass hits.
EXCLUDE_DB = {"FMP-10": ["danielreceptordb"]}


def _db_rank(db_label: str, matrix: str) -> int:
    s = str(db_label).lower()
    order = DB_PRIORITY.get(matrix, DB_PRIORITY["9-AA"])
    for k, key in enumerate(order):
        if key in s:
            return k
    return len(order) + 1  # unknown DBs ranked last


# --------------------------------------------------------------------------- #
# 4. Matching helpers
# --------------------------------------------------------------------------- #

def _var_mz(var: pd.DataFrame) -> np.ndarray:
    col = "mz_raw" if "mz_raw" in var.columns else "mz"
    if col not in var.columns:
        raise KeyError("MSI var must contain 'mz_raw' or 'mz'.")
    return pd.to_numeric(var[col], errors="coerce").to_numpy()


def _parse_names(value) -> str:
    if isinstance(value, list):
        return ", ".join(map(str, value))
    if pd.isna(value):
        return ""
    try:
        parsed = ast.literal_eval(str(value))
        return ", ".join(map(str, parsed)) if isinstance(parsed, list) else str(parsed)
    except (ValueError, SyntaxError):
        return str(value)


def _fmt_fdr(fdr) -> str:
    return "NA" if pd.isna(fdr) else f"{fdr:.2f}"


def _confidence_from_fdr(fdr) -> str:
    if pd.isna(fdr):
        return "unranked"
    if fdr <= 0.05:
        return "high"
    if fdr <= 0.10:
        return "medium"
    return "low"


def match_panel(var: pd.DataFrame, panel: pd.DataFrame, ppm_tol: float) -> tuple[pd.Series, pd.Series]:
    """Assign each panel entry to its nearest var peak within ppm_tol.

    Panel-driven (not peak-driven) so each targeted analyte maps to its single
    best peak; on conflict the closer (smaller |ppm|) panel entry wins.
    """
    mz = _var_mz(var)
    name = pd.Series("", index=var.index, dtype=object)
    ppm = pd.Series(np.nan, index=var.index, dtype=float)

    for nm, pm in zip(panel["name"].astype(str), panel["mz"].astype(float)):
        d = np.abs(mz - pm)
        if not np.isfinite(d).any():
            continue
        j = int(np.nanargmin(d))
        e = (mz[j] - pm) / pm * 1e6
        if abs(e) <= ppm_tol and (name.iat[j] == "" or abs(e) < abs(ppm.iat[j])):
            name.iat[j] = nm
            ppm.iat[j] = e
    return name, ppm


def metaspace_best_per_peak(
    var: pd.DataFrame, results: pd.DataFrame, matrix: str, ppm_tol: float
) -> dict[int, dict]:
    """For each var peak, pick the best METASPACE annotation.

    Best = lowest FDR, then matrix DB priority, then smallest |ppm|.
    Returns {peak_index: {name, db, fdr, ppm, ion}}.
    """
    if results is None or len(results) == 0 or "mz" not in results.columns:
        return {}

    res = results.copy()
    for excl in EXCLUDE_DB.get(matrix, []):
        res = res[~res["database"].astype(str).str.lower().str.contains(excl, na=False)]
    if res.empty:
        return {}

    mz = _var_mz(var)
    amz = pd.to_numeric(res["mz"], errors="coerce").to_numpy()
    afdr = pd.to_numeric(res.get("fdr", pd.Series(np.nan, index=res.index)), errors="coerce").to_numpy()
    adb = res["database"].astype(str).to_numpy()
    anames = res.get("moleculeNames", pd.Series("", index=res.index)).to_numpy()
    aion = res.get("ion", pd.Series("", index=res.index)).astype(str).to_numpy()

    best: dict[int, dict] = {}
    for r in range(len(res)):
        m = amz[r]
        if not np.isfinite(m):
            continue
        d = np.abs(mz - m)
        if not np.isfinite(d).any():
            continue
        j = int(np.nanargmin(d))
        e = (mz[j] - m) / m * 1e6
        if abs(e) > ppm_tol:
            continue
        fdr = afdr[r]
        key = (1.0 if np.isnan(fdr) else float(fdr), _db_rank(adb[r], matrix), abs(e))
        cur = best.get(j)
        if cur is None or key < cur["key"]:
            best[j] = {
                "key": key,
                "name": _parse_names(anames[r]),
                "db": adb[r],
                "fdr": fdr,
                "ppm": e,
                "ion": aion[r],
            }
    return best


# --------------------------------------------------------------------------- #
# 5. Core: build the hierarchical annotation columns
# --------------------------------------------------------------------------- #

def annotate_var_table(
    var: pd.DataFrame,
    matrix: str,
    metaspace_results: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    panel_ppm_tol: float = 20.0,
    metaspace_ppm_tol: float = 10.0,
) -> pd.DataFrame:
    """Return a DataFrame (indexed like var) of annotation columns.

    Hierarchy:
      - FMP-10 sections: MS/MS panel is primary (confidence 'validated').
        METASPACE (native DBs, danielReceptorDB excluded) fills only peaks the
        panel did not claim.
      - DHB / 9-AA sections: METASPACE is primary (DB priority is matrix-aware).
      - Every candidate from every source is recorded in 'annotation_all'.
    """
    idx = var.index
    ann = pd.Series("", index=idx, dtype=object)
    src = pd.Series("", index=idx, dtype=object)
    conf = pd.Series("", index=idx, dtype=object)
    fdr = pd.Series(np.nan, index=idx, dtype=float)
    ppm = pd.Series(np.nan, index=idx, dtype=float)
    allc = pd.Series("", index=idx, dtype=object)

    def _append_all(i, text):
        allc.iat[i] = f"{allc.iat[i]} | {text}" if allc.iat[i] else text

    # ---- Tier 1: MS/MS-validated FMP panel (FMP-10 sections only) ----
    if matrix == "FMP-10" and panel is not None and len(panel):
        pname, pppm = match_panel(var, panel, panel_ppm_tol)
        for i in range(len(idx)):
            if pname.iat[i]:
                ann.iat[i] = pname.iat[i]
                src.iat[i] = "fmp_panel"
                conf.iat[i] = "validated"
                ppm.iat[i] = pppm.iat[i]
                _append_all(i, f"{pname.iat[i]} [fmp_panel/MS2]")

    # ---- Tier 2: METASPACE (primary for DHB/9-AA, complement for FMP) ----
    if metaspace_results is not None and len(metaspace_results):
        best = metaspace_best_per_peak(var, metaspace_results, matrix, metaspace_ppm_tol)
        pos = {name: i for i, name in enumerate(idx)}
        for j, info in best.items():
            _append_all(j, f"{info['ion']} ({info['name']}) [{info['db']} FDR={_fmt_fdr(info['fdr'])}]".strip())
            if ann.iat[j] == "":  # don't displace a panel call
                ann.iat[j] = info["name"]
                src.iat[j] = f"metaspace:{info['db']}"
                conf.iat[j] = _confidence_from_fdr(info["fdr"])
                fdr.iat[j] = info["fdr"]
                ppm.iat[j] = info["ppm"]

    return pd.DataFrame(
        {
            "annotation": ann,
            "annotation_source": src,
            "annotation_confidence": conf,
            "annotation_fdr": fdr,
            "annotation_ppm": ppm,
            "annotation_all": allc,
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# 6. METASPACE results loading (reuses your cached CSV convention)
# --------------------------------------------------------------------------- #

def load_metaspace_results(out_dir: Path, sample_id: str, fdr: float) -> pd.DataFrame | None:
    """Load cached METASPACE CSVs written by your annotation script.

    Filename convention: SMA_<stem>[.<dataset_id>].<db>.fdr<fdr>.csv
    """
    stem = sample_stem(sample_id)
    label = f"{fdr:g}"
    paths = sorted(
        p
        for p in out_dir.glob(f"SMA_{stem}*.fdr{label}.csv")
        if "summary" not in p.name.lower() and ".aggregated." not in p.name.lower()
    )
    if not paths:
        return None

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "database" not in df.columns:
            # derive db label from filename: SMA_<stem>[.<id>].<db>.fdr<fdr>.csv
            mid = p.name.split(".fdr")[0]
            df["database"] = mid.split(".")[-1]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if "mz" not in out.columns:
        raise ValueError(f"METASPACE CSVs for {stem} lack an 'mz' column.")
    return out


# --------------------------------------------------------------------------- #
# 7. metadata.csv helpers
# --------------------------------------------------------------------------- #

def _load_metadata(path: Path | None) -> pd.DataFrame | None:
    if path and Path(path).exists():
        return pd.read_csv(path)
    return None


def metadata_for_sample(sample_id: str, metadata: pd.DataFrame | None) -> pd.Series | None:
    """Return the single metadata.csv row for a sample, if available."""
    if metadata is None or "Sample.ID" not in metadata.columns:
        return None

    sample_ids = metadata["Sample.ID"].astype(str)
    rows = metadata.loc[sample_ids == sample_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def _metadata_obs_column(column: str) -> str:
    """Convert metadata.csv headers into stable obs column names."""
    name = re.sub(r"[^0-9A-Za-z]+", "_", str(column)).strip("_").lower()
    return f"metadata_{name or 'field'}"


def add_sample_metadata_to_modalities(mdata, sample_id: str, metadata: pd.DataFrame | None) -> list[str]:
    """Append sample-level metadata.csv values to every modality's obs."""
    row = metadata_for_sample(sample_id, metadata)
    if row is None:
        return []

    metadata_values = {
        _metadata_obs_column(col): value
        for col, value in row.items()
    }

    for adata in mdata.mod.values():
        for col, value in metadata_values.items():
            adata.obs[col] = value

    return sorted(metadata_values)


def _default_sma_root(sma_root: Path | None, metadata_csv: Path | None) -> Path | None:
    """Locate the Mendeley SMA root containing `sma/` or `sma.zip`."""
    if sma_root is not None:
        return sma_root
    if metadata_csv is not None:
        return metadata_csv.parent
    return None


def _spaceranger_sidecar_relpath(sample_id: str, filename: str) -> str:
    array_id = sample_id.rsplit("_", 1)[0]
    return (
        f"sma/{array_id}/{sample_id}/output_data/"
        f"{sample_id}_RNA/outs/{filename}"
    )


def _read_spaceranger_sidecar(
    sma_root: Path | None,
    sample_id: str,
    filename: str,
) -> pd.DataFrame | None:
    """Read a Space Ranger sidecar CSV from extracted files or sma.zip."""
    if sma_root is None:
        return None

    relpath = _spaceranger_sidecar_relpath(sample_id, filename)
    local = sma_root / relpath
    if local.exists():
        return pd.read_csv(local, index_col=0)

    zip_path = sma_root / "sma.zip"
    if not zip_path.exists():
        return None

    try:
        with zipfile.ZipFile(zip_path) as zf:
            return pd.read_csv(io.BytesIO(zf.read(relpath)), index_col=0)
    except KeyError:
        return None


def _obs_barcodes(adata) -> pd.Index:
    """Return raw Visium barcodes for modality obs rows."""
    if "barcode" in adata.obs.columns:
        return pd.Index(adata.obs["barcode"].astype(str), name="barcode")
    return pd.Index([str(obs_name).split(":", 1)[-1] for obs_name in adata.obs_names], name="barcode")


def _normalize_visium_barcode(barcode: str) -> str:
    """Strip Seurat duplicate suffixes while preserving the 10x GEM suffix."""
    return re.sub(r"(-\d+)[_.]\d+$", r"\1", str(barcode))


def _sidecar_values_for_obs(sidecar: pd.DataFrame, barcodes: pd.Index) -> pd.Series:
    """Align a Space Ranger sidecar to modality obs barcodes."""
    sidecar_values = sidecar.iloc[:, 0]
    aligned = sidecar_values.reindex(barcodes)
    missing = aligned.isna()
    if not missing.any():
        return aligned

    normalized_sidecar = sidecar_values.copy()
    normalized_sidecar.index = pd.Index(
        [_normalize_visium_barcode(barcode) for barcode in sidecar_values.index],
        name=sidecar_values.index.name,
    )
    normalized_sidecar = normalized_sidecar.loc[~normalized_sidecar.index.duplicated(keep="first")]
    normalized_barcodes = pd.Index(
        [_normalize_visium_barcode(barcode) for barcode in barcodes],
        name=barcodes.name,
    )
    normalized_aligned = normalized_sidecar.reindex(normalized_barcodes)
    normalized_aligned.index = aligned.index
    return aligned.fillna(normalized_aligned)


def _same_obs_values(existing: pd.Series, new: pd.Series) -> bool:
    return existing.astype("string").fillna("<NA>").equals(new.astype("string").fillna("<NA>"))


def _assign_obs_if_not_redundant(adata, column: str, values: pd.Series) -> str | None:
    values = pd.Series(pd.Categorical(values), index=adata.obs.index, name=column)
    if column not in adata.obs.columns:
        adata.obs[column] = values
        return column

    if adata.obs[column].isna().all():
        adata.obs[column] = values
        return column

    if _same_obs_values(adata.obs[column], values):
        return None

    source_column = f"{column}_spaceranger"
    if source_column in adata.obs.columns and _same_obs_values(adata.obs[source_column], values):
        return None

    adata.obs[source_column] = values
    return source_column


def add_spaceranger_metadata_to_modalities(mdata, sample_id: str, sma_root: Path | None) -> list[str]:
    """Append lesion/region Space Ranger sidecars to RNA/MSI obs when available."""
    modalities = [modality for modality in ("rna", "msi") if modality in mdata.mod]
    added: list[str] = []
    for column, filename in [("lesion", "lesion.csv"), ("region", "region.csv")]:
        sidecar = _read_spaceranger_sidecar(sma_root, sample_id, filename)
        if sidecar is None or sidecar.empty:
            continue

        for modality in modalities:
            adata = mdata.mod[modality]
            values = _sidecar_values_for_obs(sidecar, _obs_barcodes(adata))
            values.index = adata.obs.index
            added_column = _assign_obs_if_not_redundant(adata, column, values)
            if added_column is not None:
                added.append(f"{modality}:{added_column}")

    return added


# --------------------------------------------------------------------------- #
# 8. CLI: annotate the 'msi' modality of each aligned .h5mu in place
# --------------------------------------------------------------------------- #

def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def _load_dotenv() -> None:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.environ["SPATIALJEPA_CONFIG"])


def _default_h5mu_dir() -> Path:
    return Path(os.getenv("SPATIALJEPA_DATA", "")) / "vicari_2023" / "h5mu_export"


def _default_metaspace_dir() -> Path:
    return Path(os.getenv("SPATIALJEPA_OUT", "")) / "metaspace_output"


def parse_args(notebook: bool = False) -> argparse.Namespace:
    _load_dotenv()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # avoid Jupyter --f=<kernel.json> matching --fdr / --fmp-...
    )
    p.add_argument(
        "--h5mu-dir", type=Path, default=_default_h5mu_dir(),
        help="Directory of aligned <sample>.h5mu files (default: $SPATIALJEPA_DATA/vicari_2023/h5mu_export)",
    )
    p.add_argument(
        "--metaspace-dir", type=Path, default=_default_metaspace_dir(),
        help="Directory of cached METASPACE CSVs (default: $SPATIALJEPA_OUT/metaspace_output)",
    )
    p.add_argument("--metadata-csv", type=Path, default=None, help="SMA metadata.csv (Sample.ID, Matrix)")
    p.add_argument("--sma-root", type=Path, default=None, help="SMA Mendeley root containing sma/ or sma.zip")
    p.add_argument("--fmp-reference-h5ad", type=Path, default=None, help="Annotated FMP h5ad to source the panel")
    p.add_argument("--samples", nargs="*", default=None, help="Sample IDs (default: all .h5mu in --h5mu-dir)")
    p.add_argument("--fdr", type=float, default=0.20)
    p.add_argument("--panel-ppm", type=float, default=20.0)
    p.add_argument("--metaspace-ppm", type=float, default=10.0)
    if notebook:
        return p.parse_known_args()[0]
    return p.parse_args()


#%%
def main() -> None:
    args = parse_args(notebook=is_notebook())

    import mudata as mu  # imported here so the core stays dependency-light

    metadata = _load_metadata(args.metadata_csv)
    sma_root = _default_sma_root(args.sma_root, args.metadata_csv)
    panel = load_fmp_panel(args.fmp_reference_h5ad)

    sample_ids = args.samples or [f.stem for f in sorted(args.h5mu_dir.glob("*.h5mu"))]
    if not sample_ids:
        raise FileNotFoundError(f"No .h5mu files in {args.h5mu_dir}")

    summary = []
    for sid in sample_ids:
        matrix = matrix_for_sample(sid, metadata)
        h5mu_path = args.h5mu_dir / f"{sid}.h5mu"
        mdata = mu.read_h5mu(h5mu_path)
        metadata_cols = add_sample_metadata_to_modalities(mdata, sid, metadata)
        sidecar_cols = add_spaceranger_metadata_to_modalities(mdata, sid, sma_root)
        if "msi" not in mdata.mod:
            print(f"[skip] {sid}: no 'msi' modality")
            continue
        msi = mdata.mod["msi"]

        results = load_metaspace_results(args.metaspace_dir, sid, args.fdr)
        cols = annotate_var_table(
            msi.var, matrix, results, panel,
            panel_ppm_tol=args.panel_ppm, metaspace_ppm_tol=args.metaspace_ppm,
        )
        for c in cols.columns:
            msi.var[c] = cols[c].reindex(msi.var.index)

        n = int((cols["annotation"] != "").sum())
        n_panel = int((cols["annotation_source"] == "fmp_panel").sum())
        print(f"[ok] {sid} (matrix={matrix}): {n}/{msi.n_vars} features annotated "
              f"({n_panel} from FMP panel); "
              f"{len(metadata_cols)} metadata obs columns added; "
              f"sidecars: {', '.join(sidecar_cols) if sidecar_cols else 'none'}")
        mdata.update()
        mdata.write(h5mu_path)
        summary.append({"sample_id": sid, "matrix": matrix, "n_annotated": n, "n_panel": n_panel})

    print("\nSummary:")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()