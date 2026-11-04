try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%%
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
import s3fs
import tangram as tg
import decoupler as dc
import tqdm
from scipy.sparse import csr_matrix

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _BAKLAVA_BASE_DIR = Path(
        os.environ.get("SPATIALJEPA_HOME", _SCRIPT_DIR.parent)
    ).resolve()
except:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.environ["SPATIALJEPA_CONFIG"], override=True)
    _BAKLAVA_BASE_DIR = os.getenv("SPATIALJEPA_HOME")

def ensure_group_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o2775)
    except PermissionError:
        # Non-owner collaborators can write via group perms but cannot chmod.
        pass


#%% configuration

ACTIVE_REGION = os.environ.get("SEA_AD_ACTIVE_REGION", "cah").lower()
SEA_AD_REGION = "us-west-2"  # SEA-AD public buckets, anonymous access

DATA_ROOT = Path(__import__("os").environ["SPATIALJEPA_DATA"])
SEA_AD_DATA_ROOT = Path(
    os.environ.get("SEA_AD_DATA_ROOT", DATA_ROOT / "SEA_AD")
).resolve()
OUTPUT_ROOT = Path(__import__("os").environ["SPATIALJEPA_OUT"]) / "sea_ad_lipid_gps_minimal"
os.umask(0o002)
ensure_group_writable_dir(OUTPUT_ROOT)

SEA_AD_QNP_WORKBOOK = (
    DATA_ROOT
    / "SEA_AD"
    / "6a4579d1d67150b0e69ad044_sea-ad-quantitative-neuropathology-063026.xlsx"
)

# Set e.g. SEA_AD_MAX_DONORS=1 for a quick smoke test of the pipeline.
MAX_DONORS = int(os.environ.get("SEA_AD_MAX_DONORS", "0")) or None
MIN_FREE_GB_HEADROOM = 20  # abort before downloading if this little free space remains

SEA_AD_DATASETS = {
    "mtg": {
        "label": "MTG",
        "cache_dir": SEA_AD_DATA_ROOT / "sea_ad_mtg",
        "output_dir": OUTPUT_ROOT / "mtg",
        "single_cell_dataset": "seaad_final",
        "section_key": "Section",
        "section_label": None,
        "spatial_group_key": "Layer annotation",
        "normalize_spatial_group": True,
        "s3": {
            "single_cell": {
                "rna_final_h5ad": (
                    "sea-ad-single-cell-profiling/MTG/RNAseq/"
                    "SEAAD_MTG_RNAseq_final-nuclei.2026-06-22.h5ad"
                ),
                "rna_all_h5ad": (
                    "sea-ad-single-cell-profiling/MTG/RNAseq/"
                    "SEAAD_MTG_RNAseq_all-nuclei.2026-06-22.h5ad"
                ),
            },
            "spatial": {
                "merfish_h5ad": (
                    "sea-ad-spatial-transcriptomics/middle-temporal-gyrus/"
                    "all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad"
                ),
            },
        },
    },
    "cah": {
        "label": "CaH",
        "cache_dir": SEA_AD_DATA_ROOT / "sea_ad_cah",
        "output_dir": OUTPUT_ROOT / "cah",
        "single_cell_dataset": "seaad_final",
        "section_key": "Specimen_ID",
        "section_label": "all_cells",
        "spatial_group_key": "Neighborhood",
        "normalize_spatial_group": False,
        "s3": {
            "single_cell": {
                "rna_all_h5ad": (
                    "sea-ad-single-cell-profiling/Caudate_Nucleus/"
                    "CaH_all_nuclei.2025-12-29.h5ad"
                ),
                "rna_final_h5ad": (
                    "sea-ad-single-cell-profiling/Caudate_Nucleus/"
                    "CaH_final_nuclei.2025-12-29.h5ad"
                ),
            },
            "spatial": {
                "merfish_h5ad": (
                    "sea-ad-spatial-transcriptomics/caudate_nucleus/"
                    "combined_anndata_object/CaH_Xenium_final.2026-01-07.h5ad"
                ),
            },
        },
    },
}
if ACTIVE_REGION not in SEA_AD_DATASETS:
    raise ValueError(
        f"Unknown SEA_AD_ACTIVE_REGION={ACTIVE_REGION!r}. "
        f"Expected one of {sorted(SEA_AD_DATASETS)}."
    )

DATASET_CONFIG = SEA_AD_DATASETS[ACTIVE_REGION]
DATASET_LABEL = DATASET_CONFIG["label"]
CACHE_DIR = DATASET_CONFIG["cache_dir"]
REGION_OUTPUT_ROOT = DATASET_CONFIG["output_dir"]
ensure_group_writable_dir(CACHE_DIR)
ensure_group_writable_dir(REGION_OUTPUT_ROOT)
ACTIVE_S3 = DATASET_CONFIG["s3"]

DONOR_KEY = "Donor ID"
SECTION_KEY = DATASET_CONFIG.get("section_key")
SECTION_LABEL = DATASET_CONFIG.get("section_label") or "all_cells"
SPATIAL_GROUP_KEY = DATASET_CONFIG["spatial_group_key"]

CAH_FIRST_RESULTS_MIN_DONORS = int(os.environ.get("CAH_FIRST_RESULTS_MIN_DONORS", "6"))
CAH_QNP_SHEET_SPECS = [
    {
        "sheet": "CaH",
        "prefix": "cah",
        "filter_column": "brain region",
        "filter_value": "CaH",
    },
    {
        "sheet": "Kana 2026 (CaH)",
        "prefix": "kana",
        "filter_column": None,
        "filter_value": None,
    },
    {
        "sheet": "Travaglini 2026 (multi-region)",
        "prefix": "travaglini",
        "filter_column": "brain region",
        "filter_value": "CaH",
    },
]


DEVICE = "cuda:2"
MIN_CELLS = 2  # tangram needs >=1 cell & >=1 spot; require a couple for a meaningful fit

# --- spatialMETA hand-off config -------------------------------------------------
ENABLE_SPATIALMETA_EMBEDDING = True
SPATIALMETA_ENV_PYTHON = Path(
    os.environ.get(
        "SPATIALMETA_ENV_PYTHON",
        __import__("os").environ['PY_MSI'],
    )
)
SPATIALMETA_MODEL_SAMPLE = os.environ.get(
    "SPATIALMETA_MODEL_SAMPLE",
    os.environ.get("GCP_VM_MODEL_SAMPLE", "V11T17-102"),
)
# encode_nuclei.py imports spatialmeta_backbone.py and spatialmeta.util._classes
# from this directory. Default to the local BAKLAVA checkout's SMA scripts dir instead
# of the GCP VM path (a remote checkout) baked into that worker.
BAKLAVAS_SMA_DIR = Path(
    os.environ.get(
        "BAKLAVAS_SMA_DIR",
        _BAKLAVA_BASE_DIR / "BAKLAVA" / "scripts" / "SMA",
    )
).expanduser().resolve()
SPATIALMETA_OUTPATH = Path(
    os.environ.get("SPATIALJEPA_OUT", _BAKLAVA_BASE_DIR / "outputs")
).resolve()


def resolve_spatialmeta_model_dir() -> Path:
    """Resolve the SpatialJEPA checkpoint directory for the hand-off worker.

  Priority:
    1. SPATIALMETA_MODEL_DIR (explicit override)
    2. $SPATIALJEPA_OUT/spatialjepa_models/$SPATIALMETA_MODEL_SAMPLE (local training output)
    """
    if override := os.environ.get("SPATIALMETA_MODEL_DIR"):
        return Path(override).expanduser().resolve()
    return SPATIALMETA_OUTPATH / "spatialjepa_models" / SPATIALMETA_MODEL_SAMPLE


def assert_spatialmeta_model_bundle(model_dir: Path) -> None:
    """Fail early with a actionable message when the model bundle is incomplete."""
    required = [
        "joint_adata.h5ad",
        "teacher.pt",
        "student.pt",
        "student_pls_decoder.joblib",
    ]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete SpatialJEPA model bundle at "
            f"{model_dir}. Missing: {missing}. "
            "Train with rna_msi/train_rna_msi.py or copy "
            "an existing bundle into $SPATIALJEPA_OUT/spatialjepa_models/<RUN_ID>/ . "
            "To push a local bundle to the GCP VM: "
            "bash gcs_scripts/gcp_connect.sh scp-models <RUN_ID>"
        )


def assert_baklavas_sma_dir(sma_dir: Path) -> None:
    """Fail early if the SMA scripts dir (spatialmeta_backbone.py, spatialmeta package) is missing."""
    required = ["spatialmeta_backbone.py"]
    missing = [name for name in required if not (sma_dir / name).is_file()]
    if missing or not sma_dir.is_dir():
        raise FileNotFoundError(
            f"BAKLAVAS_SMA_DIR={sma_dir} is missing or incomplete (missing: {missing}). "
            "Set BAKLAVAS_SMA_DIR to the local BAKLAVA/scripts/SMA checkout, e.g. "
            f"{_BAKLAVA_BASE_DIR / 'BAKLAVA' / 'scripts' / 'SMA'}."
        )


SPATIALMETA_MODEL_DIR = resolve_spatialmeta_model_dir()
SPATIALMETA_WORKER = _SCRIPT_DIR / "encode_nuclei.py"
SPATIALMETA_HANDOFF_ROOT = REGION_OUTPUT_ROOT / "spatialmeta_handoff"
SPATIALMETA_MIN_FREE_GB = 50
SPATIALMETA_STAGE_HEADROOM_GB = 20
SPATIALMETA_STALE_HOURS = 24
SPATIALMETA_MIN_MATCH_FRACTION = 0.10
SPATIALMETA_MIN_MATCH_FRACTION_BY_KIND = {
    "snrna": 0.10,
    # SEA-AD MERFISH/Xenium is a targeted panel, so it naturally overlaps only a small
    # fraction of the SpatialJEPA ST reference. Keep this nonzero to catch completely
    # mismatched features while allowing panel-sized overlap.
    "merfish": 0.01,
}
SPATIALMETA_DEVICE = os.environ.get("SPATIALMETA_DEVICE", DEVICE)
SPATIALMETA_BATCH_SIZE = int(os.environ.get("SPATIALMETA_BATCH_SIZE", "32"))
SPATIALMETA_KEEP_FAILED_INPUTS = False
SPATIALMETA_USE_CACHE = True
SPATIALMETA_FORCE_RECOMPUTE = os.environ.get("SPATIALMETA_FORCE_RECOMPUTE", "0") == "1"
SPATIALMETA_MSI_DECODER = "pls"  # "neural" or "pls"
SPATIALMETA_PLS_DECODER_PATH = SPATIALMETA_MODEL_DIR / "student_pls_decoder.joblib"
SPATIALMETA_TEACHER_OBSM = "X_spatialmeta_teacher"
SPATIALMETA_STUDENT_OBSM = "X_spatialmeta_student"
SPATIALMETA_TEACHER_MSI_OBSM = "X_spatialmeta_teacher_msi"
SPATIALMETA_STUDENT_MSI_OBSM = "X_spatialmeta_student_msi"
SPATIALMETA_ALIGNED_MUDATA_PATH = REGION_OUTPUT_ROOT / "snrna_spatialmeta_aligned.h5mu"


#%% dataset helpers

def normalize_layer_annotation(adata, column: str = "Layer annotation") -> None:
    """Treat blank layer annotations as WM while preserving categorical dtype."""
    layer_ann = adata.obs[column]
    if pd.api.types.is_categorical_dtype(layer_ann):
        if "" in layer_ann.cat.categories:
            new_categories = ["WM" if x == "" else x for x in layer_ann.cat.categories]
            adata.obs[column] = layer_ann.cat.rename_categories(new_categories)
    else:
        adata.obs[column] = layer_ann.replace({"": "WM"})


def final_h5ad_path(path: Path) -> Path:
    """Return the local final-file name used for filtered SEA-AD AnnData caches."""
    name = path.name
    if "_final" in name or "final-" in name:
        return path
    if name.startswith("CaH_all_nuclei."):
        return path.with_name(name.replace("CaH_all_nuclei.", "CaH_final_nuclei.", 1))
    if name.startswith("CaH_Xenium."):
        return path.with_name(name.replace("CaH_Xenium.", "CaH_Xenium_final.", 1))
    return path.with_name(path.stem + "_final.h5ad")


def run_pseudobulk(adata, *, sample_col: str, groups_col: str, mode: str, skip_checks: bool = False):
    """Run pseudobulk across decoupler versions.

    Some decoupler installs do not expose the preprocessing namespace as `dc.pp`,
    even though `decoupler.pp.pseudobulk` is installed.
    """
    try:
        pseudobulk = __import__("decoupler.pp", fromlist=["pseudobulk"]).pseudobulk

        return pseudobulk(
            adata=adata,
            sample_col=sample_col,
            groups_col=groups_col,
            mode=mode,
            skip_checks=skip_checks,
        )
    except ImportError:
        return dc.get_pseudobulk(
            adata=adata,
            sample_col=sample_col,
            groups_col=groups_col,
            mode=mode,
            skip_checks=skip_checks,
            min_cells=1,
            min_counts=1,
            remove_empty=False,
        )


def filter_used_in_analysis(adata, label: str):
    """Return only cells retained by SEA-AD's original `Used in analysis` flag."""
    column = "Used in analysis"
    if column not in adata.obs:
        raise KeyError(f"{label} AnnData for {DATASET_LABEL} lacks {column!r} column.")
    used = adata.obs[column]
    n_missing = int(used.isna().sum())
    if n_missing:
        raise AssertionError(f"{label} AnnData has {n_missing:,} missing {column!r} values.")
    n_false = int((~used.astype(bool)).sum())
    if not n_false:
        print(f"{label}: all {adata.n_obs:,} cells have {column!r}=True.")
        return adata
    print(
        f"{label}: filtering {n_false:,}/{adata.n_obs:,} cells with "
        f"{column!r}=False."
    )
    mask = used.astype(bool).to_numpy()
    if not adata.isbacked:
        return adata[mask].copy()

    # Backed AnnData views cannot be indexed a second time (donor/section subsets are
    # sliced out downstream), so materialize the filtered subset to its own backed h5ad
    # file once, then reopen it as a fresh (non-view) backed object.
    import anndata as ad

    source_path = Path(adata.filename)
    filtered_path = final_h5ad_path(source_path)
    if not filtered_path.exists():
        kept_fraction = mask.sum() / len(mask)
        assert_disk_headroom(
            int(source_path.stat().st_size * kept_fraction),
            context=f"writing filtered {label} backed copy",
        )
        print(f"{label}: writing filtered backed copy -> {filtered_path}")
        adata[mask].copy(filename=filtered_path)
    return ad.read_h5ad(filtered_path, backed="r")


def get_sea_ad_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": SEA_AD_REGION})


def cache_path(s3_key: str, cache_dir: Path) -> Path:
    return cache_dir / Path(s3_key).name


def assert_disk_headroom(estimated_bytes: int, *, context: str) -> None:
    usage = shutil.disk_usage(SEA_AD_DATA_ROOT)
    required = estimated_bytes + MIN_FREE_GB_HEADROOM * 1024**3
    if usage.free < required:
        raise RuntimeError(
            f"Insufficient disk space for {context}. "
            f"Free={usage.free / 1024**3:.1f}G, required={required / 1024**3:.1f}G."
        )


def download_from_s3(s3_key: str, *, cache_dir: Path, fs: s3fs.S3FileSystem) -> Path:
    """Download an S3 object to cache_dir if not already present."""
    local = cache_path(s3_key, cache_dir)
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        size = fs.info(s3_key)["size"]
        assert_disk_headroom(size, context=f"downloading {s3_key}")
        print(f"Downloading s3://{s3_key} ({size / 1e9:.2f} GB) -> {local}")
        fs.get(s3_key, str(local))
    return local


def remove_h5ad_null_encodings(path: Path) -> None:
    """Remove AnnData null-encoded HDF5 nodes that older readers cannot load.

    SEA-AD CaH h5ad files include entries such as /uns/log1p/base encoded as
    IOSpec("null", "0.1.0"). anndata 0.11.x cannot read these in backed mode,
    while deleting them is equivalent to restoring the Python value None for
    optional metadata like log1p base.
    """
    import h5py

    null_paths: list[str] = []

    def visit(name: str, obj) -> None:
        encoding_type = obj.attrs.get("encoding-type")
        if isinstance(encoding_type, bytes):
            encoding_type = encoding_type.decode()
        if encoding_type == "null":
            null_paths.append(name)

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)

    if not null_paths:
        return

    print(f"Removing {len(null_paths)} AnnData null-encoded entrie(s) from {path}:")
    with h5py.File(path, "r+") as handle:
        for name in sorted(null_paths, reverse=True):
            print(f"  /{name}")
            parent_name, key = name.rsplit("/", 1) if "/" in name else ("", name)
            parent = handle[parent_name] if parent_name else handle
            del parent[key]


def load_single_cell(fs: s3fs.S3FileSystem):
    """Download (if needed) and open the active region's snRNA-seq AnnData, backed mode."""
    import anndata as ad

    key_by_dataset = {
        "seaad_final": ACTIVE_S3["single_cell"].get("rna_final_h5ad"),
        "seaad_all": ACTIVE_S3["single_cell"].get("rna_all_h5ad"),
    }
    s3_key = key_by_dataset[DATASET_CONFIG["single_cell_dataset"]]
    local = download_from_s3(s3_key, cache_dir=CACHE_DIR, fs=fs)
    remove_h5ad_null_encodings(local)
    return ad.read_h5ad(local, backed="r")


def load_spatial_transcriptomics(fs: s3fs.S3FileSystem):
    """Download (if needed) and open the active region's spatial AnnData, backed mode."""
    import anndata as ad

    s3_key = ACTIVE_S3["spatial"]["merfish_h5ad"]
    local = download_from_s3(s3_key, cache_dir=CACHE_DIR, fs=fs)
    remove_h5ad_null_encodings(local)
    return ad.read_h5ad(local, backed="r")


#%% spatialMETA hand-off helpers
# Stages an AnnData to disk, runs `encode_nuclei.py` as a subprocess inside the
# `spatialMETA` conda env (kept separate because it pins an older anndata/torch stack for its
# SpatialJEPA teacher/student models), then reads back teacher/student embeddings plus decoded
# MSI features via NPZ/JSON. Nothing here imports spatialMETA's own dependencies directly.

def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def safe_path_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def spatialmeta_handoff_dirs() -> dict[str, Path]:
    dirs = {
        "root": SPATIALMETA_HANDOFF_ROOT,
        "outputs": SPATIALMETA_HANDOFF_ROOT / "outputs",
        "manifests": SPATIALMETA_HANDOFF_ROOT / "manifests",
        "staging": SPATIALMETA_HANDOFF_ROOT / "staging",
    }
    for path in dirs.values():
        ensure_group_writable_dir(path)
    return dirs


def estimate_adata_nbytes(adata) -> int:
    """Rough in-memory payload estimate for disk-headroom checks before staging."""
    from scipy.sparse import issparse

    x = adata.X
    if issparse(x):
        x_nbytes = x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
    else:
        x_nbytes = np.asarray(x).nbytes
    # obs/var strings and HDF5 overhead are hard to estimate; add a conservative cushion.
    metadata_cushion = (adata.n_obs + adata.n_vars) * 2048
    return int(x_nbytes + metadata_cushion)


def assert_spatialmeta_disk_headroom(
    *,
    estimated_write_nbytes: int = 0,
    context: str = "spatialMETA staging",
) -> None:
    SPATIALMETA_HANDOFF_ROOT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(SPATIALMETA_HANDOFF_ROOT)
    min_free = SPATIALMETA_MIN_FREE_GB * 1024**3
    required_for_write = estimated_write_nbytes + SPATIALMETA_STAGE_HEADROOM_GB * 1024**3
    required = max(min_free, required_for_write)
    if usage.free < required:
        raise RuntimeError(
            "Insufficient disk for spatialMETA staging "
            f"({context}). Free={usage.free / 1024**3:.1f}G, "
            f"required={required / 1024**3:.1f}G."
        )


def cleanup_spatialmeta_staging(*, failed_only: bool = False) -> None:
    if not ENABLE_SPATIALMETA_EMBEDDING:
        return
    dirs = spatialmeta_handoff_dirs()
    cmd = [
        str(SPATIALMETA_ENV_PYTHON),
        str(SPATIALMETA_WORKER),
        "--model-dir",
        str(SPATIALMETA_MODEL_DIR),
        "--handoff-root",
        str(dirs["root"]),
        "--older-than-hours",
        str(SPATIALMETA_STALE_HOURS),
        "--cleanup-failed" if failed_only else "--cleanup-staging",
    ]
    run_spatialmeta_command(cmd, log_stem="cleanup_failed" if failed_only else "cleanup_staging")


def spatialmeta_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")
    env["BAKLAVAS_SMA_DIR"] = str(BAKLAVAS_SMA_DIR)
    return env


def run_spatialmeta_command(cmd: list[str], *, log_stem: str) -> subprocess.CompletedProcess:
    dirs = spatialmeta_handoff_dirs()
    log_prefix = dirs["manifests"] / f"{utc_run_id()}__{safe_path_token(log_stem)}"
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=spatialmeta_env(),
    )
    (log_prefix.with_suffix(".stdout.log")).write_text(result.stdout, encoding="utf-8")
    (log_prefix.with_suffix(".stderr.log")).write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "spatialMETA worker failed. "
            f"See {log_prefix.with_suffix('.stderr.log')} and "
            f"{log_prefix.with_suffix('.stdout.log')}."
        )
    return result


_SPATIALMETA_SMOKE_TESTED = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spatialmeta_requested_active_msi_decoder() -> tuple[str, str]:
    requested = SPATIALMETA_MSI_DECODER.lower()
    if requested not in {"neural", "pls"}:
        raise ValueError("SPATIALMETA_MSI_DECODER must be either 'neural' or 'pls'.")
    active = "neural" if requested == "pls" and not SPATIALMETA_PLS_DECODER_PATH.exists() else requested
    return requested, active


def spatialmeta_expected_decoder_metadata() -> dict[str, str]:
    requested, active = spatialmeta_requested_active_msi_decoder()
    metadata = {"requested": requested, "active": active}
    if requested == "pls":
        metadata["pls_decoder_path"] = str(SPATIALMETA_PLS_DECODER_PATH)
        if active == "pls":
            metadata["pls_decoder_sha256"] = sha256_file(SPATIALMETA_PLS_DECODER_PATH)
        else:
            metadata["fallback_reason"] = "missing_pls_decoder"
    return metadata


def spatialmeta_cache_decoder_matches(
    metadata: dict,
    *,
    expected_decoder: dict[str, str],
) -> bool:
    decoder = metadata.get("msi_decoder")
    if not isinstance(decoder, dict):
        return False
    if (
        decoder.get("requested") != expected_decoder.get("requested")
        or decoder.get("active") != expected_decoder.get("active")
    ):
        return False
    if expected_decoder.get("active") == "pls":
        return decoder.get("pls_decoder_sha256") == expected_decoder.get("pls_decoder_sha256")
    return decoder.get("fallback_reason") == expected_decoder.get("fallback_reason")


def ensure_spatialmeta_ready() -> None:
    global _SPATIALMETA_SMOKE_TESTED
    if not ENABLE_SPATIALMETA_EMBEDDING or _SPATIALMETA_SMOKE_TESTED:
        return
    expected_decoder = spatialmeta_expected_decoder_metadata()
    if not SPATIALMETA_ENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing spatialMETA env Python: {SPATIALMETA_ENV_PYTHON}")
    if not SPATIALMETA_WORKER.exists():
        raise FileNotFoundError(f"Missing spatialMETA worker: {SPATIALMETA_WORKER}")
    if not SPATIALMETA_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Missing spatialMETA model dir: {SPATIALMETA_MODEL_DIR}. "
            f"Expected $SPATIALJEPA_OUT/spatialjepa_models/{SPATIALMETA_MODEL_SAMPLE}/ "
            "after running train_rna_msi.py, or set SPATIALMETA_MODEL_DIR."
        )
    assert_spatialmeta_model_bundle(SPATIALMETA_MODEL_DIR)
    assert_baklavas_sma_dir(BAKLAVAS_SMA_DIR)
    assert_spatialmeta_disk_headroom(context="spatialMETA batch startup")
    cleanup_spatialmeta_staging()
    cmd = [
        str(SPATIALMETA_ENV_PYTHON),
        str(SPATIALMETA_WORKER),
        "--model-dir",
        str(SPATIALMETA_MODEL_DIR),
        "--device",
        SPATIALMETA_DEVICE,
        "--batch-size",
        str(SPATIALMETA_BATCH_SIZE),
        "--msi-decoder",
        expected_decoder["requested"],
        "--pls-decoder-path",
        str(SPATIALMETA_PLS_DECODER_PATH),
        "--smoke-test",
        "--quiet",
    ]
    run_spatialmeta_command(cmd, log_stem="smoke_test")
    _SPATIALMETA_SMOKE_TESTED = True


def spatialmeta_output_paths(
    *,
    kind: Literal["snrna", "merfish"],
    donor_id: str,
    section_id: str | None = None,
) -> tuple[Path, Path]:
    dirs = spatialmeta_handoff_dirs()
    donor = safe_path_token(donor_id)
    if kind == "snrna":
        base = dirs["outputs"] / "snrna" / donor
    else:
        if section_id is None:
            raise ValueError("section_id is required for MERFISH spatialMETA outputs")
        base = dirs["outputs"] / "merfish" / donor / safe_path_token(section_id)
    return base / "embeddings.npz", base / "embeddings.json"


def spatialmeta_aligned_output_paths(donor_id: str) -> tuple[Path, Path]:
    """Return per-donor aligned RNA and imputed MSI cache paths."""
    npz_path, _ = spatialmeta_output_paths(kind="snrna", donor_id=donor_id)
    base = npz_path.parent
    return base / "aligned_rna.h5ad", base / "imputed_msi.h5ad"


def attach_spatialmeta_embeddings(
    adata,
    *,
    npz_path: Path,
    json_path: Path,
    kind: str,
    expected_decoder: dict[str, str],
) -> bool:
    if not npz_path.exists() or not json_path.exists():
        return False
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not spatialmeta_cache_decoder_matches(
            metadata,
            expected_decoder=expected_decoder,
        ):
            print(
                "spatialMETA cache decoder metadata does not match current request, "
                f"regenerating: {npz_path}"
            )
            return False
        try:
            payload = np.load(npz_path, allow_pickle=False)
            obs_names = payload["obs_names"].astype(str)
        except ValueError as exc:
            if "Object arrays cannot be loaded" not in str(exc):
                raise
            print(f"Reading legacy object-dtype obs_names cache with pickle: {npz_path}")
            payload = np.load(npz_path, allow_pickle=True)
            obs_names = payload["obs_names"].astype(str)
        required = {
            "teacher",
            "student",
            "teacher_msi",
            "student_msi",
            "msi_feature_names",
        }
        if not required.issubset(set(payload.files)):
            print(f"spatialMETA cache is missing decoded MSI keys, regenerating: {npz_path}")
            return False
        expected = adata.obs_names.astype(str).to_numpy()
        if not np.array_equal(obs_names, expected):
            return False
        teacher = np.asarray(payload["teacher"], dtype=np.float32)
        student = np.asarray(payload["student"], dtype=np.float32)
        teacher_msi = np.asarray(payload["teacher_msi"], dtype=np.float32)
        student_msi = np.asarray(payload["student_msi"], dtype=np.float32)
        msi_feature_names = payload["msi_feature_names"].astype(str)
        if (
            teacher.shape[0] != adata.n_obs
            or student.shape[0] != adata.n_obs
            or teacher_msi.shape[0] != adata.n_obs
            or student_msi.shape[0] != adata.n_obs
        ):
            return False
        adata.obsm[SPATIALMETA_TEACHER_OBSM] = teacher
        adata.obsm[SPATIALMETA_STUDENT_OBSM] = student
        adata.obsm[SPATIALMETA_TEACHER_MSI_OBSM] = teacher_msi
        adata.obsm[SPATIALMETA_STUDENT_MSI_OBSM] = student_msi
        metadata["msi_feature_names"] = msi_feature_names.tolist()
        adata.uns[f"spatialmeta_{kind}"] = metadata
        return True
    except Exception as exc:  # noqa: BLE001 - invalid caches should be regenerated
        print(f"Ignoring invalid spatialMETA cache {npz_path}: {exc}")
        return False


def spatialmeta_staging_adata(adata):
    """Build a worker-safe AnnData for spatialMETA staging.

    The spatialMETA env uses anndata 0.8.x, which cannot read newer null-encoded
    fields (for example uns metadata with JSON nulls). Strip prior spatialMETA
    artifacts and stage only the expression/coordinate payload the worker needs.
    """
    import anndata as ad

    obs = adata.obs.copy()
    var = adata.var.copy()
    obsm = {}
    if "spatial" in adata.obsm:
        obsm["spatial"] = np.asarray(adata.obsm["spatial"])
    return ad.AnnData(X=adata.X, obs=obs, var=var, obsm=obsm)


def stage_spatialmeta_input(adata, *, stage_dir: Path, file_stem: str) -> Path:
    estimated = estimate_adata_nbytes(adata)
    assert_spatialmeta_disk_headroom(
        estimated_write_nbytes=estimated,
        context=f"{file_stem} ({estimated / 1024**3:.2f}G estimated)",
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"{safe_path_token(file_stem)}.h5ad"
    spatialmeta_staging_adata(adata).write_h5ad(path)
    return path

def safe_feature_token(value: object) -> str:
    """Return a compact ASCII token for column names."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return token or "na"
    
def ensure_spatialmeta_embeddings(items: list[dict]) -> None:
    """Attach embeddings, optionally using cached NPZ/JSON outputs for exact obs matches."""
    if not ENABLE_SPATIALMETA_EMBEDDING:
        return
    ensure_spatialmeta_ready()
    expected_decoder = spatialmeta_expected_decoder_metadata()
    dirs = spatialmeta_handoff_dirs()
    run_id = utc_run_id()
    stage_dir = dirs["staging"] / run_id
    manifest_path = dirs["manifests"] / f"{run_id}__batch.json"
    summary_path = dirs["manifests"] / f"{run_id}__batch_summary.json"

    pending_jobs = []
    staged_paths: list[Path] = []
    try:
        for item in items:
            npz_path, json_path = spatialmeta_output_paths(
                kind=item["kind"],
                donor_id=item["donor_id"],
                section_id=item.get("section_id"),
            )
            aligned_rna_path = None
            imputed_msi_path = None
            if item["kind"] == "snrna":
                aligned_rna_path, imputed_msi_path = spatialmeta_aligned_output_paths(
                    item["donor_id"]
                )
            aligned_outputs_exist = (
                item["kind"] != "snrna"
                or (
                    aligned_rna_path.exists()
                    and imputed_msi_path.exists()
                )
            )
            if (
                SPATIALMETA_USE_CACHE
                and not SPATIALMETA_FORCE_RECOMPUTE
                and aligned_outputs_exist
                and attach_spatialmeta_embeddings(
                    item["adata"],
                    npz_path=npz_path,
                    json_path=json_path,
                    kind=item["kind"],
                    expected_decoder=expected_decoder,
                )
            ):
                continue

            file_stem = "__".join(
                filter(
                    None,
                    [
                        item["kind"],
                        item["donor_id"],
                        item.get("section_id"),
                    ],
                )
            )
            input_path = stage_spatialmeta_input(
                item["adata"],
                stage_dir=stage_dir,
                file_stem=file_stem,
            )
            staged_paths.append(input_path)
            pending_jobs.append(
                {
                    "kind": item["kind"],
                    "donor_id": item["donor_id"],
                    "section_id": item.get("section_id"),
                    "input_h5ad": str(input_path),
                    "output_npz": str(npz_path),
                    "output_json": str(json_path),
                    "output_aligned_rna_h5ad": (
                        str(aligned_rna_path) if aligned_rna_path is not None else None
                    ),
                    "output_imputed_msi_h5ad": (
                        str(imputed_msi_path) if imputed_msi_path is not None else None
                    ),
                    "batch_keys": ["section"],
                    "batch_value": item.get("section_id"),
                    "min_match_fraction": SPATIALMETA_MIN_MATCH_FRACTION_BY_KIND.get(
                        item["kind"],
                        SPATIALMETA_MIN_MATCH_FRACTION,
                    ),
                    "force_recompute": (
                        SPATIALMETA_FORCE_RECOMPUTE or not SPATIALMETA_USE_CACHE
                        or npz_path.exists() or json_path.exists()
                    ),
                    "delete_input_on_success": True,
                }
            )

        if pending_jobs:
            manifest = {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model_dir": str(SPATIALMETA_MODEL_DIR),
                "summary_json": str(summary_path),
                "jobs": pending_jobs,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            cmd = [
                str(SPATIALMETA_ENV_PYTHON),
                str(SPATIALMETA_WORKER),
                "--model-dir",
                str(SPATIALMETA_MODEL_DIR),
                "--device",
                SPATIALMETA_DEVICE,
                "--batch-size",
                str(SPATIALMETA_BATCH_SIZE),
                "--min-match-fraction",
                str(SPATIALMETA_MIN_MATCH_FRACTION),
                "--msi-decoder",
                expected_decoder["requested"],
                "--pls-decoder-path",
                str(SPATIALMETA_PLS_DECODER_PATH),
                "--manifest",
                str(manifest_path),
                "--quiet",
            ]
            if SPATIALMETA_KEEP_FAILED_INPUTS:
                cmd.append("--keep-failed-inputs")
            if SPATIALMETA_FORCE_RECOMPUTE:
                cmd.append("--force-recompute")
            run_spatialmeta_command(cmd, log_stem="batch")

        for item in items:
            npz_path, json_path = spatialmeta_output_paths(
                kind=item["kind"],
                donor_id=item["donor_id"],
                section_id=item.get("section_id"),
            )
            if not attach_spatialmeta_embeddings(
                item["adata"],
                npz_path=npz_path,
                json_path=json_path,
                kind=item["kind"],
                expected_decoder=expected_decoder,
            ):
                raise RuntimeError(
                    f"spatialMETA embeddings were not produced or did not validate for "
                    f"{item['kind']} {item['donor_id']} {item.get('section_id', '')}."
                )
    except Exception:
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / ".failed").touch()
        if not SPATIALMETA_KEEP_FAILED_INPUTS:
            for path in staged_paths:
                path.unlink(missing_ok=True)
        raise
    finally:
        if stage_dir.exists() and not SPATIALMETA_KEEP_FAILED_INPUTS:
            shutil.rmtree(stage_dir, ignore_errors=True)


def write_aligned_snrna_mudata(
    donor_ids: list[str],
    *,
    output_path: Path,
) -> Path:
    """Combine per-donor aligned RNA and imputed MSI into one on-disk MuData."""
    import anndata as ad
    import mudata as md

    if not donor_ids:
        raise ValueError("Cannot write aligned snRNA MuData without donors.")

    rna_parts = []
    msi_parts = []
    for donor_id in donor_ids:
        rna_path, msi_path = spatialmeta_aligned_output_paths(donor_id)
        if not rna_path.exists() or not msi_path.exists():
            raise FileNotFoundError(
                f"Missing aligned spatialMETA output for donor {donor_id}: "
                f"rna={rna_path.exists()}, msi={msi_path.exists()}."
            )
        rna_parts.append(rna_path)
        msi_parts.append(msi_path)

    ensure_group_writable_dir(output_path.parent)
    concat_dir = output_path.parent / ".snrna_spatialmeta_concat"
    ensure_group_writable_dir(concat_dir)
    combined_rna_path = concat_dir / "aligned_rna.h5ad"
    combined_msi_path = concat_dir / "imputed_msi.h5ad"
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    for path in (combined_rna_path, combined_msi_path, tmp_output_path):
        path.unlink(missing_ok=True)

    print(f"Combining {len(donor_ids)} donor RNA partitions on disk...")
    ad.experimental.concat_on_disk(rna_parts, combined_rna_path, axis=0, join="inner")
    print(f"Combining {len(donor_ids)} donor MSI partitions on disk...")
    ad.experimental.concat_on_disk(msi_parts, combined_msi_path, axis=0, join="inner")

    # PCA/neighbors/UMAP need an in-memory RNA matrix; keep MSI backed until write.
    print("Loading combined RNA for PCA/neighbors/UMAP...")
    rna = ad.read_h5ad(combined_rna_path)
    msi = ad.read_h5ad(combined_msi_path, backed="r")
    try:
        if not rna.obs_names.equals(msi.obs_names):
            raise ValueError("Combined aligned RNA and imputed MSI observation order differs.")
        if "X_umap" not in rna.obsm:
            print(
                f"Computing RNA PCA/neighbors/UMAP on {rna.n_obs:,} cells × "
                f"{rna.n_vars:,} genes..."
            )
            sc.pp.pca(rna, n_comps=50)
            sc.pp.neighbors(rna, n_neighbors=15, n_pcs=50)
            sc.tl.umap(rna)
        mudata = md.MuData({"rna": rna, "msi": msi})
        mudata.uns["spatialmeta_device"] = SPATIALMETA_DEVICE
        mudata.uns["spatialmeta_batch_size"] = SPATIALMETA_BATCH_SIZE
        mudata.uns["donor_ids"] = donor_ids
        mudata.write_h5mu(tmp_output_path)
        os.replace(tmp_output_path, output_path)
    finally:
        if getattr(msi, "isbacked", False) and getattr(msi, "file", None) is not None:
            msi.file.close()
        tmp_output_path.unlink(missing_ok=True)

    combined_rna_path.unlink(missing_ok=True)
    combined_msi_path.unlink(missing_ok=True)
    try:
        concat_dir.rmdir()
    except OSError:
        pass
    print(f"Wrote aligned all-donor snRNA/MSI MuData: {output_path}")
    return output_path


def load_cah_quantitative_neuropathology_workbook(
    workbook_path: Path = SEA_AD_QNP_WORKBOOK,
    *,
    min_donors: int = CAH_FIRST_RESULTS_MIN_DONORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load donor-level CaH quantitative neuropathology from the SEA-AD workbook."""
    if not workbook_path.exists():
        print(f"CaH QNP workbook not found: {workbook_path}")
        return pd.DataFrame(), pd.DataFrame()

    qnp_frames = []
    source_records = []
    for spec in CAH_QNP_SHEET_SPECS:
        sheet = spec["sheet"]
        try:
            df = pd.read_excel(workbook_path, sheet_name=sheet)
        except ValueError:
            print(f"CaH QNP workbook is missing sheet {sheet!r}; skipping.")
            continue
        if DONOR_KEY not in df:
            print(f"CaH QNP sheet {sheet!r} has no {DONOR_KEY!r}; skipping.")
            continue

        filter_column = spec.get("filter_column")
        filter_value = spec.get("filter_value")
        if filter_column is not None and filter_column in df:
            df = df[df[filter_column].astype(str).eq(str(filter_value))].copy()

        df[DONOR_KEY] = df[DONOR_KEY].astype(str)
        df = df.drop_duplicates(subset=[DONOR_KEY]).set_index(DONOR_KEY)
        numeric = df.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.loc[:, numeric.notna().any(axis=0)]
        if numeric.empty:
            continue

        renamed = {}
        for col in numeric.columns:
            col_token = safe_feature_token(col)
            renamed[col] = f"{spec['prefix']}__{col_token}"
            source_records.append(
                {
                    "outcome": renamed[col],
                    "sheet": sheet,
                    "original_column": col,
                    "n_nonmissing_sheet": int(numeric[col].notna().sum()),
                }
            )
        qnp_frames.append(numeric.rename(columns=renamed))

    if not qnp_frames:
        return pd.DataFrame(), pd.DataFrame(source_records)

    qnp = pd.concat(qnp_frames, axis=1)
    qnp = qnp.loc[:, ~qnp.columns.duplicated()]
    keep = (qnp.notna().sum(axis=0) >= min_donors) & (qnp.nunique(dropna=True) > 1)
    qnp = qnp.loc[:, keep].sort_index()
    qnp.index.name = DONOR_KEY

    deduped_cols = []
    duplicate_of: dict[str, str] = {}
    seen_signatures: dict[tuple, str] = {}
    for col in qnp.columns:
        signature = tuple(
            "__NA__" if pd.isna(value) else round(float(value), 12)
            for value in qnp[col].to_numpy()
        )
        if signature in seen_signatures:
            duplicate_of[col] = seen_signatures[signature]
        else:
            seen_signatures[signature] = col
            deduped_cols.append(col)
    if duplicate_of:
        print(
            f"CaH QNP workbook: dropping {len(duplicate_of)} duplicate outcome columns "
            "after merging sheets."
        )
        qnp = qnp.loc[:, deduped_cols]

    source = pd.DataFrame(source_records)
    if not source.empty:
        source = source[source["outcome"].isin(qnp.columns)].reset_index(drop=True)
    print(
        f"CaH QNP workbook: n_donors = {qnp.shape[0]}, "
        f"n_outcomes = {qnp.shape[1]}, path = {workbook_path}"
    )
    return qnp, source


#%% load SEA-AD modalities

fs = get_sea_ad_fs()
print(f"Active SEA-AD region: {DATASET_LABEL} ({ACTIVE_REGION})")

scrna = load_single_cell(fs)
print("snRNA-seq AnnData:", scrna)

#%% Run GPU inference one donor at a time to bound host and GPU memory.
all_donor_ids = sorted(scrna.obs[DONOR_KEY].dropna().astype(str).unique())
donor_ids = all_donor_ids
if MAX_DONORS is not None:
    donor_ids = donor_ids[:MAX_DONORS]
print(
    f"spatialMETA snRNA inference: {len(donor_ids)} donor(s), "
    f"device={SPATIALMETA_DEVICE}, batch_size={SPATIALMETA_BATCH_SIZE}"
)

scrna_donor_ids = scrna.obs[DONOR_KEY].astype(str)
for donor_id in tqdm.tqdm(donor_ids, desc="spatialMETA snRNA donors"):
    donor_mask = scrna_donor_ids.eq(donor_id).to_numpy()
    scrna_donor = scrna[donor_mask].to_memory()
    print(f"spatialMETA snRNA donor {donor_id}: {scrna_donor.n_obs:,} nuclei")
    ensure_spatialmeta_embeddings(
        [
            {
                "kind": "snrna",
                "adata": scrna_donor,
                "donor_id": donor_id,
                "section_id": None,
            },
        ]
    )
    del scrna_donor

mudata_output_path = SPATIALMETA_ALIGNED_MUDATA_PATH
if len(donor_ids) != len(all_donor_ids):
    mudata_output_path = mudata_output_path.with_name(
        f"{mudata_output_path.stem}.first_{len(donor_ids)}_donors"
        f"{mudata_output_path.suffix}"
    )
aligned_snrna_mudata_path = write_aligned_snrna_mudata(
    donor_ids,
    output_path=mudata_output_path,
)

# %%
qnp, qnp_source = load_cah_quantitative_neuropathology_workbook(min_donors=CAH_FIRST_RESULTS_MIN_DONORS)
print(qnp)
print(qnp_source)

# %%