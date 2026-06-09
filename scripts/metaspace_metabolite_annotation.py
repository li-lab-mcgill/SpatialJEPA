"""
Annotate SMA MALDI-MSI metabolites via METASPACE.

Primary path (no download, no upload): many of the SMA samples have already
been processed by the dataset authors and are public on METASPACE (named e.g.
"V11L12-038-D1.from_smamsi").  We look the sample up by name and pull its
annotation table directly — the 35-70 GB IBD is never touched.

Fallback path (only if the sample isn't already on METASPACE): stream the
imzML/IBD from Figshare into a TemporaryDirectory and submit a fresh job.  The
metaspace uploader requires a real seekable file (open()+seek()) sized via
stat(), so the IBD must briefly land on disk; the temp dir is deleted as soon
as the upload completes.
"""

from pathlib import Path
import tempfile
import time
import json
import requests
import pandas as pd
from metaspace import SMInstance


# ----------------------------
# USER SETTINGS
# ----------------------------

# Figshare article API endpoint for this dataset.
FIGSHARE_FILES_API = (
    "https://api.figshare.com/v2/articles/22770161/files"
)

# File stem shared by the .imzML and .ibd pair you want to annotate.
#
# METASPACE availability (as of 2026-06-09)
# ==========================================
# When PREFER_EXISTING_METASPACE=True the script searches METASPACE by name
# and skips the Figshare download entirely.  The table below shows which stems
# already have a FINISHED public result and which databases were used.
#
# Stem             METASPACE dataset name                         Databases
# ---------------  ---------------------------------------------  --------------------------------
# v11l12-038-b1  * V11L12-038-B1.from_smamsi                     HMDB/v4, CoreMetabolome, ChEBI, KEGG
# v11l12-038-d1  * V11L12-038-D1.from_smamsi                     HMDB/v4, CoreMetabolome, ChEBI, KEGG
#                * v11l12-038-d1.from_smamsi.lipid_anno           HMDB/v4, LipidMaps, SwissLipids
# v11l12-109-a1  * V11L12-109_A1.Visium.FMP.220826_smamsi        HMDB/v4 (FMP-10 matrix, 2023)
# v11l12-109-b1  * V11L12-109-B1.from_smamsi                     HMDB/v4, CoreMetabolome, ChEBI, KEGG
#                * V11L12-109_B1.Visium.FMP.220826_smamsi         HMDB/v4 (FMP-10 matrix, 2023)
# v11l12-109-c1  * V11L12-109_C1.Visium.FMP.220826_smamsi        HMDB/v4 (FMP-10 matrix, 2023)
#
# v11l12-038-a1    (not yet on METASPACE — fallback will download ~45 GB IBD)
# v11t17-085_a1    (not yet on METASPACE — fallback will download ~47 GB IBD)
# v11t17-085_b1    (not yet on METASPACE — fallback will download ~59 GB IBD)
# v11t17-085_c1    (not yet on METASPACE — fallback will download ~70 GB IBD)
#
# Note: V11T17-102_*.smamsi entries on METASPACE are a different sample ID
# (102, not 085) and do not correspond to any file in this Figshare article.
#
# Available Figshare stems:
#   Mouse striatum (DHB, positive): v11l12-038-a1  v11l12-038-b1  v11l12-038-d1
#                                   v11l12-109-a1  v11l12-109-b1  v11l12-109-c1
#   Human striatum:                 v11t17-085_a1  v11t17-085_b1  v11t17-085_c1
SAMPLE_NAME = "v11l12-109-b1"

DATASET_NAME = f"SMA_{SAMPLE_NAME}"

# If True (default), first look for an already-processed public METASPACE
# dataset matching SAMPLE_NAME and just pull its annotations — no Figshare
# download and no upload at all.  The SMA authors have already submitted
# several of these samples (named e.g. "V11L12-038-D1.from_smamsi").
# Only if no match is found do we fall back to downloading the imzML/IBD
# from Figshare and submitting a fresh job.
PREFER_EXISTING_METASPACE = True

# One of: "mouse_striatum" | "mouse_substantia_nigra" | "human_striatum"
SAMPLE_TYPE = "mouse_striatum"

# One of: "DHB" | "9-AA" | "norharmane_pos" | "norharmane_neg" | "FMP-10"
MATRIX_MODE = "DHB"

# Resolving power at m/z 400.  130 000 is a reasonable starting point for
# 7T Bruker FTICR data; confirm against the imzML header before publication.
DETECTOR_RESOLVING_POWER = 130000

IS_PUBLIC = False

DATABASES = [
    ("HMDB", "v4"),
    ("LipidMaps", "2017-12-12"),
]

FDR = 0.10
PPM = 3.0
NUM_ISOTOPIC_PEAKS = 4
POLL_SECONDS = 60

OUT_DIR = Path("/home/mcb/users/dmannk/BAKLAVA_base/outputs/metaspace_output")
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------
# FIGSHARE DOWNLOAD HELPERS
# ----------------------------

def figshare_download_urls(api_url: str) -> dict[str, str]:
    """Return {filename: download_url} for every file in the article."""
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    return {entry["name"]: entry["download_url"] for entry in resp.json()}


def stream_download(url: str, dest: Path, chunk_mb: int = 64) -> None:
    """Stream-download *url* to *dest*, printing progress."""
    chunk = chunk_mb * 1024 * 1024
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as fh:
            for block in r.iter_content(chunk_size=chunk):
                fh.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {dest.name}: {downloaded/1e9:.2f}/{total/1e9:.2f} GB  ({pct:.1f}%)", end="", flush=True)
        print()  # newline after progress


def fetch_pair(sample_name: str, tmpdir: Path) -> tuple[Path, Path]:
    """
    Look up the .imzML and .ibd download URLs for *sample_name* and stream
    both files into *tmpdir*.  Returns (imzml_path, ibd_path).
    """
    print("Fetching Figshare file list...")
    url_map = figshare_download_urls(FIGSHARE_FILES_API)

    imzml_name = f"{sample_name}.imzML"
    ibd_name   = f"{sample_name}.ibd"

    for fname in (imzml_name, ibd_name):
        if fname not in url_map:
            raise KeyError(
                f"'{fname}' not found in Figshare article. "
                f"Available files: {sorted(url_map)}"
            )

    imzml_path = tmpdir / imzml_name
    ibd_path   = tmpdir / ibd_name

    print(f"Downloading {imzml_name} (~{url_map[imzml_name][:60]})")
    stream_download(url_map[imzml_name], imzml_path)

    print(f"Downloading {ibd_name}  (this may take a while — file is 35-70 GB)")
    stream_download(url_map[ibd_name], ibd_path)

    return imzml_path, ibd_path


# ----------------------------
# METADATA
# ----------------------------

def sma_metadata(sample_type: str, matrix_mode: str, resolving_power: float) -> dict:
    """
    Metadata for Figshare/SMA MALDI-MSI data:
    Spatial Multimodal analysis (SMA) - MSI.
    Adjust fields if your specific file name indicates a different sample.
    """

    matrix_mode = matrix_mode.lower()

    if matrix_mode == "dhb":
        maldi_matrix = "2,5-dihydroxybenzoic acid (DHB)"
        polarity = "Positive"
        solvent = "50% acetonitrile + 0.2% trifluoroacetic acid"
        tissue_modification = "none"
        adduct_note = "Recommended positive-mode adducts: +H, +Na, +K"

    elif matrix_mode in {"9-aa", "9aa"}:
        maldi_matrix = "9-aminoacridine (9-AA)"
        polarity = "Negative"
        solvent = "80% methanol"
        tissue_modification = "none"
        adduct_note = "Recommended negative-mode adducts: -H, +Cl"

    elif matrix_mode == "norharmane_pos":
        maldi_matrix = "norharmane"
        polarity = "Positive"
        solvent = "80% methanol"
        tissue_modification = "none"
        adduct_note = "Recommended positive-mode adducts: +H, +Na, +K"

    elif matrix_mode == "norharmane_neg":
        maldi_matrix = "norharmane"
        polarity = "Negative"
        solvent = "80% methanol"
        tissue_modification = "none"
        adduct_note = "Recommended negative-mode adducts: -H, +Cl"

    elif matrix_mode in {"fmp-10", "fmp10"}:
        maldi_matrix = "FMP-10"
        polarity = "Positive"
        solvent = "70% acetonitrile"
        tissue_modification = (
            "On-tissue chemical derivatization with FMP-10 reactive matrix"
        )
        adduct_note = (
            "FMP-10 derivatization shifts observed masses; native HMDB/LipidMaps "
            "annotation may not be directly valid unless using appropriate "
            "chemical modifications or a custom derivatized database."
        )

    else:
        raise ValueError(f"Unsupported MATRIX_MODE: {matrix_mode}")

    if sample_type == "mouse_striatum":
        organism = "Mus musculus"
        organism_part = "Brain; striatum / caudoputamen"
        condition = (
            "Mouse brain; control or unilateral 6-OHDA Parkinson's disease model, "
            "depending on sample identifier"
        )
        sample_details = (
            "Adult male C57BL/6J mouse brain; fresh frozen / snap frozen; "
            "striatal-level section"
        )
        section_thickness = "12 µm"

    elif sample_type == "mouse_substantia_nigra":
        organism = "Mus musculus"
        organism_part = "Brain; substantia nigra"
        condition = "Unilateral 6-OHDA Parkinson's disease mouse model"
        sample_details = (
            "Adult male C57BL/6J mouse brain; fresh frozen / snap frozen; "
            "substantia-nigra-level section"
        )
        section_thickness = "12 µm"

    elif sample_type == "human_striatum":
        organism = "Homo sapiens"
        organism_part = "Brain; striatum / caudate-putamen / caudate nucleus"
        condition = "Parkinson's disease postmortem brain"
        sample_details = (
            "Human postmortem Parkinson's disease brain sample; fresh frozen / "
            "snap frozen"
        )
        section_thickness = "10 µm"

    else:
        raise ValueError(f"Unsupported SAMPLE_TYPE: {sample_type}")

    return {
        "Data_Type": "Imaging MS",

        "Sample_Information": {
            "Organism": organism,
            "Organism_Part": organism_part,
            "Condition": condition,
            "Sample_Growth_Conditions": sample_details,
        },

        "Sample_Preparation": {
            "Sample_Stabilisation": (
                f"Fresh frozen / snap frozen; stored frozen; cryosectioned at "
                f"{section_thickness}"
            ),
            "Tissue_Modification": tissue_modification,
            "MALDI_Matrix": maldi_matrix,
            "MALDI_Matrix_Application": "TM-Sprayer robotic sprayer; HTX Technologies",
            "Solvent": solvent,
        },

        "MS_Analysis": {
            "Polarity": polarity,
            "Ionisation_Source": "MALDI",
            "Analyzer": "FTICR",
            "Detector_Resolving_Power": {
                "mz": 400,
                "Resolving_Power": resolving_power,
            },
            "Pixel_Size": {
                "Xaxis": 100,
                "Yaxis": 100,
            },
        },

        "Additional_Information": {
            "Supplementary": (
                "Source: Spatial Multimodal Analysis (SMA) MALDI-MSI dataset. "
                "Use this metadata as a starting point and verify against the "
                "specific imzML header / instrument method before publication. "
                + adduct_note
            )
        },
    }


def adducts_for_matrix(matrix_mode: str) -> list[str]:
    matrix_mode = matrix_mode.lower()

    if matrix_mode in {"dhb", "norharmane_pos", "fmp-10", "fmp10"}:
        return ["+H", "+Na", "+K"]

    if matrix_mode in {"9-aa", "9aa", "norharmane_neg"}:
        return ["-H", "+Cl"]

    raise ValueError(f"Unsupported MATRIX_MODE: {matrix_mode}")


# ----------------------------
# VALIDATION
# ----------------------------

def find_processed_dataset(sm: SMInstance, sample_name: str):
    """
    Look for an already-processed, FINISHED METASPACE dataset whose name
    matches *sample_name*.  Returns the dataset object, or None.

    Figshare stems like "v11l12-109-a1" appear on METASPACE inconsistently:
    some keep the dash before the section letter ("V11L12-038-B1.from_smamsi"),
    others replace it with an underscore ("V11L12-109_A1.Visium...").
    Searching by the full stem therefore misses the underscore variants.

    Strategy: search by the plate prefix (everything up to the last '-'),
    then filter candidates whose name contains the section identifier with
    either separator.
    """
    # Split "v11l12-109-a1" into plate prefix "v11l12-109" and section "a1".
    parts = sample_name.lower().rsplit("-", 1)
    plate_prefix, section = (parts[0], parts[1]) if len(parts) == 2 else (sample_name.lower(), "")

    candidates = sm.datasets(nameMask=plate_prefix)
    finished = [d for d in candidates if getattr(d, "status", None) == "FINISHED"]

    # Keep only datasets whose name contains the section with '-' or '_'.
    if section:
        finished = [
            d for d in finished
            if f"-{section}" in d.name.lower() or f"_{section}" in d.name.lower()
        ]

    if not finished:
        return None

    # Prefer "from_smamsi" datasets (submitted by the authors from the raw data)
    # over older Visium re-runs, then fall back to most recently submitted.
    preferred = [d for d in finished if "from_smamsi" in d.name.lower()]
    chosen = (preferred or finished)[0]

    print(f"Found {len(finished)} processed match(es) for {sample_name!r}:")
    for d in finished:
        marker = " <-- using" if d is chosen else ""
        print(f"    {d.id} | {d.name}{marker}")
    return chosen


def export_results(ds, dataset_id: str) -> None:
    """Pull annotations for every database and write them to OUT_DIR."""
    print(f"Exporting annotations from {ds.name} ({dataset_id})...")

    # For an existing public dataset, export every database it was processed
    # against — the DATABASES setting only controls what a *new* submission
    # requests, so filtering by it here would silently drop e.g. CoreMetabolome
    # or KEGG that the authors used but aren't in DATABASES.
    available = [(d.name, d.version) for d in getattr(ds, "database_details", [])]
    wanted = available or DATABASES

    exported = []
    for db in wanted:
        try:
            results = ds.results(database=db, fdr=FDR).reset_index()
        except Exception as e:
            print(f"  skip {db}: {type(e).__name__}: {e}")
            continue

        db_label = "_".join(map(str, db)).replace(" ", "_").replace("/", "_")
        out_csv = OUT_DIR / f"{DATASET_NAME}.{db_label}.fdr{FDR}.csv"
        results.to_csv(out_csv, index=False)

        print(f"  {db}: {len(results):,} annotations at FDR <= {FDR} -> {out_csv}")
        exported.append({
            "dataset_id": dataset_id,
            "database": str(db),
            "fdr": FDR,
            "n_annotations": len(results),
            "csv": str(out_csv),
        })

    summary = pd.DataFrame(exported)
    summary_csv = OUT_DIR / f"{DATASET_NAME}.summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("Done.")
    print(summary)


def submit_from_figshare(sm: SMInstance) -> str:
    """
    Fallback path: stream the imzML/IBD from Figshare into a temp dir, submit
    to METASPACE, then delete the temp files.  Returns the new dataset_id.
    """
    # TemporaryDirectory is deleted automatically when the context exits,
    # even if an exception is raised.
    with tempfile.TemporaryDirectory(prefix="sma_msi_") as _tmpdir:
        tmpdir = Path(_tmpdir)

        imzml_path, ibd_path = fetch_pair(SAMPLE_NAME, tmpdir)
        print(f"imzML: {imzml_path}  ({imzml_path.stat().st_size / 1e6:.1f} MB)")
        print(f"IBD:   {ibd_path}  ({ibd_path.stat().st_size / 1e9:.2f} GB)")

        metadata = sma_metadata(
            sample_type=SAMPLE_TYPE,
            matrix_mode=MATRIX_MODE,
            resolving_power=DETECTOR_RESOLVING_POWER,
        )
        adducts = adducts_for_matrix(MATRIX_MODE)

        metadata_json = OUT_DIR / f"{DATASET_NAME}.metadata.json"
        metadata_json.write_text(json.dumps(metadata, indent=2))
        print(f"Wrote metadata: {metadata_json}")

        print("Submitting dataset to METASPACE...")
        dataset_id = sm.submit_dataset(
            imzml_fn=str(imzml_path),
            ibd_fn=str(ibd_path),
            name=DATASET_NAME,
            metadata=metadata,
            is_public=IS_PUBLIC,
            databases=DATABASES,
            adducts=adducts,
            ppm=PPM,
            num_isotopic_peaks=NUM_ISOTOPIC_PEAKS,
            description=(
                "Programmatic METASPACE annotation of SMA MALDI-MSI imzML/IBD data."
            ),
        )
        print(f"Submitted dataset_id: {dataset_id}")

    # TemporaryDirectory (and the 35-70 GB IBD) is deleted here.
    print("Temporary files cleaned up.")

    # Poll until annotation finishes.
    while True:
        ds = sm.dataset(id=dataset_id)
        print(f"Status: {ds.status}")
        if ds.status == "FINISHED":
            break
        if ds.status == "FAILED":
            raise RuntimeError(f"METASPACE annotation failed: {dataset_id}")
        time.sleep(POLL_SECONDS)

    return dataset_id


def main():
    sm = SMInstance()

    # First run only: prompts for your METASPACE API key and stores it.
    sm.save_login()

    # ---- Preferred path: no download, no upload ----
    if PREFER_EXISTING_METASPACE:
        ds = find_processed_dataset(sm, SAMPLE_NAME)
        if ds is not None:
            export_results(ds, ds.id)
            return
        print(
            f"No processed METASPACE dataset found for {SAMPLE_NAME!r}; "
            f"falling back to Figshare download + submission."
        )

    # ---- Fallback path: download from Figshare and submit ----
    dataset_id = submit_from_figshare(sm)
    ds = sm.dataset(id=dataset_id)
    print("Annotation finished.")
    export_results(ds, dataset_id)


if __name__ == "__main__":
    main()
