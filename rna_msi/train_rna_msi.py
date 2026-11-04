try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% load libraries
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.environ["SPATIALJEPA_CONFIG"])

import json
import subprocess
from pathlib import Path
import os
import numpy as np
import scipy.sparse as sp
import spatialmeta as smt
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import muon as mu

import sys
sys.path.insert(0, os.environ["SPATIALJEPA_RNA_MSI"])
from load_aligned_mudata import load_sample

BAKLAVA_BASE = Path(os.getenv("SPATIALJEPA_HOME"))
DATA_DIR = BAKLAVA_BASE / "data" / "spatialmeta_tutorial"
JOINT_RAW_PATH = DATA_DIR / "Y7_T_adata_joint_raw.h5ad"
JOINT_HVF_PATH = DATA_DIR / "Y7_T_adata_joint_hvf2800.h5ad"
JOINT_SPATIALMETA_PATH = DATA_DIR / "Y7_T_adata_joint_hvf2800_spatialmeta.h5ad"
ZENODO_JOINT_RAW_URL = (
    "https://zenodo.org/records/14986870/files/adata_joint_Y7_T_raw.h5ad?download=1"
)

FIG_DIR = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/sea_ad_lipid_gps")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_joint_adata(path: Path = JOINT_RAW_PATH):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading tutorial data to {path}")
        subprocess.run(
            ["curl", "-L", "-o", str(path), ZENODO_JOINT_RAW_URL],
            check=True,
        )
    return smt.util._classes.AnnDataJointSMST(sc.read_h5ad(path))

def _assert_identical(a, b, path="uns['spatial']"):
    assert type(a) is type(b), f"{path}: type mismatch ({type(a)} vs {type(b)})"
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for key in a:
            _assert_identical(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, f"{path}: values differ"


import sys
sys.path.insert(0, os.environ["SPATIALJEPA_RNA_MSI"])
from spatialjepa.spatialmeta_backbone import (
    spatialJEPA_model,
    save_spatialjepa_model,
    copy_decoder_weights,
    SpatialJEPA_trainer,
    _scheduled_adamw_epochs,
)
from feature_panel import (
    load_target_panel,
    restrict_st_to_target_panel,
    save_target_panel,
)

def _flatten_axes(plot_output):
    if plot_output is None:
        return []
    if isinstance(plot_output, dict):
        return [ax for item in plot_output.values() for ax in _flatten_axes(item)]
    if isinstance(plot_output, np.ndarray):
        return [ax for item in plot_output.flat for ax in _flatten_axes(item)]
    if isinstance(plot_output, (list, tuple)):
        return [ax for item in plot_output for ax in _flatten_axes(item)]
    return [plot_output]


def _section_slug(section_id: str) -> str:
    """V11T17-102_A1 -> A1 (short section label for figure filenames)."""
    return section_id.rsplit("_", 1)[-1] if "_" in section_id else section_id


def _figure_path(filename: str) -> Path:
    """Resolve plot output under the active sample id/common-root directory."""
    path = Path(filename)
    if not path.is_absolute():
        path = FIG_DIR / RUN_ID / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_plot(plot_output, filename: str) -> None:
    axes = _flatten_axes(plot_output)
    figures = []
    for ax in axes:
        fig = ax.figure
        if fig not in figures:
            figures.append(fig)
    if not figures:
        figures = [plt.gcf()]

    path = _figure_path(filename)
    for i, fig in enumerate(figures):
        fig_path = path if len(figures) == 1 else path.with_name(f"{path.stem}_{i + 1}{path.suffix}")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] wrote {fig_path}")


def _save_labeled_plots(labeled_outputs: list[tuple[str, object]], filename: str) -> None:
    """Save one figure per (section_label, plot_output) pair, e.g. *_A1.png, *_C1.png."""
    path = _figure_path(filename)
    stem, suffix = path.stem, path.suffix
    for label, plot_output in labeled_outputs:
        axes = _flatten_axes(plot_output)
        fig = axes[0].figure if axes else plt.gcf()
        fig_path = path.with_name(f"{stem}_{label}{suffix}")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] wrote {fig_path}")


#%% load data

## sample notes
# V11T17-102_B1 probably has a large fold that increases dopamine and related genes

# sample_ids may be a single sample ID (str) for vertical-only integration, or a
# list of sample IDs that share a prefix (e.g. all "V11T17-102_*") to additionally
# enable horizontal (multi-section) integration of the same sample/donor.

#sample_ids = "V11L12-109_B1"
sample_ids = ["V11T17-102_A1", "V11T17-102_C1", "V11T17-102_D1"]

# Optional env overrides for scripted reruns (unset = the values in this file):
#   SPATIALJEPA_SAMPLE_IDS            comma-separated sample IDs
#   SPATIALJEPA_RUN_TAG               suffix for RUN_ID, so model/figure dirs don't clobber
#   SPATIALJEPA_N_TARGET_TOP          target-panel size (442 reproduces the Fig. 5 model)
#   SPATIALJEPA_TEACHER_NEIGHBORHOOD  "exact" | "induced"
#   SPATIALJEPA_SEED                  seed torch/numpy/random before model construction
if os.getenv("SPATIALJEPA_SAMPLE_IDS"):
    sample_ids = [s.strip() for s in os.environ["SPATIALJEPA_SAMPLE_IDS"].split(",") if s.strip()]
    sample_ids = sample_ids[0] if len(sample_ids) == 1 else sample_ids
RUN_TAG = os.getenv("SPATIALJEPA_RUN_TAG", "")
SEED = int(os.environ["SPATIALJEPA_SEED"]) if os.getenv("SPATIALJEPA_SEED") else None

SAMPLE_IDS = [sample_ids] if isinstance(sample_ids, str) else list(sample_ids)
SECTION_KEY = "section"
MULTI = len(SAMPLE_IDS) > 1

# All sections must come from the same sample/donor (shared prefix)
PREFIX = SAMPLE_IDS[0].split("_")[0]
assert all(s.split("_")[0] == PREFIX for s in SAMPLE_IDS), (
    f"All sample IDs must share a prefix for horizontal integration; got {SAMPLE_IDS}"
)
RUN_ID = (SAMPLE_IDS[0] if not MULTI else PREFIX) + RUN_TAG
species = "human" if PREFIX == "V11T17-102" else "mouse"

METADATA_PATH = Path(os.path.join(os.getenv("SPATIALJEPA_HOME"), "data", "vicari_2023", "mendeley_sma", "metadata.csv"))
metadata = pd.read_csv(METADATA_PATH)
sample_metadata = metadata.loc[metadata["Sample.ID"].isin(SAMPLE_IDS)]
print(sample_metadata.loc[~sample_metadata['Data.Type'].eq('RNA'), ['Sample.ID', 'sample', 'Matrix', 'Data.Type']].set_index('Sample.ID'))

H5MU_EXPORT_DIR = Path(os.path.join(os.getenv("SPATIALJEPA_HOME"), "data", "vicari_2023", "h5mu_export"))
OUTPUT_DIR = Path(os.getenv("SPATIALJEPA_OUT"))
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / RUN_ID
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def assemble_section(sample_id):
    """Load one section and return a per-section ST+SM AnnData tagged with its section id."""
    # Prefer SCT-transformed h5mu if available
    sct_path = H5MU_EXPORT_DIR / f"{sample_id}_SCT.h5mu"
    if sct_path.exists():
        joint_mudata = mu.read_h5mu(sct_path)
        print(f"[{sample_id}] Using SCT-transformed h5mu")
    else:
        joint_mudata = load_sample(sample_id, export_dir=H5MU_EXPORT_DIR)
    mu.pp.intersect_obs(joint_mudata)

    rna = joint_mudata.mod["rna"].copy()
    msi = joint_mudata.mod["msi"].copy()

    # If SCT was applied, rna.X holds scale.data; stash it as a layer, then reset
    # rna.X to raw counts for joint assembly.
    if "SCT_data" in rna.layers:
        rna.layers["SCT_scale"] = rna.X.copy()  # SCT scale.data (z-scored), ST-only
    if "counts" in rna.layers:
        rna.X = rna.layers["counts"].copy()

    # Make feature names unique and modality-prefixed
    annotations = msi.var["annotation"].astype("string")
    has_annotation = annotations.notna() & annotations.str.strip().ne("")
    feature_ids = msi.var["feature_id"].astype("string") if "feature_id" in msi.var.columns else msi.var.index.astype("string")
    msi_var_names = annotations.where(has_annotation, feature_ids)
    msi.var_names = ["msi:" + str(v) for v in msi_var_names]
    rna.var_names = ["rna:" + str(v) for v in rna.var_names]

    if not msi.var_names.is_unique:
        msi = msi[:, ~msi.var_names.duplicated()]
        print(f"[{sample_id}] Removed duplicate MSI feature names!")
    assert msi.var_names.is_unique, "MSI feature (var) names are not unique!"

    # Concatenate features (modalities) into one AnnData
    adata = sc.concat(
        {"ST": rna, "SM": msi},
        axis=1,
        join="inner",
        label="type",
        merge="same",
    )
    # carry spot coordinates explicitly (per-section pixel coords)
    adata.obsm["spatial"] = rna.obsm["spatial"]

    rna_spatial = joint_mudata.mod["rna"].uns["spatial"]
    msi_spatial = joint_mudata.mod["msi"].uns["spatial"]
    _assert_identical(rna_spatial, msi_spatial)

    # Re-key the spatial dict under the section id so keys stay unique across sections
    spatial_uns = dict(rna_spatial)
    if len(spatial_uns) == 1:
        spatial_uns = {sample_id: next(iter(spatial_uns.values()))}
    adata.uns["spatial"] = spatial_uns

    adata.var = adata.var.merge(msi.var[['annotation']], left_on='feature_id', right_index=True, how='left')
    adata.obs[SECTION_KEY] = sample_id

    # Per-modality concat with merge="same" drops any obs column whose dtype/values differ
    # between modalities -- e.g. SCT rewrites the RNA cluster labels to int32 while the
    # untouched MSI side stays int64, so the shared WNN labels get dropped. Carry the
    # columns downstream scripts (3/6) need directly from the aligned rna.obs instead.
    label_cols = ("RNA_clusters", "MSI_clusters", "MM_clusters", "lesion", "region")
    for col in label_cols:
        if col in rna.obs.columns:
            adata.obs[col] = rna.obs[col].values

    # SCT layers only cover the RNA modality and are dropped by sc.concat(merge="same").
    # Rebuild them as joint [ST | SM] matrices, keeping raw MSI counts in the SM columns.
    # SCT_scale (z-scored scale.data) is ST-only; its SM columns are a raw-MSI placeholder.
    msi_raw = msi.X
    for sct_layer in ("SCT_counts", "SCT_data", "SCT_scale"):
        if sct_layer in rna.layers:
            rna_mat = rna.layers[sct_layer]
            if sp.issparse(rna_mat) and sp.issparse(msi_raw):
                adata.layers[sct_layer] = sp.hstack([rna_mat, msi_raw], format="csr")
            else:
                rna_arr = rna_mat.toarray() if sp.issparse(rna_mat) else np.asarray(rna_mat)
                msi_arr = msi_raw.toarray() if sp.issparse(msi_raw) else np.asarray(msi_raw)
                adata.layers[sct_layer] = np.hstack([rna_arr, msi_arr])

    return adata


sections = [assemble_section(s) for s in SAMPLE_IDS]
if MULTI:
    # concat spots across sections; intersect features, keep shared var columns
    adata = sc.concat(sections, axis=0, join="inner", merge="same")
    merged_spatial = {}
    for sec in sections:
        merged_spatial.update(sec.uns["spatial"])
    adata.uns["spatial"] = merged_spatial
else:
    adata = sections[0]
adata.obs[SECTION_KEY] = adata.obs[SECTION_KEY].astype("category")

joint_adata = smt.util._classes.AnnDataJointSMST(adata)

# %% identify spatially highly variable genes and metabolites

if species == "human":
    joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata) # remove HSP, MT, RPL, DNAJ features in human
else:
    joint_adata = smt.pp.removeHsp_mt_Rpl_Dnaj(joint_adata) # remove Hsp, mt, Rpl, Dnaj features in mouse

joint_adata.layers["counts"] = joint_adata.X.copy()  # raw counts (rna.X was reset above)

if "SCT_data" in joint_adata.layers:
    # SCT_data is already log-normalized; use it for ST and normalize SM from raw counts
    joint_adata.X = joint_adata.layers["SCT_data"]
    smt.pp.normalize_total_joint_adata_sm_st(
        joint_adata,
        target_sum_SM=1e4,
        target_sum_ST=None,  # ST already log-normalized by SCT
    )
else:
    smt.pp.normalize_total_joint_adata_sm_st(
        joint_adata,
        target_sum_SM=1e4,
        target_sum_ST=1e4,
    )

joint_adata.layers["normalized"] = joint_adata.X.copy()
joint_adata.raw = joint_adata

# Restrict ST features to a ranked target panel (if provided) before Moran's-I selection,
# so the candidate ST pool only contains genes the transfer target measures. Whitelisted
# target genes sit at the top of the ranking, so they are always in the candidate pool.
N_TARGET_TOP = int(os.getenv("SPATIALJEPA_N_TARGET_TOP", 2000)) # 442 for whitelist Xenium genes
if species == "human":
    TARGET_PANEL_PATH = Path(
        f"{os.environ['SPATIALJEPA_HOME']}/outputs/target_gene_rankings/"
        "target_gene_ranking_CaH_Xenium_final.2026-01-07_protein_coding.csv"
    )
elif species == "mouse":
    TARGET_PANEL_PATH = Path(
        f"{os.environ['SPATIALJEPA_HOME']}/outputs/target_gene_rankings/"
        "target_gene_ranking_p22_mouse_spatial_atac_rna.csv"
    )
    if not TARGET_PANEL_PATH.exists():
        p22_source = Path(os.getenv("SPATIALJEPA_DATA")) / "aligned_data" / "source_rna_aligned_SCT.h5ad"
        p22_mouse_rna = sc.read_h5ad(p22_source, backed="r")
        hvg_var = p22_mouse_rna.var.loc[p22_mouse_rna.var["highly_variable"]].copy()
        rank_col = "dispersions_norm" if "dispersions_norm" in hvg_var.columns else None
        if rank_col is not None:
            hvg_var = hvg_var.sort_values(rank_col, ascending=False)
        target_panel_genes = hvg_var.index
        print(f"[panel] mouse target panel: {len(target_panel_genes)} HVGs from {p22_source.name}"
              + (f", ranked by {rank_col}" if rank_col else ""))
        save_target_panel(
            target_panel_genes,
            TARGET_PANEL_PATH,
            source=str(p22_source),
        )
        print(f"[panel] wrote mouse target panel to {TARGET_PANEL_PATH}")
else:
    TARGET_PANEL_PATH = None

def prepare_hvf_joint_adata(joint_adata, *, target_panel_path=None, variant_label="panel"):
    """Apply optional panel restriction, Moran HVF selection, and final model scaling."""
    joint_adata.uns["preprocessing_variant"] = variant_label
    joint_adata.uns["target_gene_panel_source"] = (
        target_panel_path.name if target_panel_path else "nan"
    )
    joint_adata.uns["target_gene_panel_applied"] = bool(target_panel_path)

    if target_panel_path:
        joint_adata, panel_diag = restrict_st_to_target_panel(
            joint_adata,
            load_target_panel(target_panel_path),
            n_target_top=N_TARGET_TOP,
        )
        print(f"[{variant_label}] restricted ST to target panel: {panel_diag}")
    else:
        print(f"[{variant_label}] keeping all ST genes before Moran's-I selection")

    # When integrating multiple sections, the batch_key branch also removes features whose abundance differs strongly *between* sections (assumed technical batch effects). That
    # filter cannot tell a batch effect from genuine cross-section biology: e.g. Dopamine
    # (and anything co-depleted with it) varies across sections by lesion extent (log2FC ~= 19.7 between A1 and C1) and gets dropped at the default min_logfc=3. Use a very lenient threshold so only the most extreme present/absent artifacts (log2FC > 25) are removed, retaining the dopamine-correlated biological axis.
    smt.pp.spatial_variable_joint_adata_sm_st(
        joint_adata,
        n_top_genes=2000,
        n_top_metabolites=800,
        add_key="highly_variable_moranI",
        batch_key=SECTION_KEY if MULTI else None,
        min_frac=0.8,
        min_logfc=25,
    )

    joint_adata = joint_adata[:, joint_adata.var.highly_variable_moranI].copy()

    # Drop spots with zero RNA library over the retained features. The ZINB ST decoder scales px_rna_scale by lib_size = X_ST.sum(1); a zero-library spot forces logits = log(mu/theta) = log(0) = -inf -> NaN loss on the first forward (independent of learning rate). C1 is the only human sample with none of these.
    st_mask = joint_adata.var["type"].eq("ST").to_numpy()
    st_lib = np.asarray(joint_adata[:, st_mask].X.sum(1)).ravel()
    n_empty_rna = int((st_lib == 0).sum())
    if n_empty_rna:
        print(f"[{variant_label}] dropping {n_empty_rna} spots with zero RNA library over HVF genes")
        joint_adata = joint_adata[st_lib > 0].copy()

    #joint_adata = sc.read_h5ad(JOINT_HVF_PATH)
    if "SCT_counts" in joint_adata.layers:
        # SCT corrected counts for ST; raw MSI counts in SM columns
        joint_adata.X = joint_adata.layers["SCT_counts"]
    else:
        joint_adata.X = joint_adata.layers["counts"]

    smt.pp.normalize_total_joint_adata_sm_st( # again, now on the spatially variable features
        joint_adata,
        target_sum_SM=1e3,
        target_sum_ST=None,  # ZINB decoder uses lib_size = X_ST.sum(1) internally
    )

    sm_mask = joint_adata.var["type"].eq("SM").to_numpy()
    joint_adata.X[:, sm_mask] = np.log1p(joint_adata.X[:, sm_mask])

    # Optional: preserves zeros, unlike zero_center=True
    sm = joint_adata[:, sm_mask].copy()
    sc.pp.scale(sm, zero_center=False, max_value=10)
    joint_adata.X[:, sm_mask] = sm.X

    print(
        f"[{variant_label}] {joint_adata.n_obs} spots, {joint_adata.n_vars} features "
        f"({joint_adata.var['type'].eq('ST').sum()} ST)"
    )
    return joint_adata


# Build a no-panel artifact from the same canonical pre-panel object used for training.
# This replaces the self-contained rebuild previously kept in dopamine_nmf_signature.py.
#
# NOTE on the 'normalized' layer carried into this artifact: when assemble_section()
# above picks up an *_SCT.h5mu file (SCT preprocessing applied), 'normalized' is built
# from SCT_data (already log-normalized), not from raw counts + total-count
# normalization. The best-performing NMF/dopamine bivariate Moran's I run found so far
# (max ~0.50) used raw-counts + total-counts normalization -> 'normalized', from before
# SCT was introduced into assemble_section(). That exact feature is no longer
# reproducible from this SCT-based artifact; dopamine_nmf_signature.py uses
# 'SCT_counts' (SCT-corrected, non-negative counts) as the closest available substitute
# for the no-panel KL-NMF analysis.
joint_adata_no_panel = prepare_hvf_joint_adata(
    joint_adata.copy(),
    target_panel_path=None,
    variant_label="no-panel",
)
joint_adata_no_panel_path = MODEL_DIR / "joint_adata_no_panel.h5ad"
joint_adata_no_panel.write_h5ad(joint_adata_no_panel_path)
print(f"[INFO] Saved no-panel joint_adata to {joint_adata_no_panel_path.resolve()}")
del joint_adata_no_panel

# Continue the modelling path with the target-panel restriction in place.
joint_adata = prepare_hvf_joint_adata(
    joint_adata,
    target_panel_path=TARGET_PANEL_PATH,
    variant_label="panel" if TARGET_PANEL_PATH else "standard",
)

#%% differential expression/abundance analysis, comparing intact vs lesioned striatum
if species == "human":
    # create artificial lesion mask, using a per-section Dopamine threshold
    x_dopamine = joint_adata[:, 'msi:Dopamine'].X.toarray().flatten()
    thresh_dopamine = {
        "V11T17-102_A1": 1.5,
        "V11T17-102_B1": 0.6,
        "V11T17-102_C1": 1.675,
        "V11T17-102_D1": 1.,
    }
    thresh_vec = joint_adata.obs[SECTION_KEY].map(thresh_dopamine).to_numpy(dtype=float)
    for s in SAMPLE_IDS:
        m = (joint_adata.obs[SECTION_KEY] == s).to_numpy()
        plt.figure(figsize=[3,2]); plt.hist(x_dopamine[m], bins=50); plt.axvline(thresh_dopamine[s], color='r'); plt.xlabel(f'Dopamine ({s})')
        _save_plot(None, f"{s}_dopamine_threshold.png")

    joint_adata.obs['lesion'] = pd.Series(x_dopamine < thresh_vec, index=joint_adata.obs_names).map({True: 'lesioned', False: 'intact'})
    striatum_adata = joint_adata.copy()

elif species == "mouse":
    striatum_mask = joint_adata.obs["region"].eq("striatum")
    striatum_adata = joint_adata[striatum_mask].copy()

striatum_adata.X = striatum_adata.layers["normalized"]

# reference="intact" reports lesioned vs intact; reference="lesioned" reports intact vs lesioned
sc.tl.rank_genes_groups(
    striatum_adata, groupby="lesion", reference="intact", method="wilcoxon", key_added="lesioned_vs_intact"
)
sc.tl.rank_genes_groups(
    striatum_adata, groupby="lesion", reference="lesioned", method="wilcoxon", key_added="intact_vs_lesioned"
)

_save_plot(
    sc.pl.rank_genes_groups(striatum_adata, key="lesioned_vs_intact", n_genes=10, show=False),
    f"{RUN_ID}_rank_genes_lesioned_vs_intact.png",
)
_save_plot(
    sc.pl.rank_genes_groups(striatum_adata, key="intact_vs_lesioned", n_genes=10, show=False),
    f"{RUN_ID}_rank_genes_intact_vs_lesioned.png",
)

#%% plot marker features

MARKER_GENES = [
    "Pcp4", "Tac1",          # V11L12-109_B1
    "Psap", "Sort1", "Snca", # V11L12-038_D1
    "Gba2", "Ugcg",          # V11L12-038_B1
    "Snca", "Pink1", "Park7",
    "mt-Nd2", "mt-Nd4", "mt-Nd1", "Ndufb9", "Ndufa13", "Ndufa4", "Ndufa3", # Complex I
]
if species == "human":
    MARKER_GENES = [g.upper() for g in MARKER_GENES]
    
MARKER_MSI = [
    "Dopamine",
    "(3'-sulfo)Galbeta-Cer(d18:1/24:0(2OH))",
    "307.07157500000005",
    "Glucosylceramide (d18:1/24:0)",
    "PI(12:0/22:2(13Z,16Z)), PI(14:1(9Z)/20:1(11Z)), PI(15:1(9Z)/19:1(9Z)), PI(17:2(9Z,12Z)/17:0), PI(20:1(11Z)/14:1(9Z)), PI(18:2(9Z,12Z)/16:0), PI(17:1(9Z)/17:1(9Z)), PI(16:1(9Z)/18:1(9Z)), PI(14:0/20:2(11Z,14Z)), PI(17:0/17:2(9Z,12Z)), PI(19:1(9Z)/15:1(9Z)), PI(20:2(11Z,14Z)/14:0), PI(22:2(13Z,16Z)/12:0), PI(18:1(9Z)/16:1(9Z)), PI(16:0/18:2(9Z,12Z))",
    #"5-Hydroxydantrolene",
]
MARKER_FEATURES = \
    ["rna:" + g for g in MARKER_GENES if "rna:" + g in joint_adata.var_names] + \
    ["msi:" + m for m in MARKER_MSI if "msi:" + m in joint_adata.var_names]

def spatial_plot(joint_adata, mask=None, filename=None):
    name_mapper = {
        "msi:PI(12:0/22:2(13Z,16Z)), PI(14:1(9Z)/20:1(11Z)), PI(15:1(9Z)/19:1(9Z)), PI(17:2(9Z,12Z)/17:0), PI(20:1(11Z)/14:1(9Z)), PI(18:2(9Z,12Z)/16:0), PI(17:1(9Z)/17:1(9Z)), PI(16:1(9Z)/18:1(9Z)), PI(14:0/20:2(11Z,14Z)), PI(17:0/17:2(9Z,12Z)), PI(19:1(9Z)/15:1(9Z)), PI(20:2(11Z,14Z)/14:0), PI(22:2(13Z,16Z)/12:0), PI(18:1(9Z)/16:1(9Z)), PI(16:0/18:2(9Z,12Z))":
        "msi:Phosphatidylinositols (PI)",
        "msi:(3'-sulfo)Galbeta-Cer(d18:1/24:0(2OH))":
        "Sulfated Galactosylceramide",
    }
    spatial_axes = sc.pl.spatial(
        joint_adata[mask] if mask is not None else joint_adata,
        img_key="hires" if species == "mouse" else None,
        color_map="vlag",
        color=MARKER_FEATURES + [key for key in ["lesion", "region"] if key in joint_adata.obs.keys()],
        layer="normalized",
        size=0.125 if species == "human" else 0.075,
        wspace=0.005,
        show=False,
    )
    if not isinstance(spatial_axes, list):
        spatial_axes = [spatial_axes]
    for ax in spatial_axes:
        title = ax.get_title()
        if title in name_mapper:
            ax.set_title(name_mapper[title])
    if filename is not None:
        _save_plot(spatial_axes, filename)

for sample_id in SAMPLE_IDS:
    section_adata = joint_adata[joint_adata.obs[SECTION_KEY].eq(sample_id)].copy()
    # keep the {library_id: {...}} wrapper so sc.pl.spatial sees a single library
    section_adata.uns['spatial'] = {sample_id: section_adata.uns['spatial'][sample_id]}
    spatial_plot(section_adata, filename=f"{sample_id}_marker_spatial.png")
    if species == "mouse":
        section_striatum_mask = section_adata.obs["region"].eq("striatum")
        spatial_plot(section_adata, section_striatum_mask, filename=f"{sample_id}_striatum_marker_spatial.png")


#%% Run SpatialJEPA and vanilla SpatialMETA baseline

# set training hyperparameters. disable LR scheduler for human (i.e. non-mouse)
max_epoch = 1000 if species == "mouse" else 200
# the human checkpoints behind the manuscript (Jul 1, Jul 28) were trained for 1000 epochs
max_epoch = int(os.getenv("SPATIALJEPA_MAX_EPOCH", max_epoch))
learning_rate           = 1e-3  if species == "mouse" else 1e-5
lr_scheduler_step_size  = 200   if species == "mouse" else max_epoch
lr_scheduler_gamma      = 0.1   if species == "mouse" else 1.0
# Teacher mini-batch message passing: "exact" gives every batch spot its full spatial
# neighbourhood; "induced" (keep only within-batch edges, ~0.3 neighbours/spot at k=6)
# reproduces the checkpoints behind the submitted manuscript.
TEACHER_NEIGHBORHOOD = os.getenv("SPATIALJEPA_TEACHER_NEIGHBORHOOD", "exact")
# Distillation timing (see SpatialJEPA_trainer): "batch" steps the student on every
# teacher mini-batch; "epoch" runs one student epoch after each teacher epoch.
# ISOLATE_STUDENT_RNG gives the student its own RNG stream so the teacher is identical
# under either schedule.
DISTILL_SCHEDULE = os.getenv("SPATIALJEPA_DISTILL_SCHEDULE", "batch")
EPOCH_TARGETS = os.getenv("SPATIALJEPA_EPOCH_TARGETS", "train")
ISOLATE_STUDENT_RNG = os.getenv("SPATIALJEPA_ISOLATE_STUDENT_RNG", "0") == "1"
# Teacher early stopping (as config A_early of the animal-transfer benchmark): a first pass
# trains the teacher alone on 8 of 10 spatial blocks per section and scores the other 2 after
# every epoch; the teacher is then trained on every spot for the best epoch count, and the
# student keeps distilling from the frozen teacher for the rest of max_epoch.
TEACHER_EARLY_STOP = os.getenv("SPATIALJEPA_TEACHER_EARLY_STOP", "0") == "1"
ES_PATIENCE = int(os.getenv("SPATIALJEPA_ES_PATIENCE", 100))
print(f"[run] RUN_ID={RUN_ID} N_TARGET_TOP={N_TARGET_TOP} "
      f"TEACHER_NEIGHBORHOOD={TEACHER_NEIGHBORHOOD} SEED={SEED} max_epoch={max_epoch} "
      f"DISTILL_SCHEDULE={DISTILL_SCHEDULE} EPOCH_TARGETS={EPOCH_TARGETS} "
      f"ISOLATE_STUDENT_RNG={ISOLATE_STUDENT_RNG} TEACHER_EARLY_STOP={TEACHER_EARLY_STOP}")


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# For horizontal integration (MULTI) pass the section as a batch key: this enables decoder batch conditioning + the MMD alignment loss, and a block-diagonal spatial graph (no cross-section edges) for the full-graph teacher.
batch_keys = [SECTION_KEY] if MULTI else None
section_key = SECTION_KEY if MULTI else None
train_mode = "multi" if MULTI else "single"

teacher_epochs, early_stopping = max_epoch, None
if TEACHER_EARLY_STOP:
    import importlib
    from types import SimpleNamespace

    es_cfg = dict(graph="spatial", n_neighbors=6, neighborhood=TEACHER_NEIGHBORHOOD, dropout_rate=0.1,
                  sm_cross_weight=1 if train_mode == "single" else 2, teacher_weight_decay=1e-6,
                  mode=train_mode, batch_keys=batch_keys)
    es_args = SimpleNamespace(seed=SEED or 0, es_val_blocks=2, es_patience=ES_PATIENCE, device="cuda:0",
                              max_epoch=max_epoch, lr=learning_rate, lr_step=lr_scheduler_step_size,
                              lr_gamma=lr_scheduler_gamma)
    early_stopping = importlib.import_module("heldout_animals_train").select_teacher_epochs(
        {"source_joint": joint_adata}, es_cfg, es_args, multi_section=MULTI)
    teacher_epochs = max(int(early_stopping["best_epoch"]), 1)
    with open(MODEL_DIR / "early_stopping.json", "w") as handle:
        json.dump(early_stopping, handle, default=float)
    print(f"[early stop] teacher best epoch {teacher_epochs} (stopped at {early_stopping['stopped_epoch']}); "
          f"student distils to epoch {max_epoch}", flush=True)

if SEED is not None:
    # same seed -> identical teacher/student/nonspatial initial weights for any graph
    seed_everything(SEED)
teacher_model = spatialJEPA_model(joint_adata, graph_conv=True, full_graph=True, batch_keys=batch_keys, section_key=section_key, neighborhood=TEACHER_NEIGHBORHOOD)
student_model = spatialJEPA_model(joint_adata, graph_conv=True, full_graph=False, batch_keys=batch_keys, section_key=section_key)
# TODO: ablation -- train an identity-graph ConditionalVAESTSM_STtoSM with .fit() to separate the ST->SM term from spatial context in the dopamine benchmark; this nonspatial baseline is the stock class and so lacks ST->SM, confounding the teacher/student vs nonspatial gap.
nonspatial_model = smt.model.ConditionalVAESTSM(
    joint_adata,
    device="cuda:0",
    reconstruction_method_sm="g",
    reconstruction_method_st="zinb",
    batch_keys=batch_keys,
)

# instantiate trainer for SpatialJEPA
n_per_batch = 128
spatialjepa_trainer = SpatialJEPA_trainer(
    teacher_model,
    student_model,
    n_per_batch=n_per_batch,
    distill_schedule=DISTILL_SCHEDULE,
    epoch_targets=EPOCH_TARGETS,
    student_seed=((SEED or 0) + 2) if ISOLATE_STUDENT_RNG else None,
)
spatialjepa_trainer.student_lr_scheduler = optim.lr_scheduler.StepLR(
    spatialjepa_trainer.student_optimizer,
    step_size=lr_scheduler_step_size,
    gamma=lr_scheduler_gamma,
)
print(
    "[LR] StepLR enabled for teacher/student/nonspatial: "
    f"step_size={lr_scheduler_step_size} epochs, gamma={lr_scheduler_gamma}"
)
# train models; mode="multi" upweights the MMD loss for horizontal integration
with _scheduled_adamw_epochs(
    label="teacher",
    steps_per_epoch=spatialjepa_trainer.batches_per_epoch,
    step_size=lr_scheduler_step_size,
    gamma=lr_scheduler_gamma,
) as teacher_lr_states:
    loss_dict = spatialjepa_trainer.fit_teacher(
        max_epoch=teacher_epochs,
        lr=learning_rate,
        mode=train_mode,
        cap_kl_warmup=not TEACHER_EARLY_STOP,  # keep the selection pass's KL schedule
    )
loss_dict["epoch_teacher_lr_list"] = (
    teacher_lr_states[0].epoch_lr_list if teacher_lr_states else []
)
# an early-stopped teacher stays frozen while the student finishes its schedule
for _ in range(max_epoch - teacher_epochs):
    spatialjepa_trainer.distill_epoch_end()

if SEED is not None:
    # reseed so the nonspatial baseline's training is independent of the teacher's RNG use
    seed_everything(SEED + 1)
nonspatial_batches_per_epoch = len(
    nonspatial_model.as_dataloader(batch_size=n_per_batch, shuffle=True)
)
with _scheduled_adamw_epochs(
    label="nonspatial",
    steps_per_epoch=nonspatial_batches_per_epoch,
    step_size=lr_scheduler_step_size,
    gamma=lr_scheduler_gamma,
) as nonspatial_lr_states:
    nonspatial_loss_dict = nonspatial_model.fit(
        max_epoch=max_epoch,
        n_per_batch=n_per_batch,
        lr=learning_rate,
        mode=train_mode,
    )
nonspatial_loss_dict["epoch_lr_list"] = (
    nonspatial_lr_states[0].epoch_lr_list if nonspatial_lr_states else []
)
# extract outputs
epoch_H = spatialjepa_trainer.epoch_H
batch_H = spatialjepa_trainer.batch_H
student_distill_loss = spatialjepa_trainer.student_distill_loss
epoch_student_distill_loss = spatialjepa_trainer.epoch_student_distill_loss

# CAPTION: Training loss curves for SpatialMETA (ConditionalVAESTSM). Each panel shows one tracked loss term (ST/SM reconstruction, correlation branches, KL, MMD). Use to assess convergence and balance between transcriptomics and metabolomics objectives.

def plot_training_losses(training_losses: dict, filename: str) -> None:
    ncols = 4
    nrows = int(np.ceil(len(training_losses) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.3 * nrows))
    axes = np.asarray(axes).flatten()
    for ax, (k, v) in zip(axes, training_losses.items()):
        ax.plot(v)
        ax.set_title(k)
    for ax in axes[len(training_losses):]:
        ax.axis("off")
    _save_plot(fig.axes, filename)


plot_training_losses(loss_dict, f"{RUN_ID}_training_losses.png")
plot_training_losses(nonspatial_loss_dict, f"{RUN_ID}_nonspatial_training_losses.png")

# %%

DOMAIN_MODELS = {
    "teacher": teacher_model,
    "student": student_model,
    "nonspatial": nonspatial_model,
}

CONTRIBUTION_CMAP = smt.pl.make_colormap(["#2ec4b6", "#ffffff", "#ff9f1c"])

def domain_key(domain: str, suffix: str) -> str:
    return f"{domain}_{suffix}"


@torch.no_grad()
def get_joint_and_modality_latents(model, n_per_batch: int = 128):
    """Mini-batched encode mirroring ConditionalVAESTSM.get_latent_embedding, but also
    returning the per-modality expert latents (RNA-only H['st']['q_mu'], MSI-only
    H['sm']['q_mu']) from the *same* batches. Using the identical mini-batched path
    keeps all three latents mutually consistent -- for the full-graph teacher a full-batch
    encode would use a different (whole-graph) message-passing regime than the n_per_batch
    one used to produce the joint embedding (and to train the teacher)."""
    import scipy.sparse

    model.eval()
    dataloader = model.as_dataloader(batch_size=n_per_batch, shuffle=False)
    Zs, Zs_st, Zs_sm = [], [], []
    for batch_idx in dataloader:
        indices = batch_idx[0].cpu().numpy()
        X_batch = np.stack([
            model.X.getrow(i).toarray().squeeze() if scipy.sparse.issparse(model.X) else model.X[i]
            for i in indices
        ])
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(model.device)
        H = model.encode(X_batch)
        Zs.append(H["q_mu"].detach().cpu().numpy())
        Zs_st.append(H["st"]["q_mu"].detach().cpu().numpy())
        Zs_sm.append(H["sm"]["q_mu"].detach().cpu().numpy())
    return np.vstack(Zs), np.vstack(Zs_st), np.vstack(Zs_sm)


def process_latent_embedding(domain: str) -> None:
    embedding_model = DOMAIN_MODELS[domain]

    # Joint latent (Z, == get_latent_embedding) plus the per-modality expert latents,
    # all from one mini-batched pass. The per-modality latents are persisted so
    # dopamine_embedding_benchmark.py can compute cross-modal FOSCTTM/iLISI directly from
    # joint_adata.h5ad, without reloading the SpatialMETA models (no spatialmeta dep).
    Z, Z_st, Z_sm = get_joint_and_modality_latents(embedding_model)
    X = embedding_model.get_normalized_expression()
    C = embedding_model.get_modality_contribution()

    joint_adata.layers[domain_key(domain, "reconstruction")] = X
    joint_adata.obsm[domain_key(domain, "X_emb")] = Z
    joint_adata.obsm[domain_key(domain, "X_emb_st")] = Z_st
    joint_adata.obsm[domain_key(domain, "X_emb_sm")] = Z_sm
    joint_adata.obs[domain_key(domain, "contribution_st")] = C
    joint_adata.obs[domain_key(domain, "contribution_sm")] = 1 - C

    neighbors_key = domain_key(domain, "neighbors")
    umap_key = domain_key(domain, "umap")

    sc.pp.neighbors(
        joint_adata,
        use_rep=domain_key(domain, "X_emb"),
        n_neighbors=15,
        key_added=neighbors_key,
    )
    sc.tl.umap(
        joint_adata,
        min_dist=1,
        spread=1,
        neighbors_key=neighbors_key,
    )
    joint_adata.obsm[umap_key] = joint_adata.obsm["X_umap"].copy()
    sc.tl.leiden(
        joint_adata,
        key_added=domain_key(domain, "VAE_clusters_latent10"),
        neighbors_key=neighbors_key,
    )


def _spatial_by_section(adata, **kwargs):
    """sc.pl.spatial once per section: with multiple sections, uns['spatial'] holds
    one library per section, so plotting the pooled object raises 'multiple libraries'.
    Subset to each section's spots and narrow uns to its single library. A single
    library (single-section run) falls through to one plain call.

    Returns a list of (section_label, plot_output) tuples for labeled figure saves.
    """
    libs = list(adata.uns["spatial"].keys()) if "spatial" in adata.uns else []
    if len(libs) <= 1:
        lib = libs[0] if libs else (SAMPLE_IDS[0] if SAMPLE_IDS else "all")
        return [(_section_slug(lib), sc.pl.spatial(adata, **kwargs))]
    outputs = []
    for lib in libs:
        sub = adata[adata.obs[SECTION_KEY].eq(lib)].copy()
        sub.uns["spatial"] = {lib: adata.uns["spatial"][lib]}
        outputs.append((_section_slug(lib), sc.pl.spatial(sub, **kwargs)))
    return outputs


def plot_domain_results(domain: str, plot_marker: str) -> None:
    cluster_col = domain_key(domain, "VAE_clusters_latent10")

    fig = sc.pl.embedding(
        joint_adata,
        color=[cluster_col, "lesion", plot_marker] + (["region"] if species == "mouse" else []),
        ncols=2 if species == "mouse" else 3,
        size=100,
        wspace=0.3,
        color_map="Reds",
        basis=domain_key(domain, "umap"),
        show=False,
        return_fig=True,
    )
    _save_plot(fig.axes, f"{RUN_ID}_{domain}_umap.png")
    _save_labeled_plots(
        _spatial_by_section(
            joint_adata,
            img_key="hires" if species == "mouse" else None,
            color=[cluster_col, "lesion", plot_marker] + (["region"] if species == "mouse" else []),
            size=0.125 if species == "human" else 0.075,
            show=False,
            ncols=2 if species == "mouse" else 3,
        ),
        f"{RUN_ID}_{domain}_spatial_clusters.png",
    )
    _save_labeled_plots(
        _spatial_by_section(
            joint_adata,
            img_key="hires" if species == "mouse" else None,
            color_map="vlag",
            color=MARKER_FEATURES,
            layer=domain_key(domain, "reconstruction"),
            size=0.125 if species == "human" else 0.075,
            wspace=0.005,
            show=False,
        ),
        f"{RUN_ID}_{domain}_spatial_reconstruction.png",
    )
    _save_labeled_plots(
        _spatial_by_section(
            joint_adata,
            img_key="hires" if species == "mouse" else None,
            color_map=CONTRIBUTION_CMAP,
            color=[
                domain_key(domain, "contribution_st"),
                domain_key(domain, "contribution_sm"),
            ],
            layer="normalized",
            wspace=0.005,
            show=False,
            alpha_img=0.1,
            size=0.125 if species == "human" else 0.075,
        ),
        f"{RUN_ID}_{domain}_spatial_modality_contribution.png",
    )

    obs_filter_df = pd.concat([
        joint_adata.obs[[cluster_col, domain_key(domain, "contribution_st")]]
        .rename(columns={domain_key(domain, "contribution_st"): "contribution"})
        .assign(type="st"),
        joint_adata.obs[[cluster_col, domain_key(domain, "contribution_sm")]]
        .rename(columns={domain_key(domain, "contribution_sm"): "contribution"})
        .assign(type="sm"),
    ])
    fig, ax = smt.pl.create_fig(figsize=(12, 4))
    sns.violinplot(
        data=obs_filter_df,
        x=cluster_col,
        y="contribution",
        hue="type",
        split=True,
        inner="quart",
        palette=["#2ec4b6", "#FFCC70"],
        scale="width",
        bw=0.2,
        cut=0,
    )
    plt.xticks(rotation=90)
    _save_plot(ax, f"{RUN_ID}_{domain}_cluster_modality_contribution.png")


for domain in DOMAIN_MODELS:
    process_latent_embedding(domain)


#%% train PLS decoder, post-hoc

from sklearn.cross_decomposition import PLSRegression
import anndata as ad
import joblib

OUTPUT_DIR = Path(os.getenv("SPATIALJEPA_OUT"))
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / RUN_ID
MODEL_DIR.mkdir(parents=True, exist_ok=True)

emb = joint_adata.obsm["student_X_emb"]
sm_features = joint_adata.var_names[joint_adata.var['type'].eq('SM')]
sm = joint_adata[:, sm_features].X.toarray()

## log transform and scale the SM features
sm = np.log1p(sm)
sm = sc.pp.scale(sm, zero_center=True, max_value=10)
sm_keep_mask = np.logical_not(np.isnan(sm).all(0))
sm = sm[:,sm_keep_mask]
sm[sm < -np.nanmax(sm)] = -np.nanmax(sm)
sm_features = sm_features[sm_keep_mask]

plt.hist(sm.flatten(), bins=100); plt.show()
plt.hist(sm[:,sm_features.isin(['msi:Dopamine'])]); plt.show()

pls = PLSRegression(n_components=4)
pls.fit(emb, sm)
pls_decoder_path = MODEL_DIR / "student_pls_decoder.joblib"
joblib.dump(
    {
        "model": pls,
        "embedding_key": "student_X_emb",
        "target_features": sm_features.astype(str).tolist(),
        "target_transform": {
            "log1p": True,
            "scale_zero_center": True,
            "scale_max_value": 10,
            "output_space": "scaled_log1p_sm",
        },
        "run_id": RUN_ID,
    },
    pls_decoder_path,
)
print(f"[INFO] Saved student PLS decoder to {pls_decoder_path.resolve()}")
T = pls.transform(emb)

joint_pls_adata = ad.AnnData(
    T,
    obs=joint_adata.obs,
    var=pd.DataFrame(index=[f'PLS_{i}' for i in range(T.shape[1])]),
    obsm={"student_umap": joint_adata.obsm["student_umap"]}
    )
    
#joint_adata.obsm["student_X_emb_pls"] = T
#joint_adata.layers["student_X_pls"] = pls.transform(joint_adata.X)

pls_coef_df = pd.DataFrame(
    pls.y_loadings_.T,
    index=[f'PLS_{i}' for i in range(pls.y_weights_.shape[1])],
    columns=sm_features
)

## check the coefficients for Dopamine
pls_coef_df['msi:Dopamine'].plot(kind='bar')
plt.show()

sc.pl.embedding(joint_pls_adata, basis='student_umap', color=[f'PLS_{i}' for i in range(pls.y_weights_.shape[1])],
    cmap='coolwarm') # set reverse colormap (_r) if weights are negative for dominant PLS components
sc.pl.embedding(joint_adata, basis='student_umap', color=['msi:Dopamine'], cmap='Reds')

# %%
# CAPTION: UMAP of the joint ST+SM latent embedding (10-dim VAE, Leiden clusters). Colors: VAE clusters, tissue region, lesion status, and Dopamine (MSI). Shows how anatomy and pathology align with the integrated representation.
plot_marker = MARKER_FEATURES[-1]
print("Plotting marker:", plot_marker)

for domain in DOMAIN_MODELS:
    plot_domain_results(domain, plot_marker)

if MULTI:
    cluster_crosstab = pd.crosstab(joint_adata.obs['teacher_VAE_clusters_latent10'], joint_adata.obs['section'])
    ax = cluster_crosstab.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.title("Cluster distribution by section")
    plt.ylabel("Count")
    _save_plot(ax, f"{RUN_ID}_teacher_cluster_distribution_by_section.png")

#%%
print(f"[INFO] Saving models to {MODEL_DIR.resolve()}")

# The student's decoder is never trained (distillation only touches the encoder/latent
# heads), so copy the teacher's trained, graph-free decoder weights before saving.
copied, skipped = copy_decoder_weights(teacher_model, student_model)
print(f"[student] copied decoder weights: {copied}; skipped (graph): {skipped}")

def save_vanilla_spatialmeta_model(model, path: Path) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_class": "spatialmeta.model.ConditionalVAESTSM",
            "full_graph": False,
            "graph_conv": False,
            "reconstruction_method_sm": model.reconstruction_method_sm,
            "reconstruction_method_st": model.reconstruction_method_st,
            "hidden_stacks": model.hidden_stacks,
            "n_latent": model.n_latent,
            "batch_keys": getattr(model, "batch_keys", None),
        },
        path,
    )


save_spatialjepa_model(teacher_model, MODEL_DIR / "teacher.pt", full_graph=True)
save_spatialjepa_model(student_model, MODEL_DIR / "student.pt", full_graph=False)
save_vanilla_spatialmeta_model(nonspatial_model, MODEL_DIR / "nonspatial.pt")

# model.initialize_dataset() reads from adata.X at construction time
joint_adata.write_h5ad(MODEL_DIR / "joint_adata.h5ad")
print(f"[INFO] Saved teacher.pt, student.pt, nonspatial.pt, joint_adata.h5ad under {MODEL_DIR.resolve()}")

# %%
