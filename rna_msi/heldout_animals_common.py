"""Shared helpers for the cross-animal SpatialJEPA ablation benchmark.

Used by ``heldout_animals_train.py`` (GPU, one fold x config x seed per call) and
``heldout_animals_evaluate.py`` (CPU, PLS readouts + metrics + figures).

Design: train on one animal's FMP-10 section (or two, for the optional
leave-one-animal-out mode) and predict MSI from RNA alone on the other animals'
sections of the same region. Every preprocessing statistic comes from the source
section(s); target sections only contribute RNA to the model, and their measured MSI
and coordinates are written to separate files that the training script never reads.
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

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BAKLAVA_BASE = Path(os.environ.get("SPATIALJEPA_HOME", os.environ['SPATIALJEPA_HOME']))
SPATIALJEPA_OUT = Path(os.environ.get("SPATIALJEPA_OUT", BAKLAVA_BASE / "outputs"))
H5MU_EXPORT_DIR = BAKLAVA_BASE / "data" / "vicari_2023" / "h5mu_export"
# override for a separate benchmark (e.g. another gene panel): fold caches + runs
BENCH_DIR = Path(os.environ.get("ANIMAL_TRANSFER_BENCH_DIR",
                                SPATIALJEPA_OUT / "spatialjepa_benchmark" / "animal_transfer"))
# trained runs live here (fold caches always under BENCH_DIR); override for smoke tests
RUNS_DIR = Path(os.environ.get("ANIMAL_TRANSFER_RUNS_DIR", BENCH_DIR))
# fixed external gene ranking used by the Fig. 5 mouse model (train_rna_msi.py)
TARGET_PANEL_PATH = SPATIALJEPA_OUT / "target_gene_rankings" / "target_gene_ranking_p22_mouse_spatial_atac_rna.csv"
# 442 reproduces the submitted Fig. 5 model (V11L12-109_B1: 388 ST genes after Moran's I);
# the revised manuscript's Fig. 5 (fix2000_s3) uses 2000 (1,806 ST genes).
N_TARGET_TOP = int(os.environ.get("ANIMAL_TRANSFER_PANEL_TOP", 442))

SECTION_KEY = "section"
N_PER_BATCH = 128

# FMP-10 Parkinsonian mice (mendeley_sma/metadata.csv): A1/B1/C1 = mPD1/mPD3/mPD4 on
# both the striatum (V11L12-109) and substantia nigra (V11T16-085) slides.
ANIMALS = ("mPD1", "mPD3", "mPD4")
SECTIONS = {
    ("mPD1", "striatum"): "V11L12-109_A1",
    ("mPD3", "striatum"): "V11L12-109_B1",
    ("mPD4", "striatum"): "V11L12-109_C1",
    ("mPD1", "sn"): "V11T16-085_A1",
    ("mPD3", "sn"): "V11T16-085_B1",
    ("mPD4", "sn"): "V11T16-085_C1",
}
# obs['region'] value of the anatomical region each slide targets
REGION_LABEL = {"striatum": "striatum", "sn": "SubNig"}

# annotated FMP-10 neurotransmitters; always kept as model features and PLS targets
NEUROTRANSMITTERS = (
    "Dopamine", "3-MT", "GABA", "Serotonin", "Norepinephrine",
    "Taurine", "Histidine", "Tocopherol", "Dopamine (single)",
)
PRIMARY_METABOLITE = "msi:Dopamine"

# Teacher configurations. B/D teachers are graph-free and double as the direct
# (no-teacher) controls E/F: the student never sends gradients to the teacher.
CONFIGS = {
    "A": dict(kind="jepa", graph="spatial", n_neighbors=6, sm_cross_weight=1.0, neighborhood="exact"),
    "A30": dict(kind="jepa", graph="spatial", n_neighbors=30, sm_cross_weight=1.0, neighborhood="exact"),
    "B": dict(kind="jepa", graph="identity", n_neighbors=6, sm_cross_weight=1.0, neighborhood="exact"),
    "C": dict(kind="jepa", graph="spatial", n_neighbors=6, sm_cross_weight=0.0, neighborhood="exact"),
    "C30": dict(kind="jepa", graph="spatial", n_neighbors=30, sm_cross_weight=0.0, neighborhood="exact"),
    "D": dict(kind="jepa", graph="identity", n_neighbors=6, sm_cross_weight=0.0, neighborhood="exact"),
    "A_perm": dict(kind="jepa", graph="permuted", n_neighbors=6, sm_cross_weight=1.0, neighborhood="exact"),
    "A_induced": dict(kind="jepa", graph="spatial", n_neighbors=6, sm_cross_weight=1.0, neighborhood="induced"),
    # A with the teacher early-stopped on held-out spatial blocks of the source, then
    # retrained on every source spot for the selected number of epochs (7__)
    "A_early": dict(kind="jepa", graph="spatial", n_neighbors=6, sm_cross_weight=1.0, neighborhood="exact",
                    early_stop=True),
    # A with strong decoupled weight decay on the teacher (AdamW 1.0 instead of 1e-6), full
    # schedule: the setting that kept the held-out-block loss flat in the mPD3 pilots
    "A_wd": dict(kind="jepa", graph="spatial", n_neighbors=6, sm_cross_weight=1.0, neighborhood="exact",
                 teacher_weight_decay=1.0),
    "G": dict(kind="stock"),
}
IDENTITY_TEACHER_NAMES = {"B": "E", "D": "F"}

# The RNA-only student cannot carry observation-specific MSI information, so it is
# distilled on the joint and RNA-expert latents only.
RNA_ONLY_DISTILL_TARGETS = (
    (("q_mu",), 1.0),
    (("st", "q_mu"), 0.5),
)


# ---------------------------------------------------------------------------
# Paths and identifiers
# ---------------------------------------------------------------------------

def parse_train_unit(unit: str) -> tuple[str, int | None]:
    """'mPD3' -> ('mPD3', None); 'mPD3@h0' -> ('mPD3', 0): half 0/1 of that animal's section."""
    animal, _, half = unit.partition("@h")
    if animal not in ANIMALS or half not in ("", "0", "1"):
        raise ValueError(f"training unit must be one of {ANIMALS}, optionally with @h0/@h1: {unit!r}")
    return animal, (int(half) if half else None)


def base_section(section_id: str) -> str:
    """'V11L12-109_B1@h0' -> 'V11L12-109_B1'."""
    return section_id.split("@", 1)[0]


def unit_section_id(unit: str, region: str) -> str:
    animal, half = parse_train_unit(unit)
    sample_id = SECTIONS[(animal, region)]
    return sample_id if half is None else f"{sample_id}@h{half}"


def target_section_ids(train_animals, region: str) -> list[str]:
    """Held-out sections of a fold.

    Whole-section folds hold out the other animals' sections. Half-section folds
    (``mPD3@h0``) hold out the complementary half of the same section *and* the other
    animals' full sections, so one trained model is scored on unseen tissue from its own
    section (no batch effect) and from other animals.
    """
    parsed = [parse_train_unit(u) for u in train_animals]
    trained = {a for a, _ in parsed}
    targets = []
    for animal, half in parsed:
        if half is not None:
            targets.append(f"{SECTIONS[(animal, region)]}@h{1 - half}")
    targets += [SECTIONS[(a, region)] for a in ANIMALS if a not in trained]
    return targets


def section_half_mask(adata, half: int) -> np.ndarray:
    """Split a section across its intact-lesioned axis into two halves.

    The axis joining the intact and lesioned hemisphere centroids is left-right; the
    cut runs along it, at the median of the striatum spots' position on the orthogonal
    axis, so each half keeps intact and lesioned striatum in roughly equal numbers.
    """
    coords = np.asarray(adata.obsm["spatial"], float)
    intact = adata.obs["lesion"].astype(str).eq("intact").to_numpy()
    striatum = adata.obs["region"].astype(str).isin(set(REGION_LABEL.values())).to_numpy()
    u = coords[intact].mean(0) - coords[~intact].mean(0)
    u /= np.linalg.norm(u)
    proj = coords @ np.array([-u[1], u[0]])
    upper = proj > np.median(proj[striatum])
    return upper if half == 1 else ~upper


def assemble_unit(section_id: str, msi_names: dict[str, str]):
    """A whole section, or one half of it for ids like 'V11L12-109_B1@h0'."""
    adata = assemble_section(base_section(section_id), msi_names)
    if "@h" in section_id:
        adata = adata[section_half_mask(adata, int(section_id.rsplit("@h", 1)[1]))].copy()
        adata.obs[SECTION_KEY] = section_id
    return adata


def target_animals(train_animals) -> list[str]:
    return [a for a in ANIMALS if a not in train_animals]


def fold_name(train_animals) -> str:
    return "train-" + "+".join(train_animals)


def fold_dir(region: str, train_animals) -> Path:
    return BENCH_DIR / region / fold_name(train_animals)


def fold_data_dir(region: str, train_animals) -> Path:
    return fold_dir(region, train_animals) / "data"


def run_id(region: str, train_animals, config: str, seed: int) -> str:
    return f"{region}__{fold_name(train_animals)}__{config}__seed{seed}"


def run_dir(region: str, train_animals, config: str, seed: int) -> Path:
    return RUNS_DIR / region / fold_name(train_animals) / config / f"seed{seed}"


def representation_names(config: str, rna_student: bool = True) -> dict[str, str]:
    """Map evaluated representation name -> trained model key for one run."""
    cfg = CONFIGS[config]
    if cfg["kind"] == "stock":
        return {config: "model"}
    reps = {config: "student"}
    if rna_student:
        reps[f"{config}_rna"] = "student_rna"
    if cfg["graph"] == "identity":
        reps[IDENTITY_TEACHER_NAMES.get(config, f"{config}_teacher")] = "teacher"
    return reps


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spatial_blocks(coords: np.ndarray, sections=None, n_blocks: int = 10) -> np.ndarray:
    """Contiguous spatial blocks (KMeans on coordinates, per section) for spatial CV."""
    from sklearn.cluster import KMeans

    sections = np.zeros(len(coords), dtype=object) if sections is None else np.asarray(sections)
    groups = np.empty(len(coords), dtype=object)
    for s in np.unique(sections):
        m = sections == s
        labels = KMeans(n_clusters=n_blocks, random_state=0, n_init=10).fit_predict(coords[m])
        groups[m] = [f"{s}:{b}" for b in labels]
    return groups


def validation_blocks(groups: np.ndarray, sections, seed: int, n_val: int = 2) -> np.ndarray:
    """Boolean mask of ``n_val`` blocks per section drawn with ``seed`` (validation spots)."""
    rng = np.random.default_rng(seed)
    sections = np.asarray(sections)
    val = np.zeros(len(groups), bool)
    for s in np.unique(sections):
        blocks = np.unique(groups[sections == s])
        val |= np.isin(groups, rng.choice(blocks, size=n_val, replace=False))
    return val


def parameter_hash(*models) -> str:
    """Hash of model parameters (not buffers), to verify identical initialisation."""
    h = hashlib.sha1()
    for model in models:
        for name, param in sorted(model.named_parameters()):
            h.update(name.encode())
            h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Section assembly
# ---------------------------------------------------------------------------

def _dense(X) -> np.ndarray:
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def canonical_msi_names(sample_ids=None) -> dict[str, str]:
    """Fixed m/z -> ``msi:<name>`` map shared by all sections.

    Annotation coverage differs between sections (the striatum sections annotate a few
    extra peaks), so naming each section by its own annotations would mis-align
    features. A peak takes the first annotation found over sections in a fixed order;
    unannotated peaks keep their m/z. Duplicate names get the m/z appended.
    """
    sample_ids = list(sample_ids or SECTIONS.values())
    names: dict[str, str] = {}
    for sample_id in sample_ids:
        msi_var = _read_msi_var(sample_id)
        annotations = msi_var["annotation"].astype("string")
        for mz, annotation in zip(msi_var.index.astype(str), annotations):
            if mz in names and names[mz] != mz:
                continue
            has = annotation is not pd.NA and str(annotation).strip() not in ("", "nan", "None")
            names[mz] = str(annotation).strip() if has else mz

    out, seen = {}, set()
    for mz, name in names.items():
        label = f"msi:{name}"
        if label in seen:
            label = f"msi:{name} [{mz}]"
        seen.add(label)
        out[mz] = label
    return out


def _h5mu_path(sample_id: str) -> Path:
    sct_path = H5MU_EXPORT_DIR / f"{sample_id}_SCT.h5mu"
    if not sct_path.exists():
        raise FileNotFoundError(f"SCT-transformed export not found: {sct_path}")
    return sct_path


def _read_msi_var(sample_id: str) -> pd.DataFrame:
    import mudata as mu

    mdata = mu.read_h5mu(_h5mu_path(sample_id), backed="r")
    return mdata.mod["msi"].var.copy()


def assemble_section(sample_id: str, msi_names: dict[str, str]):
    """One section as a joint ST|SM AnnData (counterpart of script 3's assemble_section).

    X holds raw counts (RNA) and raw intensities (MSI). Layers ``SCT_counts`` and
    ``SCT_data`` hold the per-section SCTransform outputs in the ST columns and raw MSI
    in the SM columns, as in ``train_rna_msi.py``.
    """
    import anndata as ad
    import muon as mu

    mdata = mu.read_h5mu(_h5mu_path(sample_id))
    mu.pp.intersect_obs(mdata)
    rna = mdata.mod["rna"]
    msi = mdata.mod["msi"][rna.obs_names]

    msi_raw = sp.csr_matrix(msi.X, dtype=np.float32)

    def joint_block(rna_mat):
        return sp.hstack([sp.csr_matrix(rna_mat, dtype=np.float32), msi_raw], format="csr")

    st_names = ["rna:" + str(g) for g in rna.var_names]
    sm_names = [msi_names[str(mz)] for mz in msi.var_names]
    var = pd.DataFrame(
        {
            "type": ["ST"] * len(st_names) + ["SM"] * len(sm_names),
            "feature_id": list(map(str, rna.var_names)) + list(map(str, msi.var_names)),
        },
        index=st_names + sm_names,
    )
    if not var.index.is_unique:
        raise ValueError(f"{sample_id}: joint feature names are not unique")

    animal, region = next(k for k, v in SECTIONS.items() if v == sample_id)
    obs = pd.DataFrame(index=rna.obs_names.astype(str))
    for col in ("lesion", "region", "RNA_clusters", "MSI_clusters", "MM_clusters"):
        if col in rna.obs.columns:
            obs[col] = rna.obs[col].astype(str).values
    obs[SECTION_KEY] = sample_id
    obs["animal"] = animal

    adata = ad.AnnData(
        X=joint_block(rna.layers["counts"]),
        obs=obs,
        var=var,
        layers={
            "SCT_counts": joint_block(rna.layers["SCT_counts"]),
            "SCT_data": joint_block(rna.layers["SCT_data"]),
        },
        obsm={"spatial": np.asarray(rna.obsm["spatial"], dtype=float)},
    )
    return adata


def _concat_sections(sections):
    import anndata as ad

    if len(sections) == 1:
        return sections[0]
    out = ad.concat(sections, axis=0, join="inner", merge="same")
    out.obs[SECTION_KEY] = out.obs[SECTION_KEY].astype(str)
    return out


# ---------------------------------------------------------------------------
# Fold preparation (source-only statistics)
# ---------------------------------------------------------------------------

def _tic_normalize(X: np.ndarray, target_sum: float) -> np.ndarray:
    totals = X.sum(axis=1, keepdims=True)
    return X / np.where(totals > 0, totals, 1.0) * target_sum


def _sm_model_space(sm_tic: np.ndarray, sm_std: np.ndarray, clip: bool = True) -> np.ndarray:
    """log1p(TIC) / source std (scanpy scale, zero_center=False, max_value=10)."""
    out = np.log1p(sm_tic) / sm_std
    return np.minimum(out, 10.0) if clip else out


def prepare_fold(
    region: str,
    train_animals,
    *,
    use_target_panel: bool = True,
    n_target_top: int = N_TARGET_TOP,
    out_dir: Path | None = None,
) -> Path:
    """Build and cache the source/target matrices for one fold.

    Mirrors ``prepare_hvf_joint_adata`` in ``train_rna_msi.py`` with all
    statistics (feature selection, SM scaling) computed on the source section(s):

    - ``source_joint.h5ad``: model input (ST = SCT_counts, SM = TIC(1e3) -> log1p ->
      / source std, clipped at 10) with ``obsm['spatial']`` for the teacher graph.
    - ``target_<id>_joint.h5ad``: target RNA reindexed to the source features, missing
      genes zero-filled, **SM block all zeros**, no coordinates.
    - ``{source,target_<id>}_msi.h5ad``: measured MSI (X = model space, layer ``tic``)
      with lesion/region labels and coordinates -- evaluation only.
    """
    import scanpy as sc
    import spatialmeta as smt
    from feature_panel import load_target_panel, restrict_st_to_target_panel

    train_animals = list(train_animals)
    out_dir = Path(out_dir or fold_data_dir(region, train_animals))
    out_dir.mkdir(parents=True, exist_ok=True)
    multi = len(train_animals) > 1

    source_ids = [unit_section_id(a, region) for a in train_animals]
    target_ids = target_section_ids(train_animals, region)
    msi_names = canonical_msi_names()

    source = _concat_sections([assemble_unit(s, msi_names) for s in source_ids])
    joint = smt.util._classes.AnnDataJointSMST(source)
    joint = smt.pp.removeHsp_mt_Rpl_Dnaj(joint)

    # normalized view used for spatial-variability selection (script 3: SCT_data for ST,
    # total-count 1e4 for SM)
    joint.X = joint.layers["SCT_data"].copy()
    smt.pp.normalize_total_joint_adata_sm_st(joint, target_sum_SM=1e4, target_sum_ST=None)

    panel_diag = None
    if use_target_panel:
        joint, panel_diag = restrict_st_to_target_panel(
            joint, load_target_panel(TARGET_PANEL_PATH), n_target_top=n_target_top
        )

    smt.pp.spatial_variable_joint_adata_sm_st(
        joint,
        n_top_genes=2000,
        n_top_metabolites=800,
        add_key="highly_variable_moranI",
        batch_key=SECTION_KEY if multi else None,
        min_frac=0.8,
        min_logfc=25,
    )
    forced = {f"msi:{name}" for name in NEUROTRANSMITTERS} & set(joint.var_names)
    selected = joint.var["highly_variable_moranI"].to_numpy(dtype=bool)
    forced_added = sorted(f for f in forced if not selected[joint.var_names.get_loc(f)])
    keep = selected | joint.var_names.isin(sorted(forced))
    joint = joint[:, keep].copy()
    joint.var["forced_include"] = joint.var_names.isin(forced_added)

    st_mask = joint.var["type"].eq("ST").to_numpy()
    sm_mask = ~st_mask

    # zero RNA library over the retained genes -> log(0) in the ZINB loss (script 3)
    st_lib = np.asarray(joint[:, st_mask].X.sum(1)).ravel()
    n_empty = int((st_lib == 0).sum())
    if n_empty:
        joint = joint[st_lib > 0].copy()
    # as_dataloader has no drop_last: a size-1 final batch breaks BatchNorm
    n_guard = 0
    if joint.n_obs % N_PER_BATCH == 1:
        joint = joint[: joint.n_obs - 1].copy()
        n_guard = 1

    st_genes = joint.var_names[st_mask]
    sm_feats = joint.var_names[sm_mask]

    X_st = _dense(joint[:, st_genes].layers["SCT_counts"]).astype(np.float32)
    sm_tic = _tic_normalize(_dense(joint[:, sm_feats].layers["SCT_counts"]).astype(np.float64), 1e3)
    sm_std = np.log1p(sm_tic).std(axis=0, ddof=1)
    sm_std[~np.isfinite(sm_std) | (sm_std == 0)] = 1.0
    X_sm = _sm_model_space(sm_tic, sm_std).astype(np.float32)

    import anndata as ad

    var = joint.var[["type", "feature_id", "forced_include"]].copy()
    var["source_sm_std"] = np.nan
    var.loc[sm_feats, "source_sm_std"] = sm_std
    obs_cols = [c for c in ("lesion", "region", SECTION_KEY, "animal") if c in joint.obs.columns]

    source_joint = ad.AnnData(
        X=np.hstack([X_st, X_sm]),
        obs=joint.obs[obs_cols].copy(),
        var=var,
        obsm={"spatial": np.asarray(joint.obsm["spatial"])},
    )
    source_joint.write_h5ad(out_dir / "source_joint.h5ad")
    _msi_adata(joint.obs[obs_cols], sm_feats, sm_tic, sm_std, joint.obsm["spatial"]).write_h5ad(
        out_dir / "source_msi.h5ad"
    )

    missing_genes = {}
    for target_id in target_ids:
        target = assemble_unit(target_id, msi_names)
        tgt_st = pd.DataFrame(
            _dense(target.layers["SCT_counts"][:, target.var["type"].eq("ST").to_numpy()]),
            index=target.obs_names,
            columns=target.var_names[target.var["type"].eq("ST").to_numpy()],
        )
        missing = [g for g in st_genes if g not in tgt_st.columns]
        missing_genes[target_id] = missing
        X_tst = tgt_st.reindex(columns=st_genes, fill_value=0.0).to_numpy(np.float32)
        target_joint = ad.AnnData(
            X=np.hstack([X_tst, np.zeros((target.n_obs, len(sm_feats)), dtype=np.float32)]),
            obs=target.obs[obs_cols].copy(),
            var=var.copy(),
        )
        target_joint.write_h5ad(out_dir / f"target_{target_id}_joint.h5ad")

        tgt_sm = target[:, sm_feats]
        tgt_tic = _tic_normalize(_dense(tgt_sm.layers["SCT_counts"]).astype(np.float64), 1e3)
        _msi_adata(target.obs[obs_cols], sm_feats, tgt_tic, sm_std, target.obsm["spatial"]).write_h5ad(
            out_dir / f"target_{target_id}_msi.h5ad"
        )

    fold_meta = dict(
        region=region,
        train_animals=train_animals,
        source_sections=source_ids,
        target_sections=target_ids,
        n_source_obs=int(source_joint.n_obs),
        n_st=int(st_mask.sum()),
        n_sm=int(sm_mask.sum()),
        forced_include=forced_added,
        dropped_zero_rna_library=n_empty,
        dropped_batch_guard=n_guard,
        missing_genes={k: len(v) for k, v in missing_genes.items()},
        missing_gene_examples={k: v[:10] for k, v in missing_genes.items()},
        target_panel=str(TARGET_PANEL_PATH) if use_target_panel else None,
        n_target_top=n_target_top if use_target_panel else None,
        panel_diagnostics=panel_diag,
    )
    with open(out_dir / "fold.json", "w") as handle:
        json.dump(fold_meta, handle, indent=2, default=str)
    return out_dir


def _msi_adata(obs, sm_feats, sm_tic, sm_std, spatial):
    import anndata as ad

    return ad.AnnData(
        X=_sm_model_space(sm_tic, sm_std).astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame({"source_sm_std": sm_std}, index=pd.Index(sm_feats)),
        layers={"tic": sm_tic.astype(np.float32)},
        obsm={"spatial": np.asarray(spatial, dtype=float)},
    )


def ensure_fold_cache(
    region: str,
    train_animals,
    *,
    use_target_panel: bool = True,
    n_target_top: int = N_TARGET_TOP,
    poll_s: int = 30,
) -> Path:
    """Return the fold cache, building it under an atomic lock if missing."""
    import time

    data_dir = fold_data_dir(region, train_animals)
    if (data_dir / "fold.json").exists():
        with open(data_dir / "fold.json") as handle:
            cached = json.load(handle).get("n_target_top")
        wanted = n_target_top if use_target_panel else None
        if cached != wanted:
            raise RuntimeError(
                f"{data_dir} was built with n_target_top={cached}, not {wanted}; "
                "delete it to rebuild with the new panel."
            )
        return data_dir
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = data_dir.parent / "data.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        while not (data_dir / "fold.json").exists():
            if not lock.exists():
                return ensure_fold_cache(
                    region, train_animals, use_target_panel=use_target_panel, n_target_top=n_target_top
                )
            time.sleep(poll_s)
        return data_dir
    try:
        prepare_fold(
            region,
            train_animals,
            use_target_panel=use_target_panel,
            n_target_top=n_target_top,
            out_dir=data_dir,
        )
    finally:
        lock.rmdir()
    return data_dir


def load_fold(region: str, train_animals) -> dict:
    """Load a cached fold: source joint/msi and per-target joint/msi AnnData."""
    import anndata as ad

    data_dir = fold_data_dir(region, train_animals)
    with open(data_dir / "fold.json") as handle:
        meta = json.load(handle)
    fold = dict(
        meta=meta,
        source_joint=ad.read_h5ad(data_dir / "source_joint.h5ad"),
        source_msi=ad.read_h5ad(data_dir / "source_msi.h5ad"),
        target_joint={},
        target_msi={},
    )
    for target_id in meta["target_sections"]:
        fold["target_joint"][target_id] = ad.read_h5ad(data_dir / f"target_{target_id}_joint.h5ad")
        fold["target_msi"][target_id] = ad.read_h5ad(data_dir / f"target_{target_id}_msi.h5ad")
    return fold


# ---------------------------------------------------------------------------
# Encoding trained models on new data
# ---------------------------------------------------------------------------

def mask_sm_input(joint):
    """Copy of ``joint`` with the SM (MSI) input block set to zero."""
    out = joint.copy()
    sm_mask = out.var["type"].eq("SM").to_numpy()
    X = _dense(out.X).astype(np.float32)
    X[:, sm_mask] = 0.0
    out.X = X
    return out


def encode_latents(model, joint, *, mask_sm: bool, device: str = "cuda:0", batch_size: int = N_PER_BATCH) -> dict:
    """Joint (``q_mu``) and RNA-expert (``st.q_mu``) latents of a graph-free model.

    The model is re-hosted on ``joint`` (same features) with its trained weights; only
    graph-free models qualify, so a target's coordinates can never enter the encoder.
    """
    import spatialmeta as smt
    import torch
    from spatialjepa.spatialmeta_backbone import rebuild_spatialjepa_model

    data = smt.util._classes.AnnDataJointSMST(mask_sm_input(joint) if mask_sm else joint.copy())
    meta = getattr(model, "_spatialjepa_meta", None)
    if meta is not None:
        if meta["graph"] != "identity":
            raise ValueError("only graph-free (identity-graph) models can be encoded on new data")
        hosted = rebuild_spatialjepa_model(
            model.state_dict(), dict(meta, batch_keys=None), data, device=device
        )
        for encoder in (hosted.encoder_ST, hosted.encoder_SM):
            edge_index = encoder.layers[0].edge_index
            assert torch.equal(edge_index[0], edge_index[1]), "re-hosted encoder is not identity-graph"
    else:
        hosted = smt.model.ConditionalVAESTSM(
            data, device=device, reconstruction_method_sm="g", reconstruction_method_st="zinb"
        )
        hosted.load_state_dict(model.state_dict())
        hosted.eval()

    hosted.eval()
    z_joint, z_st = [], []
    with torch.no_grad():
        for batch in hosted.as_dataloader(batch_size=batch_size, shuffle=False):
            idx = batch[0].detach().cpu().numpy()
            X = torch.as_tensor(_dense(hosted.X[idx]), dtype=torch.float32, device=hosted.device)
            H = hosted.encode(X)
            z_joint.append(H["q_mu"].detach().cpu().numpy())
            z_st.append(H["st"]["q_mu"].detach().cpu().numpy())
    del hosted
    return {"joint": np.vstack(z_joint).astype(np.float32), "st": np.vstack(z_st).astype(np.float32)}


def encode_teacher_insample(run_dir: Path, source_joint, *, device: str = "cuda:0",
                            batch_size: int = N_PER_BATCH) -> dict | None:
    """In-sample latents of a run's graph teacher on its own source section(s).

    Graph teachers cannot be applied to held-out animals (their graph would need the
    target's coordinates), but on the source they reproduce the manuscript's Fig. 5
    setting. Encoding uses sequential mini-batches of ``batch_size`` as in
    ``train_rna_msi.get_joint_and_modality_latents``; with
    ``neighborhood="induced"`` this keeps the legacy within-batch subgraphs. Returns
    ``{"full"|"masked": {"joint"|"st": Z}}``, or None for identity-graph teachers.
    """
    import spatialmeta as smt
    import torch
    from spatialjepa.spatialmeta_backbone import rebuild_spatialjepa_model

    ckpt = torch.load(Path(run_dir) / "teacher.pt", map_location=device)
    if ckpt.get("graph", "spatial") == "identity":
        return None
    multi = source_joint.obs[SECTION_KEY].nunique() > 1
    meta = dict(ckpt, section_key=ckpt.get("section_key") or (SECTION_KEY if multi else None))
    out = {}
    for inputs in ("full", "masked"):
        data = smt.util._classes.AnnDataJointSMST(
            mask_sm_input(source_joint) if inputs == "masked" else source_joint.copy()
        )
        model = rebuild_spatialjepa_model(ckpt["state_dict"], meta, data, device=device)
        out[inputs] = _encode_sequential(model, batch_size)
        del model
    return out


def _encode_sequential(model, batch_size: int = N_PER_BATCH) -> dict:
    """Joint and RNA-expert latents over ``model.X`` in sequential mini-batches."""
    import torch

    model.eval()
    z_joint, z_st = [], []
    with torch.no_grad():
        for batch in model.as_dataloader(batch_size=batch_size, shuffle=False):
            idx = batch[0].detach().cpu().numpy()
            X = torch.as_tensor(_dense(model.X[idx]), dtype=torch.float32, device=model.device)
            H = model.encode(X)
            z_joint.append(H["q_mu"].detach().cpu().numpy())
            z_st.append(H["st"]["q_mu"].detach().cpu().numpy())
    return {"joint": np.vstack(z_joint).astype(np.float32), "st": np.vstack(z_st).astype(np.float32)}


def paired_target_joint(fold: dict, target_id: str):
    """A held-out section with its *measured* MSI as model input (Fig. 5d-f condition).

    The training script only ever sees targets with the MSI block zeroed; this paired
    version is for representation analyses, not prediction. The MSI block is the target's
    measured MSI in model space (source-fitted scaling) and ``obsm['spatial']`` carries
    the target's own coordinates, so a graph teacher can build the section's graph at
    inference.
    """
    joint = fold["target_joint"][target_id].copy()
    msi = fold["target_msi"][target_id]
    sm_cols = joint.var["type"].eq("SM").to_numpy()
    if list(joint.var_names[sm_cols]) != list(msi.var_names):
        raise ValueError(f"{target_id}: MSI feature order differs from the model input")
    if list(joint.obs_names) != list(msi.obs_names):
        raise ValueError(f"{target_id}: spot order differs between joint and MSI files")
    X = _dense(joint.X).astype(np.float32)
    X[:, sm_cols] = _dense(msi.X).astype(np.float32)
    joint.X = X
    joint.obsm["spatial"] = np.asarray(msi.obsm["spatial"], dtype=float)
    return joint


def encode_checkpoint(ckpt_path: Path, joint, *, device: str = "cuda:0", section_key=None) -> dict:
    """Host a saved benchmark model on ``joint`` and return its latents.

    SpatialJEPA checkpoints are rebuilt with their own graph mode: identity models need
    nothing else, graph teachers build their graph from ``joint.obsm['spatial']``. Stock
    SpatialMETA (config G) checkpoints load into a fresh ``ConditionalVAESTSM``.
    """
    import spatialmeta as smt
    import torch
    from spatialjepa.spatialmeta_backbone import rebuild_spatialjepa_model

    ckpt = torch.load(Path(ckpt_path), map_location=device)
    data = smt.util._classes.AnnDataJointSMST(joint.copy())
    if ckpt.get("model_class") == "spatialmeta.model.ConditionalVAESTSM":
        model = smt.model.ConditionalVAESTSM(
            data, device=device, reconstruction_method_sm="g", reconstruction_method_st="zinb"
        )
        model.load_state_dict(ckpt["state_dict"])
    else:
        meta = dict(ckpt, batch_keys=None, section_key=ckpt.get("section_key") or section_key)
        model = rebuild_spatialjepa_model(ckpt["state_dict"], meta, data, device=device)
    out = _encode_sequential(model)
    del model
    return out


def manuscript_dopamine(sample_id: str) -> pd.Series:
    """Dopamine as in the manuscript's Fig. 5 metrics (dopamine_embedding_benchmark.DOPA_LAYER).

    Total-count normalization to 1e4 over *all* MSI peaks (script 3's ``normalized``
    layer), independent of any fold's feature selection. Indexed by spot name.
    """
    import muon as mu

    mdata = mu.read_h5mu(_h5mu_path(base_section(sample_id)))
    mu.pp.intersect_obs(mdata)
    msi = mdata.mod["msi"][mdata.mod["rna"].obs_names]
    X = _dense(msi.X).astype(np.float64)
    names = canonical_msi_names()
    j = [names[str(mz)] for mz in msi.var_names].index(PRIMARY_METABOLITE)
    return pd.Series(_tic_normalize(X, 1e4)[:, j], index=msi.obs_names.astype(str), name="dopamine")


# ---------------------------------------------------------------------------
# PLS readout (manuscript Supp. Fig. S8 decoder, source-fitted)
# ---------------------------------------------------------------------------

class PLSReadout:
    """K-component PLS from a representation to the source MSI features.

    Targets follow ``train_rna_msi.py:1057-1067``: log1p of the model-space
    SM, z-scored (here with source statistics reused on the target) and clipped at 10
    (lower tail at -max).
    """

    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit_targets(self, sm_model_space: np.ndarray):
        """Source target statistics only (shared by every readout of a fold)."""
        Y = np.log1p(sm_model_space)
        self.y_mean_ = Y.mean(0)
        self.y_std_ = Y.std(0, ddof=1)
        self.keep_ = np.isfinite(self.y_std_) & (self.y_std_ > 0)
        return self

    def scaled_targets(self, sm_model_space: np.ndarray) -> np.ndarray:
        return self._scale(np.log1p(sm_model_space))

    def fit(self, Z: np.ndarray, sm_model_space: np.ndarray):
        from sklearn.cross_decomposition import PLSRegression

        self.fit_targets(sm_model_space)
        Y = self.scaled_targets(sm_model_space)
        self.pls_ = PLSRegression(n_components=self.n_components).fit(Z, Y)
        return self

    def _scale(self, Y):
        Y = (Y[:, self.keep_] - self.y_mean_[self.keep_]) / self.y_std_[self.keep_]
        Y = np.minimum(Y, 10.0)
        return np.maximum(Y, -np.nanmax(Y))

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self.pls_.predict(Z), dtype=np.float32)


def measured_eval_space(msi_adata, keep: np.ndarray | None = None) -> np.ndarray:
    """Measured MSI for evaluation: log1p(log1p(TIC) / source std), unclipped.

    Monotone in the TIC-normalized intensity (so rank metrics equal those on TIC) and on
    the same log scale as the PLS targets for Moran's I.
    """
    tic = _dense(msi_adata.layers["tic"]).astype(np.float64)
    std = msi_adata.var["source_sm_std"].to_numpy(float)
    Y = np.log1p(np.log1p(tic) / std)
    return Y if keep is None else Y[:, keep]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spatial_knn_graph(coords: np.ndarray, k: int = 6):
    """Symmetric spatial kNN connectivity (dopamine_nmf_signature.py:438)."""
    from sklearn.neighbors import kneighbors_graph

    A = kneighbors_graph(np.asarray(coords), n_neighbors=k, mode="connectivity", include_self=False)
    return A.maximum(A.T).tocsr()


def _z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def bivariate_morans_i(x: np.ndarray, y: np.ndarray, W) -> float:
    """(n / S0) * z_x . (W z_y) / (z_x . z_x)  (dopamine_nmf_signature.py:93)."""
    zx, zy = _z(x), _z(y)
    denom = zx @ zx
    if denom == 0:
        return float("nan")
    return float((len(zx) / W.sum()) * (zx @ (W @ zy)) / denom)


def _rank(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(v)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def partial_spearman(a: np.ndarray, b: np.ndarray, strata: np.ndarray) -> float:
    """Spearman after removing stratum means of the ranks (compartment-adjusted)."""
    ra, rb = _rank(a), _rank(b)
    strata = np.asarray(strata)
    for s in np.unique(strata):
        m = strata == s
        ra[m] -= ra[m].mean()
        rb[m] -= rb[m].mean()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def columnwise_spearman(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-column Spearman between two (n, p) matrices."""
    from scipy.stats import rankdata

    RA = rankdata(A, axis=0)
    RB = rankdata(B, axis=0)
    RA -= RA.mean(0)
    RB -= RB.mean(0)
    denom = np.sqrt((RA ** 2).sum(0) * (RB ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (RA * RB).sum(0) / denom, np.nan)


def lesion_auroc(score: np.ndarray, lesion: np.ndarray) -> float:
    """AUROC of ``score`` for intact (positive) vs lesioned spots."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(lesion) == "intact"
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, score))


def metabolite_metrics(pred: np.ndarray, meas: np.ndarray, obs: pd.DataFrame, W, region: str) -> dict:
    """All metrics for one predicted vs measured metabolite on one section."""
    in_region = obs["region"].astype(str).eq(REGION_LABEL[region]).to_numpy()
    lesion = obs["lesion"].astype(str).to_numpy()
    strata = (obs["lesion"].astype(str) + "|" + obs["region"].astype(str)).to_numpy()
    smooth = sp.diags(1.0 / (np.asarray(W.sum(1)).ravel() + 1.0)) @ (W + sp.identity(W.shape[0]))

    out = {
        "spearman_section": spearman(pred, meas),
        "spearman_region": spearman(pred[in_region], meas[in_region]),
        "spearman_intact_region": spearman(pred[in_region & (lesion == "intact")], meas[in_region & (lesion == "intact")]),
        "spearman_lesioned_region": spearman(pred[in_region & (lesion == "lesioned")], meas[in_region & (lesion == "lesioned")]),
        "spearman_partial": partial_spearman(pred, meas, strata),
        "pearson_section": float(np.corrcoef(pred, meas)[0, 1]) if np.ptp(pred) > 0 and np.ptp(meas) > 0 else float("nan"),
        "lesion_auroc_region": lesion_auroc(pred[in_region], lesion[in_region]),
        "bivariate_moran": bivariate_morans_i(pred, meas, W),
        "spearman_smoothed": spearman(smooth @ pred, smooth @ meas),
    }
    return out


def target_quality(meas: np.ndarray, tic: np.ndarray, obs: pd.DataFrame, W, region: str) -> dict:
    """Is the measured signal informative enough to evaluate on this section?"""
    in_region = obs["region"].astype(str).eq(REGION_LABEL[region]).to_numpy()
    lesion = obs["lesion"].astype(str).to_numpy()
    return {
        "n_spots": int(len(meas)),
        "n_region_spots": int(in_region.sum()),
        "frac_nonzero_section": float((tic > 0).mean()),
        "frac_nonzero_region": float((tic[in_region] > 0).mean()) if in_region.any() else float("nan"),
        "moran_measured": bivariate_morans_i(meas, meas, W),
        "lesion_auroc_measured_region": lesion_auroc(meas[in_region], lesion[in_region]),
    }
