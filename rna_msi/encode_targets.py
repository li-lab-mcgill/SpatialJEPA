#%% Project Spatial RNA-ATAC target data into the SMA-trained RNA latent/MSI space.
#
# The SMA SpatialJEPA model (script 3) is a joint RNA("ST")+metabolomics("SM") VAE.
# Spatial RNA-ATAC shares only the RNA modality with SMA, so we push the target RNA
# through the trained student and teacher. The student joint latent H["q_mu"] is
# used for the post-hoc PLS MSI decoder; the teacher RNA-only latent H["st"]["q_mu"]
# is used for the trained neural SM decoder. The SM block is filled with zeros in
# the model input.
#
# ST inputs are RAW counts: the model was trained on raw HVG counts (script 3's final
# normalize_total uses target_sum_ST=None, which skips ST), and encode() applies
# log(X_ST + 1) internally. So the target ST block is raw counts reindexed to the SMA
# gene set (zero-filled for genes absent from the target).

from __future__ import annotations
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.environ["SPATIALJEPA_CONFIG"])

import os
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
try:
    import mudata as mu
except ImportError as exc:
    raise ImportError(
        "encode_targets.py requires the mudata package to write "
        "trimodal .h5mu outputs. Run this script in the SMA environment with "
        "mudata installed."
    ) from exc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spatialmeta as smt
from anndata import AnnData

sys.path.insert(0, os.environ["SPATIALJEPA_RNA_MSI"])
from spatialjepa.spatialmeta_backbone import load_spatialjepa_model

SAMPLE_ID = ["V11T17-102_A1", "V11T17-102_C1", "V11T17-102_D1"]  # SMA sample/run whose trained student we project with
DECODER_MODE = "auto"         # "auto" -> PLS if present, otherwise neural fallback
BATCH_SIZE = 1024
MIN_ST_OVERLAP_FRACTION = 0.10
MSI_OBSM_KEY = "student_msi"
TEACHER_MSI_OBSM_KEY = "teacher_msi"
DOPAMINE_FEATURE = "msi:Dopamine"

# Optional env overrides, matching train_rna_msi.py: SPATIALJEPA_SAMPLE_IDS
# (comma-separated), SPATIALJEPA_RUN_TAG (model/projection dir suffix) and
# SPATIALJEPA_FIG_DIR (where the imputed-MSI spatial figures go).
if os.getenv("SPATIALJEPA_SAMPLE_IDS"):
    SAMPLE_ID = [s.strip() for s in os.environ["SPATIALJEPA_SAMPLE_IDS"].split(",") if s.strip()]
    SAMPLE_ID = SAMPLE_ID[0] if len(SAMPLE_ID) == 1 else SAMPLE_ID
SAMPLE_IDS = [SAMPLE_ID] if isinstance(SAMPLE_ID, str) else list(SAMPLE_ID)
PREFIX = SAMPLE_IDS[0].split("_")[0]
# same run naming as script 3: the sample for one section, the shared prefix for several
RUN_ID = (SAMPLE_IDS[0] if len(SAMPLE_IDS) == 1 else PREFIX) + os.getenv("SPATIALJEPA_RUN_TAG", "")

species = "human" if PREFIX == "V11T17-102" else "mouse"
ALIGNED_DATA_DIR = Path(os.getenv("SPATIALJEPA_DATA")) / "aligned_data"


def env_path(name: str, default: Path | str) -> Path:
    value = os.getenv(name)
    return Path(value) if value else Path(default)


# Species-specific spatial target RNA defaults (override with SPATIAL_TARGET_RNA_PATH /
# TARGET_RNA_PATH only when needed). Human path matches the CaH Xenium panel used for
# target-gene ranking in script 3.
if species == "mouse":
    SPATIAL_TARGET_RNA_PATH = ALIGNED_DATA_DIR / "source_rna_aligned_SCT.h5ad"
elif species == "human":
    SPATIAL_TARGET_RNA_PATH = (
        Path(os.getenv("SPATIALJEPA_DATA"))
        / "SEA_AD"
        / "sea_ad_cah"
        / "CaH_Xenium_final.2026-01-07.h5ad"
    )
else:
    raise ValueError(f"Unsupported species for spatial target RNA: {species!r}")

spatial_target_rna_path_env = os.getenv("SPATIAL_TARGET_RNA_PATH") or os.getenv("TARGET_RNA_PATH")
if spatial_target_rna_path_env:
    SPATIAL_TARGET_RNA_PATH = Path(spatial_target_rna_path_env)

TARGET_RNA_PATH = SPATIAL_TARGET_RNA_PATH  # Backwards-compatible metadata alias.
SPATIAL_TARGET_ATAC_PATH = env_path(
    "SPATIAL_TARGET_ATAC_PATH",
    ALIGNED_DATA_DIR / "source_atac_aligned.h5ad",
)
MULTIOME_TARGET_RNA_PATH = env_path(
    "MULTIOME_TARGET_RNA_PATH",
    ALIGNED_DATA_DIR / "target_rna_aligned_SCT.h5ad",
)
MULTIOME_TARGET_ATAC_PATH = env_path(
    "MULTIOME_TARGET_ATAC_PATH",
    ALIGNED_DATA_DIR / "target_atac_aligned.h5ad",
)

OUTPUT_DIR = Path(os.getenv("SPATIALJEPA_OUT"))
MODEL_DIR = OUTPUT_DIR / "spatialjepa_models" / RUN_ID
PROJ_DIR = OUTPUT_DIR / "spatialjepa_projection" / RUN_ID
PROJ_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RNA_PREFIX = "rna:"


def bare_feature_symbol(name) -> str:
    """Return a feature name without a leading modality prefix."""
    text = str(name).strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text


def feature_token(name) -> str:
    """Case-insensitive gene matching token shared by source and target schemas."""
    return bare_feature_symbol(name).upper()


def choose_decoder_mode(mode: str, pls_decoder_path: Path) -> str:
    mode = mode.lower()
    valid_modes = {"auto", "pls", "neural"}
    if mode not in valid_modes:
        raise ValueError(f"DECODER_MODE must be one of {sorted(valid_modes)}; got {mode!r}")
    if mode == "pls":
        if not pls_decoder_path.exists():
            raise FileNotFoundError(f"Requested PLS decoder is missing: {pls_decoder_path}")
        return "pls"
    if mode == "neural":
        return "neural"
    if pls_decoder_path.exists():
        return "pls"
    print(f"[WARN] PLS decoder missing at {pls_decoder_path}; using neural MSI decoder.")
    return "neural"


def matrix_rows_to_numpy(matrix, indices):
    rows = matrix[indices]
    if sp.issparse(rows):
        rows = rows.toarray()
    return np.asarray(rows, dtype=np.float32)


def batch_index_for_model(model, indices):
    batch_codes = getattr(model, "batch_codes", None)
    if batch_codes is None:
        return None

    idx = np.asarray(indices, dtype=int)
    codes = []
    for code in batch_codes:
        if torch.is_tensor(code):
            code_values = code.detach().cpu().numpy()
        else:
            code_values = np.asarray(code)
        codes.append(
            torch.as_tensor(code_values[idx], dtype=torch.long, device=model.device)
            .unsqueeze(1)
        )
    return torch.hstack(codes)


@torch.no_grad()
def encode_model(model, batch_size: int, *, decode_neural: bool = False):
    model.eval()
    z_joint, z_st, neural_msi = [], [], []

    for batch_idx in model.as_dataloader(batch_size=batch_size, shuffle=False):
        indices = batch_idx[0].detach().cpu().numpy()
        X_batch = torch.as_tensor(
            matrix_rows_to_numpy(model.X, indices),
            dtype=torch.float32,
            device=model.device,
        )
        H = model.encode(X_batch)
        z_joint.append(H["q_mu"].detach().cpu().numpy())
        z_st_batch = H["st"]["q_mu"]
        z_st.append(z_st_batch.detach().cpu().numpy())

        if decode_neural:
            z_decode = z_st_batch
            batch_index = batch_index_for_model(model, indices)
            if batch_index is not None:
                z_decode = torch.hstack([z_decode, batch_index.to(z_decode.device)])
            decoded = model.px_sm_scale_decoder(model.decoder_sm(z_decode.to(model.device)))
            neural_msi.append(decoded.detach().cpu().numpy())

    decoded_msi = np.vstack(neural_msi).astype(np.float32) if decode_neural else None
    return np.vstack(z_joint).astype(np.float32), np.vstack(z_st).astype(np.float32), decoded_msi


def decode_msi_pls(embedding: np.ndarray, pls_decoder_path: Path):
    payload = joblib.load(pls_decoder_path)
    pls = payload["model"]
    target_features = [str(f) for f in payload["target_features"]]
    expected_dim = getattr(pls, "n_features_in_", None)
    if expected_dim is not None and embedding.shape[1] != expected_dim:
        raise ValueError(
            f"PLS decoder expects {expected_dim} latent dimensions, "
            f"but student_X_emb has {embedding.shape[1]}."
        )

    msi_matrix = np.asarray(pls.predict(embedding), dtype=np.float32)
    if msi_matrix.ndim == 1:
        msi_matrix = msi_matrix[:, None]
    if msi_matrix.shape[1] != len(target_features):
        raise ValueError(
            f"PLS output has {msi_matrix.shape[1]} columns but "
            f"{len(target_features)} target features were recorded."
        )

    metadata = {
        "mode": "pls",
        "decoder_path": str(pls_decoder_path),
        "embedding_key": payload.get("embedding_key", "student_X_emb"),
        "target_transform": payload.get("target_transform", {}),
        "run_id": payload.get("run_id", SAMPLE_ID),
    }
    return msi_matrix, target_features, metadata


def attach_msi_output(
    adata,
    *,
    obsm_key: str,
    msi_matrix: np.ndarray,
    msi_feature_names: list[str],
    decoder_metadata: dict,
    gene_overlap: dict,
):
    if msi_matrix.shape != (adata.n_obs, len(msi_feature_names)):
        raise ValueError(
            f"{obsm_key} shape {msi_matrix.shape} does not match "
            f"(n_obs={adata.n_obs}, n_features={len(msi_feature_names)})."
        )

    adata.obsm[obsm_key] = msi_matrix
    adata.uns[f"{obsm_key}_feature_names"] = np.asarray(msi_feature_names, dtype=str)
    decoder_metadata["gene_overlap"] = gene_overlap
    adata.uns[f"{obsm_key}_decoder"] = decoder_metadata
    print(
        f"[INFO] {obsm_key}: decoded MSI matrix {msi_matrix.shape} "
        f"via {decoder_metadata['mode']}"
    )

    if DOPAMINE_FEATURE not in msi_feature_names:
        print(f"[WARN] {DOPAMINE_FEATURE} not found in {obsm_key} features.")
        return None

    dopamine_idx = msi_feature_names.index(DOPAMINE_FEATURE)
    dopamine_obs_key = f"{obsm_key}_Dopamine"
    adata.obs[dopamine_obs_key] = msi_matrix[:, dopamine_idx]
    print(f"[INFO] Wrote obs['{dopamine_obs_key}'] from {DOPAMINE_FEATURE}")
    return dopamine_obs_key


def write_msi_cache(
    *,
    target_label: str,
    obsm_key: str,
    msi_matrix: np.ndarray,
    msi_feature_names: list[str],
    dopamine_obs_key,
    decoder_metadata: dict,
    obs_names: pd.Index,
    target_rna_path: Path,
):
    npz_path = PROJ_DIR / f"{target_label}_{obsm_key}_imputed.npz"
    np.savez_compressed(
        npz_path,
        **{
            obsm_key: msi_matrix,
            "msi_feature_names": np.asarray(msi_feature_names, dtype=str),
            "obs_names": np.asarray(obs_names.tolist(), dtype=str),
        },
    )

    json_path = PROJ_DIR / f"{target_label}_{obsm_key}_imputed.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample_id": SAMPLE_ID,
                "target_label": target_label,
                "target_rna_path": str(target_rna_path),
                "msi_obsm_key": obsm_key,
                "msi_feature_names": msi_feature_names,
                "dopamine_feature": DOPAMINE_FEATURE,
                "dopamine_obs_key": dopamine_obs_key,
                "decoder": decoder_metadata,
            },
            handle,
            indent=2,
        )
    print(f"[INFO] Wrote MSI cache to {npz_path} and {json_path}")

#%% Load the trained feature space (template) and define reusable target helpers.

template = sc.read_h5ad(MODEL_DIR / "joint_adata.h5ad")
st_mask = (template.var["type"].values == "ST")
sm_mask = (template.var["type"].values == "SM")
st_cols = np.where(st_mask)[0]
# SMA RNA gene symbols, in the exact column order encoder_ST expects.
sma_genes = pd.Index([bare_feature_symbol(v) for v in template.var_names[st_mask]])
sma_tokens = pd.Index([feature_token(v) for v in template.var_names[st_mask]])
print(
    f"[INFO] Template: {template.shape} "
    f"({int(st_mask.sum())} ST + {int(sm_mask.sum())} SM features)"
)

student_ckpt_path = MODEL_DIR / "student.pt"
teacher_ckpt_path = MODEL_DIR / "teacher.pt"
pls_decoder_path = MODEL_DIR / "student_pls_decoder.joblib"
decoder_mode = choose_decoder_mode(DECODER_MODE, pls_decoder_path)
THESIS_FIG_DIR = env_path("SPATIALJEPA_FIG_DIR", f"{os.environ['SPATIALJEPA_MANUSCRIPT']}/figures")
THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_paired_rna_atac(
    *,
    target_label: str,
    rna_path: Path,
    atac_path: Path,
    spatial_target: bool,
):
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    if not rna.obs_names.is_unique or not atac.obs_names.is_unique:
        raise ValueError(f"{target_label}: RNA and ATAC obs_names must be unique.")

    rna.var_names_make_unique()
    atac.var_names_make_unique()

    shared_obs = rna.obs_names[rna.obs_names.isin(atac.obs_names)]
    if len(shared_obs) == 0:
        raise ValueError(f"{target_label}: RNA and ATAC share zero observations.")
    if len(shared_obs) != rna.n_obs or len(shared_obs) != atac.n_obs:
        print(
            f"[WARN] {target_label}: RNA/ATAC partial overlap; "
            f"keeping {len(shared_obs)} shared observations "
            f"(RNA={rna.n_obs}, ATAC={atac.n_obs})."
        )

    rna = rna[shared_obs].copy()
    atac = atac[shared_obs].copy()
    if not rna.obs_names.equals(atac.obs_names):
        raise RuntimeError(f"{target_label}: failed to align ATAC obs_names to RNA order.")

    if spatial_target:
        if "spatial" not in rna.obsm:
            raise KeyError(f"{target_label}: spatial target RNA is missing obsm['spatial'].")
        spatial = np.asarray(rna.obsm["spatial"], dtype=np.float32) * np.asarray(
            [1.0, -1.0],
            dtype=np.float32,
        )
        rna.obsm["spatial"] = spatial
        atac.obsm["spatial"] = spatial.copy()

    print(
        f"[INFO] {target_label}: RNA {rna.shape} from {rna_path.name}; "
        f"ATAC {atac.shape} from {atac_path.name}"
    )
    return rna, atac


def build_projection_joint(target_label: str, target_rna: AnnData):
    target_counts = target_rna.layers["counts"] if "counts" in target_rna.layers else target_rna.X

    target_token_to_idx = {}
    duplicate_target_tokens = set()
    for i, token in enumerate(feature_token(v) for v in target_rna.var_names):
        if token and token not in target_token_to_idx:
            target_token_to_idx[token] = i
        elif token:
            duplicate_target_tokens.add(token)

    idx = np.asarray([target_token_to_idx.get(token, -1) for token in sma_tokens], dtype=int)
    present = idx >= 0
    n_overlap = int(present.sum())
    overlap_fraction = n_overlap / max(len(sma_genes), 1)
    print(
        f"[INFO] {target_label}: gene overlap {n_overlap}/{len(sma_genes)} "
        f"SMA ST genes found ({100 * overlap_fraction:.1f}%)."
    )
    if duplicate_target_tokens:
        print(
            f"[INFO] {target_label}: duplicate target gene tokens ignored after first "
            f"occurrence: {sorted(duplicate_target_tokens)[:10]}"
        )
    missing_examples = list(sma_genes[~present][:10])
    print(f"[INFO] {target_label}: example missing genes (zero-filled): {missing_examples}")
    if overlap_fraction < MIN_ST_OVERLAP_FRACTION:
        raise ValueError(
            f"{target_label}: gene overlap {overlap_fraction:.3f} is below "
            f"MIN_ST_OVERLAP_FRACTION={MIN_ST_OVERLAP_FRACTION:.3f}."
        )

    gene_overlap = {
        "n_overlap": n_overlap,
        "n_reference_st": int(len(sma_genes)),
        "overlap_fraction": float(overlap_fraction),
        "min_overlap_fraction": float(MIN_ST_OVERLAP_FRACTION),
        "missing_examples": [str(v) for v in missing_examples],
        "duplicate_target_token_examples": sorted(duplicate_target_tokens)[:10],
    }

    n_obs = target_rna.n_obs
    X_st = np.zeros((n_obs, len(sma_genes)), dtype=np.float32)
    src = target_counts[:, idx[present]]
    X_st[:, present] = src.toarray() if sp.issparse(src) else np.asarray(src)

    X = np.zeros((n_obs, template.n_vars), dtype=np.float32)
    X[:, st_cols] = X_st

    target_joint = smt.util._classes.AnnDataJointSMST(
        AnnData(X=X, obs=target_rna.obs.copy(), var=template.var.copy())
    )
    if "spatial" in target_rna.obsm:
        target_joint.obsm["spatial"] = np.asarray(target_rna.obsm["spatial"]).copy()
    target_joint.layers["counts"] = X.copy()
    target_joint.uns["target_label"] = target_label
    print(
        f"[INFO] {target_label}: projection joint adata {target_joint.shape}; "
        f"batches: "
        f"{target_joint.obs['batch'].value_counts().to_dict() if 'batch' in target_joint.obs else 'n/a'}"
    )
    return target_joint, gene_overlap


def run_student_projection(
    *,
    target_label: str,
    target_joint,
    gene_overlap: dict,
    target_rna_path: Path,
):
    student_model = load_spatialjepa_model(student_ckpt_path, target_joint, device=DEVICE)
    Z_joint, Z_st, neural_msi = encode_model(
        student_model,
        BATCH_SIZE,
        decode_neural=(decoder_mode == "neural"),
    )
    assert not np.isnan(Z_joint).any(), f"{target_label} student: NaNs in joint embedding"
    assert not np.isnan(Z_st).any(), f"{target_label} student: NaNs in RNA-only embedding"
    target_joint.obsm["student_X_emb"] = Z_joint
    target_joint.obsm["student_X_emb_st"] = Z_st
    print(
        f"[INFO] {target_label} student: joint embedding {Z_joint.shape}; "
        f"RNA-only embedding {Z_st.shape}"
    )

    if decoder_mode == "pls":
        msi_matrix, msi_feature_names, decoder_metadata = decode_msi_pls(
            target_joint.obsm["student_X_emb"],
            pls_decoder_path,
        )
    else:
        if neural_msi is None:
            raise RuntimeError("Internal error: neural MSI decode was requested but not computed.")
        msi_matrix = neural_msi
        msi_feature_names = [str(v) for v in template.var_names[sm_mask]]
        decoder_metadata = {
            "mode": "neural",
            "decoder_path": str(student_ckpt_path),
            "embedding_key": "student_X_emb_st",
            "target_transform": {"output_space": "student_neural_sm_scale"},
            "run_id": SAMPLE_ID,
        }

    dopamine_obs_key = attach_msi_output(
        target_joint,
        obsm_key=MSI_OBSM_KEY,
        msi_matrix=msi_matrix,
        msi_feature_names=msi_feature_names,
        decoder_metadata=decoder_metadata,
        gene_overlap=gene_overlap,
    )

    del student_model
    torch.cuda.empty_cache()

    write_msi_cache(
        target_label=target_label,
        obsm_key=MSI_OBSM_KEY,
        msi_matrix=msi_matrix,
        msi_feature_names=msi_feature_names,
        dopamine_obs_key=dopamine_obs_key,
        decoder_metadata=decoder_metadata,
        obs_names=target_joint.obs_names,
        target_rna_path=target_rna_path,
    )
    return {
        "matrix": msi_matrix,
        "feature_names": msi_feature_names,
        "decoder_metadata": decoder_metadata,
        "dopamine_obs_key": dopamine_obs_key,
    }


def run_teacher_projection(
    *,
    target_label: str,
    target_joint,
    gene_overlap: dict,
    target_rna_path: Path,
):
    teacher_model = load_spatialjepa_model(teacher_ckpt_path, target_joint, device=DEVICE)
    Z_teacher_joint, Z_teacher_st, teacher_msi = encode_model(
        teacher_model,
        BATCH_SIZE,
        decode_neural=True,
    )
    assert not np.isnan(Z_teacher_joint).any(), f"{target_label} teacher: NaNs in joint embedding"
    assert not np.isnan(Z_teacher_st).any(), f"{target_label} teacher: NaNs in RNA-only embedding"
    target_joint.obsm["teacher_X_emb"] = Z_teacher_joint
    target_joint.obsm["teacher_X_emb_st"] = Z_teacher_st
    print(
        f"[INFO] {target_label} teacher: joint embedding {Z_teacher_joint.shape}; "
        f"RNA-only embedding {Z_teacher_st.shape}"
    )

    teacher_msi_feature_names = [str(v) for v in template.var_names[sm_mask]]
    teacher_decoder_metadata = {
        "mode": "neural",
        "decoder_path": str(teacher_ckpt_path),
        "embedding_key": "teacher_X_emb_st",
        "target_transform": {"output_space": "teacher_neural_sm_scale"},
        "run_id": SAMPLE_ID,
    }
    dopamine_obs_key = attach_msi_output(
        target_joint,
        obsm_key=TEACHER_MSI_OBSM_KEY,
        msi_matrix=teacher_msi,
        msi_feature_names=teacher_msi_feature_names,
        decoder_metadata=teacher_decoder_metadata,
        gene_overlap=gene_overlap,
    )

    del teacher_model
    torch.cuda.empty_cache()

    write_msi_cache(
        target_label=target_label,
        obsm_key=TEACHER_MSI_OBSM_KEY,
        msi_matrix=teacher_msi,
        msi_feature_names=teacher_msi_feature_names,
        dopamine_obs_key=dopamine_obs_key,
        decoder_metadata=teacher_decoder_metadata,
        obs_names=target_joint.obs_names,
        target_rna_path=target_rna_path,
    )
    return {
        "matrix": teacher_msi,
        "feature_names": teacher_msi_feature_names,
        "decoder_metadata": teacher_decoder_metadata,
        "dopamine_obs_key": dopamine_obs_key,
    }


def embed_and_plot(target_label: str, target_joint, name: str, dopamine_obs_key=None) -> None:
    emb_key = f"{name}_X_emb_st"
    neigh_key = f"{target_label}_{name}_neighbors"
    umap_key = f"{name}_umap"
    cluster_key = f"{name}_leiden"
    file_prefix = "" if target_label == "spatial_target" else f"{target_label}_"

    sc.pp.neighbors(target_joint, use_rep=emb_key, n_neighbors=15, key_added=neigh_key)
    sc.tl.umap(target_joint, min_dist=0.3, neighbors_key=neigh_key)
    target_joint.obsm[umap_key] = target_joint.obsm["X_umap"].copy()
    sc.tl.leiden(target_joint, key_added=cluster_key, neighbors_key=neigh_key)

    color = [cluster_key]
    for c in ("RNA_clusters", "ATAC_clusters", "batch", dopamine_obs_key):
        if c is not None and c in target_joint.obs:
            color.append(c)
    fig = sc.pl.embedding(
        target_joint,
        basis=umap_key,
        color=color,
        ncols=2,
        wspace=0.3,
        show=False,
        return_fig=True,
    )
    umap_path = PROJ_DIR / f"{file_prefix}{name}_umap.png"
    fig.savefig(umap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] {target_label} {name}: wrote {umap_path}")

    if "spatial" not in target_joint.obsm:
        return
    fig_spatial = sc.pl.embedding(
        target_joint,
        basis="spatial",
        color=[c for c in color if c != "batch"],
        show=False,
        return_fig=True,
        s=60,
    )
    spatial_path = PROJ_DIR / f"{file_prefix}{name}_spatial.png"
    fig_spatial.savefig(spatial_path, dpi=150, bbox_inches="tight")
    plt.close(fig_spatial)
    print(f"[INFO] {target_label} {name}: wrote {spatial_path}")


def make_msi_adata(
    *,
    target_joint,
    msi_matrix: np.ndarray,
    msi_feature_names: list[str],
    decoder_metadata: dict,
    projection_umap_key=None,
    projection_obsm_key=None,
):
    obsm = {}
    if projection_umap_key is not None and projection_umap_key in target_joint.obsm:
        obsm[projection_obsm_key or projection_umap_key] = np.asarray(
            target_joint.obsm[projection_umap_key]
        )
    if "spatial" in target_joint.obsm:
        obsm["spatial"] = np.asarray(target_joint.obsm["spatial"])

    msi_adata = AnnData(
        X=np.asarray(msi_matrix, dtype=np.float32),
        obs=target_joint.obs.copy(),
        var=pd.DataFrame(index=pd.Index(msi_feature_names, dtype=str)),
        obsm=obsm,
    )
    msi_adata.uns["decoder"] = decoder_metadata
    return msi_adata


def add_msi_umap(adata: AnnData, label: str) -> None:
    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    if n_comps < 2:
        print(f"[WARN] {label}: skipping MSI UMAP because n_comps would be {n_comps}.")
        return
    sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)


def plot_spatial_msi_figure(adata: AnnData, label: str, filename: str) -> None:
    if "spatial" not in adata.obsm:
        return
    color = []
    if DOPAMINE_FEATURE in adata.var_names:
        color.append(DOPAMINE_FEATURE)
    for key in ("RNA_clusters", "ATAC_clusters"):
        if key in adata.obs:
            color.append(key)
    if not color:
        print(f"[WARN] {label}: no available colors for MSI spatial plot.")
        return

    fig = sc.pl.embedding(
        adata,
        basis="spatial",
        color=color,
        size=60,
        ncols=len(color),
        show=False,
        return_fig=True,
    )
    fig_path = THESIS_FIG_DIR / filename
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {label} MSI spatial figure: {fig_path}")


def write_trimodal_mudata(
    *,
    target_label: str,
    rna: AnnData,
    atac: AnnData,
    student_msi: AnnData,
    teacher_msi: AnnData = None,
):
    modalities = {
        "rna": rna.copy(),
        "atac": atac.copy(),
        "msi_student": student_msi.copy(),
    }
    if teacher_msi is not None:
        modalities["msi_teacher"] = teacher_msi.copy()

    canonical_obs = rna.obs_names
    for mod_name, adata in modalities.items():
        if not adata.obs_names.equals(canonical_obs):
            raise ValueError(f"{target_label}: modality {mod_name!r} obs_names are not aligned.")

    kwargs = {
        "obs": rna.obs.copy(),
        "uns": {
            "sample_id": SAMPLE_ID,
            "target_label": target_label,
            "spatialjepa_model_dir": str(MODEL_DIR),
        },
    }
    if "spatial" in rna.obsm:
        kwargs["obsm"] = {"spatial": np.asarray(rna.obsm["spatial"]).copy()}

    mdata = mu.MuData(modalities, **kwargs)
    out_path = PROJ_DIR / f"{target_label}_rna_atac_msi.h5mu"
    mdata.write_h5mu(out_path)
    print(f"[INFO] Saved trimodal {target_label} MuData to {out_path}")
    return out_path


def project_target(
    *,
    target_label: str,
    rna_path: Path,
    atac_path: Path,
    spatial_target: bool,
    run_teacher: bool,
    embedded_h5ad_path: Path,
    write_spatial_figures: bool = False,
):
    rna, atac = load_paired_rna_atac(
        target_label=target_label,
        rna_path=rna_path,
        atac_path=atac_path,
        spatial_target=spatial_target,
    )
    target_joint, gene_overlap = build_projection_joint(target_label, rna)
    target_joint.uns["target_rna_path"] = str(rna_path)
    target_joint.uns["target_atac_path"] = str(atac_path)

    student_result = run_student_projection(
        target_label=target_label,
        target_joint=target_joint,
        gene_overlap=gene_overlap,
        target_rna_path=rna_path,
    )
    if spatial_target:
        embed_and_plot(target_label, target_joint, "student", student_result["dopamine_obs_key"])

    teacher_result = None
    if run_teacher:
        teacher_result = run_teacher_projection(
            target_label=target_label,
            target_joint=target_joint,
            gene_overlap=gene_overlap,
            target_rna_path=rna_path,
        )
        embed_and_plot(target_label, target_joint, "teacher", teacher_result["dopamine_obs_key"])

    student_msi_adata = make_msi_adata(
        target_joint=target_joint,
        msi_matrix=student_result["matrix"],
        msi_feature_names=student_result["feature_names"],
        decoder_metadata=student_result["decoder_metadata"],
        projection_umap_key="student_umap",
        projection_obsm_key="ST_student_umap",
    )

    teacher_msi_adata = None
    if teacher_result is not None:
        teacher_msi_adata = make_msi_adata(
            target_joint=target_joint,
            msi_matrix=teacher_result["matrix"],
            msi_feature_names=teacher_result["feature_names"],
            decoder_metadata=teacher_result["decoder_metadata"],
            projection_umap_key="teacher_umap",
            projection_obsm_key="ST_teacher_umap",
        )

    if write_spatial_figures:
        add_msi_umap(student_msi_adata, f"{target_label} student MSI")
        plot_spatial_msi_figure(
            student_msi_adata,
            f"{target_label} student",
            "p22_student_imputed_msi_spatial.png",
        )
        if teacher_msi_adata is not None:
            add_msi_umap(teacher_msi_adata, f"{target_label} teacher MSI")
            plot_spatial_msi_figure(
                teacher_msi_adata,
                f"{target_label} teacher",
                "p22_teacher_imputed_msi_spatial.png",
            )

    mudata_path = write_trimodal_mudata(
        target_label=target_label,
        rna=rna,
        atac=atac,
        student_msi=student_msi_adata,
        teacher_msi=teacher_msi_adata,
    )

    if teacher_result is not None:
        Zs, Zt = target_joint.obsm["student_X_emb_st"], target_joint.obsm["teacher_X_emb_st"]
        per_dim_r = [float(np.corrcoef(Zs[:, d], Zt[:, d])[0, 1]) for d in range(Zs.shape[1])]
        print(f"[INFO] {target_label}: student-vs-teacher per-latent-dim Pearson r:")
        print("       " + ", ".join(f"{r:.3f}" for r in per_dim_r))
        print(f"[INFO] {target_label}: mean |r| across dims: {np.nanmean(np.abs(per_dim_r)):.3f}")

    target_joint.write_h5ad(embedded_h5ad_path)
    print(f"[INFO] Saved embedded {target_label} target to {embedded_h5ad_path}")
    return {
        "rna": rna,
        "atac": atac,
        "target_joint": target_joint,
        "mudata_path": mudata_path,
        "embedded_h5ad_path": embedded_h5ad_path,
    }


#%% Project the spatial p22 target and write RNA+ATAC+student/teacher MSI.

spatial_target_result = project_target(
    target_label="spatial_target",
    rna_path=SPATIAL_TARGET_RNA_PATH,
    atac_path=SPATIAL_TARGET_ATAC_PATH,
    spatial_target=True,
    run_teacher=True,
    embedded_h5ad_path=PROJ_DIR / "target_spatial_atac_rna_embedded.h5ad",
    write_spatial_figures=True,
)

#%% Project the non-spatial multiome target and write RNA+ATAC+student MSI only.

multiome_target_result = project_target(
    target_label="multiome_target",
    rna_path=MULTIOME_TARGET_RNA_PATH,
    atac_path=MULTIOME_TARGET_ATAC_PATH,
    spatial_target=False,
    run_teacher=False,
    embedded_h5ad_path=PROJ_DIR / "multiome_target_rna_embedded.h5ad",
)

# %%
