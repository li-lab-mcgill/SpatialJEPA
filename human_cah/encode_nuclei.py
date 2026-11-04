#!/usr/bin/env python
"""Batch worker for SEA-AD cell-level spatialMETA embeddings.

Run this script with the spatialMETA conda environment, not the SEA-AD/Tangram
environment. Inputs are staged AnnData files and outputs are compact NPZ
embedding files keyed by obs_names.
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
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_config")
os.umask(0o002)

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


BYTES_PER_GB = 1024**3
DEFAULT_MIN_MATCH_FRACTION = 0.10
BAKLAVAS_SMA_DIR = Path(
    os.environ["SPATIALJEPA_RNA_MSI"]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_progress(message: str) -> None:
    print(f"{utc_now()} [spatialMETA] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def atomic_write_h5ad(path: Path, adata: ad.AnnData) -> None:
    """Write AnnData atomically so interrupted jobs cannot leave valid-looking files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp.unlink(missing_ok=True)
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def cleanup_staging(handoff_root: Path, *, older_than_hours: float, failed_only: bool) -> int:
    staging_root = handoff_root / "staging"
    if not staging_root.exists():
        return 0

    cutoff = time.time() - older_than_hours * 3600
    removed = 0
    for child in staging_root.iterdir():
        if not child.is_dir():
            continue
        if failed_only and not (child / ".failed").exists():
            continue
        if child.stat().st_mtime > cutoff:
            continue
        shutil.rmtree(child)
        removed += 1
    return removed


def load_reference_schema(model_dir: Path) -> dict[str, Any]:
    ref_path = model_dir / "joint_adata.h5ad"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing spatialMETA reference data: {ref_path}")

    ref = ad.read_h5ad(ref_path, backed="r")
    try:
        var = ref.var.copy()
        var_names = pd.Index(ref.var_names.astype(str))
        if "type" not in var.columns:
            raise ValueError(f"{ref_path} is missing required var['type'] column.")
        batch_categories: dict[str, list[str]] = {}
        for key in ref.obs.columns:
            if key == "section":
                batch_categories[key] = [
                    str(x) for x in pd.Categorical(ref.obs[key]).categories
                ]
    finally:
        ref.file.close()

    return {
        "reference_path": str(ref_path),
        "var": var,
        "var_names": var_names,
        "var_types": var["type"].astype(str).to_numpy(),
        "batch_categories": batch_categories,
    }


def feature_tokens(value: Any) -> set[str]:
    text = str(value).strip()
    if not text or text == "nan":
        return set()

    tokens = {text.upper()}
    if ":" in text:
        tokens.add(text.split(":", 1)[1].upper())
    else:
        tokens.add(f"RNA:{text}".upper())
        tokens.add(f"rna:{text}".upper())
    return tokens


def source_feature_lookup(adata: ad.AnnData) -> dict[str, int]:
    lookup: dict[str, int] = {}
    candidate_columns = [
        "gene_symbol",
        "gene_name",
        "feature_name",
        "feature_id",
        "symbol",
        "name",
    ]
    for idx, name in enumerate(adata.var_names):
        for token in feature_tokens(name):
            lookup.setdefault(token, idx)
        for col in candidate_columns:
            if col in adata.var.columns:
                for token in feature_tokens(adata.var.iloc[idx][col]):
                    lookup.setdefault(token, idx)
    return lookup


def spatial_coordinates_for_output(adata: ad.AnnData) -> tuple[np.ndarray, str]:
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"]), "obsm['spatial']"
    for x_col, y_col in [("x", "y"), ("X", "Y"), ("x_centroid", "y_centroid")]:
        if x_col in adata.obs.columns and y_col in adata.obs.columns:
            coords = adata.obs[[x_col, y_col]].to_numpy(dtype=float)
            return coords, f"obs[{x_col!r}, {y_col!r}]"
    return np.zeros((adata.n_obs, 2), dtype=np.float32), "zeros"


def align_to_reference(
    source: ad.AnnData,
    schema: dict[str, Any],
    *,
    batch_keys: list[str] | None,
    batch_value: str | None,
    min_match_fraction: float,
) -> tuple[ad.AnnData, dict[str, Any]]:
    target_var_names: pd.Index = schema["var_names"]
    target_var_types = np.asarray(schema["var_types"], dtype=str)
    source_lookup = source_feature_lookup(source)

    target_indices: list[int] = []
    source_indices: list[int] = []
    missing_transcriptomic_targets = 0

    for target_idx, target_name in enumerate(target_var_names):
        matched_source_idx = None
        for token in feature_tokens(target_name):
            matched_source_idx = source_lookup.get(token)
            if matched_source_idx is not None:
                break
        if matched_source_idx is not None:
            target_indices.append(target_idx)
            source_indices.append(matched_source_idx)
        elif target_var_types[target_idx] == "ST":
            missing_transcriptomic_targets += 1

    n_target = len(target_var_names)
    n_st_targets = int(np.sum(target_var_types == "ST"))
    n_matches = len(source_indices)
    match_fraction = n_matches / max(n_st_targets, 1)
    if match_fraction < min_match_fraction:
        raise ValueError(
            "Feature overlap with spatialMETA ST reference is too low: "
            f"{n_matches}/{n_st_targets} ({match_fraction:.3f}) matched; "
            f"minimum is {min_match_fraction:.3f}."
        )

    target_indices_arr = np.asarray(target_indices, dtype=np.int64)
    source_indices_arr = np.asarray(source_indices, dtype=np.int64)
    if sparse.issparse(source.X):
        source_slice = source.X[:, source_indices_arr].tocoo()
        aligned_x = sparse.csr_matrix(
            (
                source_slice.data.astype(np.float32, copy=False),
                (source_slice.row, target_indices_arr[source_slice.col]),
            ),
            shape=(source.n_obs, n_target),
            dtype=np.float32,
        )
    else:
        aligned_x = np.zeros((source.n_obs, n_target), dtype=np.float32)
        if len(source_indices_arr):
            aligned_x[:, target_indices_arr] = np.asarray(
                source.X[:, source_indices_arr],
                dtype=np.float32,
            )

    obs = source.obs.copy()
    batch_fallbacks: dict[str, str] = {}
    for key in batch_keys or []:
        categories = schema["batch_categories"].get(key)
        if not categories:
            continue
        selected = batch_value if batch_value in categories else categories[0]
        obs[key] = pd.Categorical([selected] * source.n_obs, categories=categories)
        if batch_value != selected:
            batch_fallbacks[key] = selected

    aligned = ad.AnnData(
        X=aligned_x,
        obs=obs,
        var=schema["var"].copy(),
    )
    coords, spatial_source = spatial_coordinates_for_output(source)
    aligned.obsm["spatial"] = coords

    diagnostics = {
        "source_n_obs": int(source.n_obs),
        "source_n_vars": int(source.n_vars),
        "target_n_vars": int(n_target),
        "target_st_vars": int(n_st_targets),
        "matched_features": int(n_matches),
        "missing_transcriptomic_targets": int(missing_transcriptomic_targets),
        "dropped_source_features": int(source.n_vars - len(set(source_indices))),
        "matched_fraction_of_st": float(match_fraction),
        "batch_fallbacks": batch_fallbacks,
        "spatial_coordinates_source": spatial_source,
    }
    return aligned, diagnostics


def import_spatialjepa_model():
    try:
        if str(BAKLAVAS_SMA_DIR) not in sys.path:
            sys.path.insert(0, str(BAKLAVAS_SMA_DIR))
        import torch
        from spatialmeta.util._classes import AnnDataJointSMST
        from spatialjepa.spatialmeta_backbone import load_spatialjepa_model
    except Exception as exc:  # noqa: BLE001 - surface import/runtime issues clearly
        raise RuntimeError(
            "Could not import BAKLAVAS SpatialJEPA machinery. Ensure this worker is "
            "run with the spatialMETA env, torch_geometric is installed, and "
            f"BAKLAVAS_SMA_DIR points at the SMA scripts directory ({BAKLAVAS_SMA_DIR})."
        ) from exc
    return torch, AnnDataJointSMST, load_spatialjepa_model


def load_checkpoint_metadata(path: Path, torch_module: Any) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing spatialMETA checkpoint: {path}")
    checkpoint = torch_module.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a spatialMETA checkpoint dict with state_dict.")
    return checkpoint


def load_model_for_adata(
    aligned: ad.AnnData,
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    torch_module, joint_cls, load_spatialjepa_model = import_spatialjepa_model()
    checkpoint = load_checkpoint_metadata(checkpoint_path, torch_module)
    try:
        model = load_spatialjepa_model(
            checkpoint_path,
            joint_cls(aligned),
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 - preserve the source loader as authority
        raise RuntimeError(
            f"BAKLAVAS spatialmeta_backbone.load_spatialjepa_model failed for "
            f"{checkpoint_path.name}."
        ) from exc

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "n_latent": int(checkpoint.get("n_latent", 10)),
        "hidden_stacks": checkpoint.get("hidden_stacks", [128]),
        "batch_keys": checkpoint.get("batch_keys"),
        "baklavas_sma_dir": str(BAKLAVAS_SMA_DIR),
        "model_loader": "spatialmeta_backbone.load_spatialjepa_model",
        "ignored_state_keys": [
            key for key in checkpoint["state_dict"] if key.endswith(".edge_index")
        ],
    }
    return model, metadata


def get_spatialjepa_latents_and_msi_decoding(
    model: Any,
    *,
    batch_size: int,
    show_progress: bool,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project RNA-only target data and decode MSI means from H['st']['q_mu']."""
    import torch

    model.eval()
    dataloader = model.as_dataloader(batch_size=batch_size, shuffle=False)
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    progress_every = max(1, total_batches // 10) if total_batches else None
    if show_progress:
        from tqdm import tqdm

        dataloader = tqdm(dataloader, desc="ST latent + MSI decoding")

    st_zs = []
    joint_zs = []
    msi = []
    for batch_idx, batch in enumerate(dataloader, start=1):
        indices = batch[0].detach().cpu().numpy()
        if sparse.issparse(model.adata.X):
            x_batch = model.adata.X[indices].toarray()
        else:
            x_batch = np.asarray(model.adata.X[indices])
        x_batch = torch.as_tensor(
            np.asarray(x_batch),
            dtype=torch.float32,
            device=model.device,
        )
        if model.batch_codes is not None:
            batch_index = [
                torch.as_tensor(code[indices], dtype=torch.long, device=model.device)
                .unsqueeze(1)
                for code in model.batch_codes
            ]
            batch_index = torch.hstack(batch_index)
        else:
            batch_index = None

        with torch.no_grad():
            h = model.encode(x_batch)
        try:
            st_z = h["st"]["q_mu"]
            joint_z = h["q_mu"]
        except KeyError as exc:
            raise RuntimeError(
                "SpatialJEPA model did not return H['st']['q_mu'] and H['q_mu']."
            ) from exc
        z_for_decoder = st_z
        if batch_index is not None:
            z_for_decoder = torch.hstack([z_for_decoder, batch_index])
        with torch.no_grad():
            px_sm = model.decoder_sm(z_for_decoder.to(model.device))
            px_sm_scale = model.px_sm_scale_decoder(px_sm)
        st_zs.append(st_z.detach().cpu().numpy())
        joint_zs.append(joint_z.detach().cpu().numpy())
        msi.append(px_sm_scale.detach().cpu().numpy())
        if total_batches and (
            batch_idx == 1
            or batch_idx == total_batches
            or (progress_every and batch_idx % progress_every == 0)
        ):
            log_progress(f"{progress_label}: batch {batch_idx}/{total_batches}")
    return np.vstack(st_zs), np.vstack(joint_zs), np.vstack(msi)


def embed_aligned(
    aligned: ad.AnnData,
    checkpoint_path: Path,
    *,
    device: str,
    batch_size: int,
    quiet: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    model, metadata = load_model_for_adata(aligned, checkpoint_path, device=device)
    log_progress(
        f"{checkpoint_path.name}: loaded model on {device}; "
        f"n_obs={aligned.n_obs:,} batch_size={batch_size}"
    )
    embedding, pls_embedding, decoded_msi = get_spatialjepa_latents_and_msi_decoding(
        model,
        batch_size=batch_size,
        show_progress=not quiet,
        progress_label=checkpoint_path.name,
    )
    log_progress(f"{checkpoint_path.name}: finished embedding")
    metadata["embedding_latent"] = "st.q_mu"
    metadata["pls_decoder_embedding_latent"] = "q_mu"
    metadata["decoded_msi_from"] = "st.q_mu -> decoder_sm -> px_sm_scale"
    return (
        np.asarray(embedding, dtype=np.float32),
        np.asarray(pls_embedding, dtype=np.float32),
        np.asarray(decoded_msi, dtype=np.float32),
        metadata,
    )


def patch_sklearn_pls_compat() -> None:
    """Patch old sklearn/scipy combos enough to unpickle newer PLSRegression."""
    try:
        import scipy.linalg as scipy_linalg

        if not hasattr(scipy_linalg, "pinv2"):
            scipy_linalg.pinv2 = scipy_linalg.pinv
    except Exception:
        pass


def load_spatialmeta_pls_decoder(pls_decoder_path: str | Path) -> dict[str, Any]:
    """Load the BAKLAVAS student PLS decoder payload for direct script use."""
    path = Path(pls_decoder_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing PLS decoder: {path}")

    patch_sklearn_pls_compat()
    try:
        import joblib
    except Exception as exc:  # noqa: BLE001 - joblib is required only for PLS mode
        raise RuntimeError("Could not import joblib to load the PLS decoder.") from exc

    try:
        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - invalid persisted model should be explicit
        raise RuntimeError(f"Failed to load PLS decoder: {path}") from exc

    required = {"model", "embedding_key", "target_features", "target_transform"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(
            f"{path} is not a BAKLAVAS PLS decoder payload with keys {sorted(required)}."
        )

    model = payload["model"]
    if not hasattr(model, "predict"):
        raise ValueError(f"{path} payload['model'] does not expose predict().")
    if not hasattr(model, "transform"):
        raise ValueError(f"{path} payload['model'] does not expose transform().")

    target_features = [str(feature) for feature in payload["target_features"]]
    if not target_features:
        raise ValueError(f"{path} has no target_features.")

    return payload


def pls_decoder_metadata(
    payload: dict[str, Any],
    pls_decoder_path: str | Path,
) -> dict[str, Any]:
    target_features = [str(feature) for feature in payload["target_features"]]
    model = payload["model"]
    return {
        "active": "pls",
        "pls_decoder_path": str(pls_decoder_path),
        "pls_decoder_sha256": sha256_file(Path(pls_decoder_path)),
        "embedding_key": str(payload["embedding_key"]),
        "target_features": target_features,
        "target_feature_count": len(target_features),
        "target_transform": payload["target_transform"],
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
    }


def transform_pls_components(
    decoder_payload: dict[str, Any],
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = decoder_payload["model"]
    embedding = np.asarray(embedding, dtype=np.float32)
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != embedding.shape[1]:
        raise ValueError(
            "PLS decoder input dimension mismatch: "
            f"model expects {int(model.n_features_in_)}, got {embedding.shape[1]}."
        )
    components = np.asarray(model.transform(embedding), dtype=np.float32)
    component_names = np.asarray(
        [f"PLS_{i}" for i in range(components.shape[1])],
        dtype=str,
    )
    return components, component_names


def load_msi_decoder(
    model_dir: Path,
    *,
    requested: str,
    pls_decoder_path: Path | None,
) -> dict[str, Any]:
    requested = requested.lower()
    if requested not in {"neural", "pls"}:
        raise ValueError("--msi-decoder must be either 'neural' or 'pls'.")

    path = pls_decoder_path or model_dir / "student_pls_decoder.joblib"
    metadata: dict[str, Any] = {
        "requested": requested,
        "active": "neural",
    }
    if requested == "neural":
        return {"metadata": metadata, "payload": None}

    metadata["pls_decoder_path"] = str(path)
    if not path.exists():
        metadata["fallback_reason"] = "missing_pls_decoder"
        return {"metadata": metadata, "payload": None}

    payload = load_spatialmeta_pls_decoder(path)
    metadata.update(
        pls_decoder_metadata(payload, path)
    )
    return {"metadata": metadata, "payload": payload}


def decoder_metadata_matches(metadata: dict[str, Any], decoder_info: dict[str, Any]) -> bool:
    cached = metadata.get("msi_decoder")
    expected = decoder_info["metadata"]
    if not isinstance(cached, dict):
        return False
    if (
        cached.get("requested") != expected.get("requested")
        or cached.get("active") != expected.get("active")
    ):
        return False
    if expected.get("active") == "pls":
        return cached.get("pls_decoder_sha256") == expected.get("pls_decoder_sha256")
    return cached.get("fallback_reason") == expected.get("fallback_reason")


def predict_pls_msi(
    decoder_info: dict[str, Any],
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    payload = decoder_info["payload"]
    if payload is None:
        raise ValueError("PLS MSI prediction requested without a loaded PLS payload.")

    model = payload["model"]
    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != embedding.shape[1]:
        raise ValueError(
            "PLS decoder input dimension mismatch: "
            f"model expects {int(model.n_features_in_)}, got {embedding.shape[1]}."
        )

    predicted = predict_pls_regression(model, np.asarray(embedding, dtype=np.float32))
    if predicted.ndim == 1:
        predicted = predicted.reshape(-1, 1)

    target_features = np.asarray(
        [str(feature) for feature in payload["target_features"]],
        dtype=str,
    )
    if predicted.shape != (embedding.shape[0], len(target_features)):
        raise ValueError(
            "PLS decoder output shape mismatch: "
            f"expected {(embedding.shape[0], len(target_features))}, got {predicted.shape}."
        )
    return predicted, target_features


def predict_pls_regression(model: Any, embedding: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(model.predict(embedding), dtype=np.float32)
    except ValueError as exc:
        coef = getattr(model, "coef_", None)
        if coef is None:
            raise

        coef = np.asarray(coef)
        n_features = embedding.shape[1]
        x_mean = np.asarray(getattr(model, "_x_mean", np.zeros(n_features)))
        centered = embedding - x_mean

        if coef.ndim == 2 and coef.shape[1] == n_features:
            intercept = getattr(model, "intercept_", getattr(model, "_y_mean", None))
            if intercept is None:
                raise
            return np.asarray(centered @ coef.T + np.asarray(intercept), dtype=np.float32)

        if coef.ndim == 2 and coef.shape[0] == n_features:
            x_std = np.asarray(getattr(model, "_x_std", np.ones(n_features)))
            y_mean = getattr(model, "_y_mean", getattr(model, "intercept_", None))
            if y_mean is None:
                raise
            return np.asarray((centered / x_std) @ coef + np.asarray(y_mean), dtype=np.float32)

        raise exc


def subset_msi_features(
    matrix: np.ndarray,
    source_feature_names: np.ndarray,
    target_feature_names: np.ndarray,
    *,
    matrix_label: str,
) -> np.ndarray:
    lookup = {str(name): idx for idx, name in enumerate(source_feature_names)}
    missing = [str(name) for name in target_feature_names if str(name) not in lookup]
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += ", ..."
        raise ValueError(
            f"Cannot align {matrix_label} MSI output to PLS target features; "
            f"missing {len(missing)} features: {preview}"
        )
    indices = np.asarray([lookup[str(name)] for name in target_feature_names], dtype=np.int64)
    return matrix[:, indices]


def process_job(
    job: dict[str, Any],
    schema: dict[str, Any],
    model_dir: Path,
    *,
    device: str,
    batch_size: int,
    min_match_fraction: float,
    quiet: bool,
    keep_failed_inputs: bool,
    force_recompute: bool,
    decoder_info: dict[str, Any],
) -> dict[str, Any]:
    input_path = Path(job["input_h5ad"])
    output_npz = Path(job["output_npz"])
    output_json = Path(job["output_json"])
    output_aligned_rna = (
        Path(job["output_aligned_rna_h5ad"])
        if job.get("output_aligned_rna_h5ad")
        else None
    )
    output_imputed_msi = (
        Path(job["output_imputed_msi_h5ad"])
        if job.get("output_imputed_msi_h5ad")
        else None
    )

    job_force_recompute = bool(job.get("force_recompute", False))
    expected_outputs = [output_npz, output_json]
    expected_outputs.extend(
        path for path in (output_aligned_rna, output_imputed_msi) if path is not None
    )
    if all(path.exists() for path in expected_outputs) and not (
        force_recompute or job_force_recompute
    ):
        try:
            with output_json.open("r", encoding="utf-8") as handle:
                cached_metadata = json.load(handle)
        except Exception:
            cached_metadata = {}
        if decoder_metadata_matches(cached_metadata, decoder_info):
            log_progress(
                f"job skipped_existing kind={job.get('kind')} donor={job.get('donor_id')} "
                f"section={job.get('section_id')}"
            )
            return {
                "status": "skipped_existing",
                "input_h5ad": str(input_path),
                "output_npz": str(output_npz),
                "output_json": str(output_json),
                "output_aligned_rna_h5ad": (
                    str(output_aligned_rna) if output_aligned_rna is not None else None
                ),
                "output_imputed_msi_h5ad": (
                    str(output_imputed_msi) if output_imputed_msi is not None else None
                ),
                "msi_decoder": decoder_info["metadata"],
            }

    metadata: dict[str, Any] = {
        "status": "started",
        "created_at": utc_now(),
        "kind": job.get("kind"),
        "donor_id": job.get("donor_id"),
        "section_id": job.get("section_id"),
        "input_h5ad": str(input_path),
        "output_npz": str(output_npz),
        "model_dir": str(model_dir),
        "reference_path": schema["reference_path"],
        "device": device,
        "min_match_fraction": float(job.get("min_match_fraction", min_match_fraction)),
        "msi_decoder": decoder_info["metadata"],
    }
    try:
        log_progress(
            f"job start kind={job.get('kind')} donor={job.get('donor_id')} "
            f"section={job.get('section_id')} input={input_path}"
        )
        source = ad.read_h5ad(input_path)
        log_progress(f"loaded input: n_obs={source.n_obs:,} n_vars={source.n_vars:,}")
        batch_keys = job.get("batch_keys")
        batch_value = job.get("batch_value")
        aligned, alignment = align_to_reference(
            source,
            schema,
            batch_keys=batch_keys,
            batch_value=batch_value,
            min_match_fraction=metadata["min_match_fraction"],
        )
        metadata["alignment"] = alignment
        log_progress(
            "aligned input: "
            f"matched={alignment['matched_features']:,}/{alignment['target_st_vars']:,} "
            f"ST features ({alignment['matched_fraction_of_st']:.3f}); "
            f"target_vars={alignment['target_n_vars']:,}"
        )

        log_progress("teacher embedding start")
        teacher, _, teacher_msi, teacher_meta = embed_aligned(
            aligned,
            model_dir / "teacher.pt",
            device=device,
            batch_size=batch_size,
            quiet=quiet,
        )
        log_progress(f"teacher embedding done: shape={list(teacher.shape)} msi={list(teacher_msi.shape)}")
        log_progress("student embedding start")
        student, student_pls_embedding, student_neural_msi, student_meta = embed_aligned(
            aligned,
            model_dir / "student.pt",
            device=device,
            batch_size=batch_size,
            quiet=quiet,
        )
        log_progress(
            f"student embedding done: shape={list(student.shape)} "
            f"neural_msi={list(student_neural_msi.shape)}"
        )
        msi_feature_names = np.asarray(
            aligned.var_names[aligned.var["type"].astype(str).to_numpy() == "SM"].astype(str),
            dtype=str,
        )
        if decoder_info["metadata"]["active"] == "pls":
            log_progress("PLS MSI prediction start")
            student_msi, pls_feature_names = predict_pls_msi(
                decoder_info,
                student_pls_embedding,
            )
            teacher_msi = subset_msi_features(
                teacher_msi,
                msi_feature_names,
                pls_feature_names,
                matrix_label="teacher",
            )
            msi_feature_names = pls_feature_names
            student_meta["decoded_msi_from"] = "q_mu -> PLSRegression.predict"
            student_meta["pls_decoder_embedding_key"] = decoder_info["metadata"].get(
                "embedding_key"
            )
            log_progress(f"PLS MSI prediction done: n_features={len(msi_feature_names):,}")
        else:
            student_msi = student_neural_msi

        # The teacher's own neural decoder reconstructs all 787 MSI features from a
        # single shared decoder, which makes them extremely collinear (median
        # cross-feature correlation with the per-cell mean ~0.85 empirically) and
        # washes out feature-specific structure (e.g. dopamine) into a generic
        # intensity axis. The PLS decoder is fit per-feature against the student
        # embedding and preserves that structure (median correlation ~0.41), so
        # prefer it whenever it's the active decoder.
        if decoder_info["metadata"]["active"] == "pls":
            imputed_msi_x = student_msi
            imputed_msi_source = "student_msi (PLSRegression.predict)"
        else:
            imputed_msi_x = teacher_msi
            imputed_msi_source = "teacher_msi"

        if output_aligned_rna is not None:
            st_mask = aligned.var["type"].astype(str).to_numpy() == "ST"
            aligned_rna = aligned[:, st_mask].copy()
            aligned_rna.uns["spatialmeta_alignment"] = alignment
            atomic_write_h5ad(output_aligned_rna, aligned_rna)

        if output_imputed_msi is not None:
            imputed_msi = ad.AnnData(
                X=np.asarray(imputed_msi_x, dtype=np.float32),
                obs=source.obs.copy(),
                var=pd.DataFrame(index=pd.Index(msi_feature_names, name="feature")),
            )
            imputed_msi.obsm["X_spatialmeta_teacher"] = np.asarray(
                teacher, dtype=np.float32
            )
            imputed_msi.obsm["X_spatialmeta_student"] = np.asarray(
                student, dtype=np.float32
            )
            imputed_msi.uns["spatialmeta_msi_decoder"] = decoder_info["metadata"]
            imputed_msi.uns["spatialmeta_msi_source"] = imputed_msi_source
            atomic_write_h5ad(output_imputed_msi, imputed_msi)

        atomic_write_npz(
            output_npz,
            obs_names=np.asarray(source.obs_names.astype(str), dtype=str),
            teacher=teacher,
            student=student,
            teacher_msi=teacher_msi,
            student_msi=student_msi,
            msi_feature_names=msi_feature_names,
        )
        metadata.update(
            {
                "status": "success",
                "finished_at": utc_now(),
                "teacher": teacher_meta,
                "student": student_meta,
                "embedding_shape": {
                    "teacher": list(teacher.shape),
                    "student": list(student.shape),
                },
                "decoded_msi_shape": {
                    "teacher": list(teacher_msi.shape),
                    "student": list(student_msi.shape),
                },
                "decoded_msi_feature_count": int(len(msi_feature_names)),
                "output_imputed_msi_source": imputed_msi_source,
                "output_aligned_rna_h5ad": (
                    str(output_aligned_rna) if output_aligned_rna is not None else None
                ),
                "output_imputed_msi_h5ad": (
                    str(output_imputed_msi) if output_imputed_msi is not None else None
                ),
                "cleanup_status": "input_retained",
            }
        )
        log_progress(
            f"job success kind={job.get('kind')} donor={job.get('donor_id')} "
            f"section={job.get('section_id')} output={output_npz}"
        )
        if job.get("delete_input_on_success", True):
            input_path.unlink(missing_ok=True)
            metadata["cleanup_status"] = "input_deleted"
        atomic_write_json(output_json, metadata)
        return metadata
    except Exception as exc:  # noqa: BLE001 - write diagnostics for batch failures
        metadata.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": repr(exc),
                "cleanup_status": "input_retained",
            }
        )
        if not keep_failed_inputs:
            input_path.unlink(missing_ok=True)
            metadata["cleanup_status"] = "failed_input_deleted"
        atomic_write_json(output_json.with_name(output_json.name + ".failed"), metadata)
        raise


def run_manifest(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    jobs = manifest.get("jobs", [])
    log_progress(
        f"manifest start: jobs={len(jobs)} device={args.device} "
        f"batch_size={args.batch_size} decoder={args.msi_decoder} quiet={args.quiet}"
    )
    schema = load_reference_schema(model_dir)
    decoder_info = load_msi_decoder(
        model_dir,
        requested=args.msi_decoder,
        pls_decoder_path=Path(args.pls_decoder_path) if args.pls_decoder_path else None,
    )

    results = []
    for job_idx, job in enumerate(jobs, start=1):
        log_progress(
            f"manifest job {job_idx}/{len(jobs)}: kind={job.get('kind')} "
            f"donor={job.get('donor_id')} section={job.get('section_id')}"
        )
        results.append(
            process_job(
                job,
                schema,
                model_dir,
                device=args.device,
                batch_size=args.batch_size,
                min_match_fraction=args.min_match_fraction,
                quiet=args.quiet,
                keep_failed_inputs=args.keep_failed_inputs,
                force_recompute=args.force_recompute,
                decoder_info=decoder_info,
            )
        )
        log_progress(f"manifest job {job_idx}/{len(jobs)} complete")

    summary_path = manifest.get("summary_json")
    if summary_path:
        atomic_write_json(
            Path(summary_path),
            {
                "status": "success",
                "finished_at": utc_now(),
                "manifest": str(args.manifest),
                "n_jobs": len(results),
                "msi_decoder": decoder_info["metadata"],
                "results": results,
            },
        )
    log_progress("manifest complete")


def smoke_test(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    log_progress(
        f"smoke-test start: model_dir={model_dir} device={args.device} decoder={args.msi_decoder}"
    )
    schema = load_reference_schema(model_dir)
    decoder_info = load_msi_decoder(
        model_dir,
        requested=args.msi_decoder,
        pls_decoder_path=Path(args.pls_decoder_path) if args.pls_decoder_path else None,
    )
    ref = ad.read_h5ad(model_dir / "joint_adata.h5ad")
    try:
        for name in ["teacher.pt", "student.pt"]:
            _, metadata = load_model_for_adata(
                ref,
                model_dir / name,
                device=args.device,
            )
            print(
                f"Loaded {name}: latent={metadata['n_latent']} "
                f"ignored_edge_keys={len(metadata['ignored_state_keys'])}"
            )
        print(
            "Reference schema:",
            f"n_vars={len(schema['var_names'])}",
            f"batch_categories={schema['batch_categories']}",
        )
        print(
            "MSI decoder:",
            f"requested={decoder_info['metadata']['requested']}",
            f"active={decoder_info['metadata']['active']}",
            f"fallback={decoder_info['metadata'].get('fallback_reason')}",
        )
        if decoder_info["metadata"]["active"] == "pls":
            model = decoder_info["payload"]["model"]
            n_features = getattr(model, "n_features_in_", None)
            if n_features is None and hasattr(model, "x_weights_"):
                n_features = model.x_weights_.shape[0]
            if n_features is None:
                raise ValueError("Could not infer PLS decoder input feature count.")
            probe = np.zeros((1, int(n_features)), dtype=np.float32)
            probe_msi, probe_features = predict_pls_msi(decoder_info, probe)
            print(
                "PLS decoder validation:",
                f"target_features={len(probe_features)}",
                f"probe_prediction_shape={list(probe_msi.shape)}",
            )
        log_progress("smoke-test complete")
    finally:
        del ref


def self_test(args: argparse.Namespace) -> None:
    schema = load_reference_schema(Path(args.model_dir))
    st_features = [
        str(name).split(":", 1)[1].lower()
        for name, typ in zip(schema["var_names"], schema["var_types"])
        if typ == "ST"
    ][:20]
    if len(st_features) < 5:
        raise AssertionError("Reference schema did not expose enough ST features.")

    x = sparse.csr_matrix(np.eye(4, len(st_features), dtype=np.float32))
    source = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(index=st_features),
    )
    aligned, diagnostics = align_to_reference(
        source,
        schema,
        batch_keys=["section"],
        batch_value=None,
        min_match_fraction=0.0,
    )
    assert aligned.n_obs == source.n_obs
    assert aligned.n_vars == len(schema["var_names"])
    assert diagnostics["matched_features"] == len(st_features)
    assert "section" in aligned.obs.columns
    assert "spatial" in aligned.obsm

    empty_source = ad.AnnData(
        X=sparse.csr_matrix((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["NOT_A_MODEL_GENE", "ALSO_NOT_A_MODEL_GENE"]),
    )
    try:
        align_to_reference(
            empty_source,
            schema,
            batch_keys=["section"],
            batch_value=None,
            min_match_fraction=0.10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Low-overlap feature alignment should fail.")
    print("self-test ok", diagnostics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-match-fraction", type=float, default=DEFAULT_MIN_MATCH_FRACTION)
    parser.add_argument("--msi-decoder", choices=["neural", "pls"], default="neural")
    parser.add_argument("--pls-decoder-path")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--keep-failed-inputs", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--cleanup-staging", action="store_true")
    mode.add_argument("--cleanup-failed", action="store_true")

    parser.add_argument("--handoff-root")
    parser.add_argument("--older-than-hours", type=float, default=24.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cleanup_staging or args.cleanup_failed:
            if not args.handoff_root:
                raise ValueError("--handoff-root is required for cleanup modes.")
            removed = cleanup_staging(
                Path(args.handoff_root),
                older_than_hours=args.older_than_hours,
                failed_only=args.cleanup_failed,
            )
            print(f"removed {removed} staging directories")
        elif args.smoke_test:
            smoke_test(args)
        elif args.self_test:
            self_test(args)
        elif args.manifest:
            run_manifest(args)
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should provide clear terminal error
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
