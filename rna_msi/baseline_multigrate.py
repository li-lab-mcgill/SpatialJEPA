#!/usr/bin/env python
#%%
"""Train a multigrate MultiVAE on a single SMA spatial RNA + metabolomics (MSI) sample.

multigrate models each modality with its own likelihood, so the SMA RNA block uses an
``nb`` (negative-binomial, raw counts) loss and the continuous MSI block uses an ``mse``
loss -- the metabolite intensities are modelled directly, without discretization and
without TOTALVI's count-protein background-prior workaround.

Beyond the joint latent, this script extracts the **modality-specific** latents
(RNA-only and MSI-only posterior means) and exports all three to disk, so they can be
reused downstream (e.g. cross-modal benchmarking).

Data source: the already-processed ``joint_adata.h5ad`` written by
``train_rna_msi.py`` at
``${SPATIALJEPA_OUT}/spatialjepa_models/<sample_id>/joint_adata.h5ad``. That object has
Moran's-I spatially-variable feature selection, normalization layers, var typing
(``var["type"]`` in {"ST", "SM"}), and obs annotations (region/lesion/section/
RNA_clusters/MSI_clusters) already applied. We split it into RNA + MSI modalities,
organize them into a multigrate multiome AnnData, train, embed, and save.

Env: ``conda activate scvi_env`` (multigrate 0.0.2, scvi-tools 1.3.3).
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import os
import site
import sys


def _configure_python_runtime() -> None:
    """Keep notebook kernels from importing packages out of ~/.local."""

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("TMPDIR", "/tmp")
    os.environ.setdefault("PIP_CACHE_DIR", "/tmp/pip_cache")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

    user_site = site.getusersitepackages()
    user_site_abs = os.path.abspath(user_site)
    sys.path[:] = [
        path for path in sys.path
        if os.path.abspath(path or os.curdir) != user_site_abs
    ]
    site.ENABLE_USER_SITE = False

    watched = {"anndata", "mudata", "muon", "scanpy", "scvi", "multigrate"}
    loaded_from_user_site = {}
    for name, module in sys.modules.items():
        if name.partition(".")[0] not in watched:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and os.path.abspath(module_file).startswith(user_site_abs):
            loaded_from_user_site[name] = module_file

    if loaded_from_user_site:
        details = "\n".join(
            f"  {name}: {path}" for name, path in sorted(loaded_from_user_site.items())
        )
        raise RuntimeError(
            "User-site packages were already imported before runtime setup. "
            "Restart the Jupyter kernel and run this script from the top.\n"
            f"{details}"
        )


_configure_python_runtime()

import argparse
from pprint import pprint

import multigrate
import torch
from dotenv import dotenv_values, load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")

ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
DEFAULT_SAMPLE_ID = "V11L12-109_B1"

RNA_PREFIX = "rna:"
MSI_PREFIX = "msi:"

MULTIGRATE_LATENT_KEY = "X_multigrate"
MULTIGRATE_RNA_LATENT_KEY = "X_multigrate_rna"
MULTIGRATE_MSI_LATENT_KEY = "X_multigrate_msi"

SPATIALJEPA_OUT = None

#%%
def bootstrap_runtime():
    global SPATIALJEPA_OUT

    load_dotenv(dotenv_path=ENV_FILE_PATH)
    print("Loaded environment variables from .env or env:", end="\n\n")
    pprint(dotenv_values(ENV_FILE_PATH))

    SPATIALJEPA_OUT = os.getenv("SPATIALJEPA_OUT")
    if SPATIALJEPA_OUT is None:
        raise EnvironmentError(
            "SPATIALJEPA_OUT is not set. Export SPATIALJEPA_OUT to the base outputs directory, e.g. "
            "'$SPATIALJEPA_HOME/outputs'."
        )


def is_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        if shell == "TerminalInteractiveShell":
            return False
        return False
    except Exception:
        return False


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(
        description="Train multigrate MultiVAE on one SMA sample (RNA nb + MSI mse)."
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=DEFAULT_SAMPLE_ID,
        help="SMA sample id whose joint_adata.h5ad (from script 3) to train on.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Number of epochs to train MultiVAE.",
    )
    parser.add_argument(
        "--n-latent",
        type=int,
        default=16,
        help="Dimensionality of the MultiVAE joint latent space (z_dim).",
    )
    parser.add_argument(
        "--msi-input",
        type=str,
        choices=["X", "normalized", "counts"],
        default="X",
        help=(
            "joint_adata source for the MSI (mse) modality. 'X' = the processed "
            "log1p+scaled SM block (model-ready for MSE); 'normalized' = TIC-normalized "
            "intensities; 'counts' = raw MSI intensities."
        ),
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default=None,
        help="Optional obs key to integrate over as MultiVAE batch (default None; single sample).",
    )
    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


def validate_args(args):
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be a positive integer.")
    if args.n_latent <= 0:
        raise ValueError("--n-latent must be a positive integer.")


def _patch_multigrate_scvi_compat():
    """Compat shims for multigrate 0.0.2 against scvi-tools 1.3.3.

    1. multigrate's MultiVAE.train passes ``use_gpu=...`` into scvi-tools' ``DataSplitter``,
       which (since scvi-tools 1.0) stores unknown kwargs as torch ``DataLoader`` kwargs and
       raises ``DataLoader.__init__() got an unexpected keyword argument 'use_gpu'``. Strip it
       (also covers multigrate's GroupDataSplitter subclass).
    2. scvi-tools' AdversarialTrainingPlan.training_step reads ``inference_outputs["z"]``, but
       multigrate's MultiVAETorch.inference returns the joint latent under ``z_joint``. Alias
       ``z`` -> ``z_joint`` on the inference output."""
    from scvi.dataloaders import _data_splitting as _ds

    if not getattr(_ds.DataSplitter.__init__, "_use_gpu_shim", False):
        _orig_init = _ds.DataSplitter.__init__

        def _init(self, *args, **kwargs):
            kwargs.pop("use_gpu", None)
            _orig_init(self, *args, **kwargs)

        _init._use_gpu_shim = True
        _ds.DataSplitter.__init__ = _init

    from multigrate.module import _multivae_torch as _mt

    if not getattr(_mt.MultiVAETorch.inference, "_z_alias_shim", False):
        _orig_inference = _mt.MultiVAETorch.inference

        def _inference(self, *args, **kwargs):
            out = _orig_inference(self, *args, **kwargs)
            if isinstance(out, dict) and "z" not in out and "z_joint" in out:
                out["z"] = out["z_joint"]
            return out

        _inference._z_alias_shim = True
        _mt.MultiVAETorch.inference = _inference


def _to_csr(mat):
    return mat.tocsr() if sp.issparse(mat) else sp.csr_matrix(np.asarray(mat))


def _strip_prefix(var_names, prefix):
    return [v[len(prefix):] if v.startswith(prefix) else v for v in var_names]


def _block(joint, mask, source):
    """Return the (ST or SM) feature block of joint_adata from the chosen source."""
    if source == "X":
        return joint.X[:, mask]
    return joint.layers[source][:, mask]


def load_sma_joint_adata(sample_id, msi_input):
    """Load the processed joint_adata and split it into RNA + MSI modality AnnDatas."""
    joint_path = os.path.join(SPATIALJEPA_OUT, "spatialjepa_models", sample_id, "joint_adata.h5ad")
    if not os.path.exists(joint_path):
        raise FileNotFoundError(
            "joint_adata.h5ad not found: {}.\nRun train_rna_msi.py for "
            "sample_id='{}' first.".format(joint_path, sample_id)
        )

    joint = sc.read_h5ad(joint_path)
    if "type" not in joint.var.columns:
        raise ValueError("joint_adata.var is missing the 'type' column (expected 'ST'/'SM').")
    if msi_input != "X" and msi_input not in joint.layers:
        raise ValueError(
            "Requested --msi-input '{}' not in joint_adata.layers ({}).".format(
                msi_input, list(joint.layers.keys())
            )
        )

    st_mask = (joint.var["type"].values == "ST")
    sm_mask = (joint.var["type"].values == "SM")
    print(
        "[INFO] joint_adata {}: {} ST (RNA) + {} SM (MSI) features".format(
            joint.shape, int(st_mask.sum()), int(sm_mask.sum())
        )
    )

    # RNA modality: raw integer counts (nb loss + RNA size factors).
    rna = joint[:, st_mask].copy()
    rna.var_names = _strip_prefix(rna.var_names, RNA_PREFIX)
    rna.layers["counts"] = _to_csr(joint.layers["counts"][:, st_mask])
    rna.X = rna.layers["counts"].copy()

    # MSI modality: continuous intensities for the mse loss (NOT discretized).
    msi = joint[:, sm_mask].copy()
    msi.var_names = _strip_prefix(msi.var_names, MSI_PREFIX)
    msi.layers["counts"] = _to_csr(_block(joint, sm_mask, msi_input))
    msi.X = msi.layers["counts"].copy()

    return rna, msi, joint


@torch.inference_mode()
def get_latent_means(model, adata, batch_size=256):
    """Deterministic joint + per-modality posterior means from a trained MultiVAE.

    multigrate's public get_latent_representation only writes the sampled joint latent.
    Here we replicate MultiVAETorch.inference's encoder path to read the per-modality
    means (mu) and combine them with the product-of-experts for the joint mean -- all
    deterministic. Valid only when encoders are not covariate-conditioned.
    """
    module = model.module
    assert not module.condition_encoders, (
        "get_latent_means assumes condition_encoders=False (no covariate concat in encoders)."
    )
    module.eval()
    device = next(module.parameters()).device  # replicate inference's @auto_move_data
    scdl = model._make_data_loader(adata=adata, batch_size=batch_size)

    joint_chunks = []
    modality_chunks = None
    for tensors in scdl:
        x = module._get_inference_input(tensors)["x"].to(device)
        xs = torch.split(x, module.input_dims, dim=-1)
        outs = [module._bottleneck(module._x_to_h(xm, m), m) for m, xm in enumerate(xs)]
        mus = [o[1] for o in outs]
        logvars = [o[2] for o in outs]
        masks = torch.stack([xm.sum(dim=1) > 0 for xm in xs], dim=1)
        mu_joint, _ = module._product_of_experts(
            torch.stack(mus, dim=1), torch.stack(logvars, dim=1), masks
        )
        joint_chunks.append(mu_joint.cpu())
        if modality_chunks is None:
            modality_chunks = [[] for _ in mus]
        for m, mu_m in enumerate(mus):
            modality_chunks[m].append(mu_m.cpu())

    joint = torch.cat(joint_chunks).numpy()
    per_modality = [torch.cat(chunks).numpy() for chunks in modality_chunks]
    return joint, per_modality


def compute_umap_and_leiden(adata, color_keys, out_dir, sample_id, point_size=60):
    """Joint-latent UMAP + Leiden, plotted on UMAP and (if present) spatial coords."""
    sc.pp.neighbors(adata, use_rep=MULTIGRATE_LATENT_KEY)
    sc.tl.umap(adata, min_dist=0.2)
    sc.tl.leiden(adata, resolution=0.25)

    present = [k for k in color_keys if k in adata.obs.columns] + ["leiden"]

    fig = sc.pl.umap(adata, color=present, show=False, return_fig=True)
    fig.savefig(os.path.join(out_dir, f"{sample_id}_multigrate_umap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"UMAP saved to {os.path.join(out_dir, f'{sample_id}_multigrate_umap.png')}")

    if "spatial" in adata.obsm:
        fig = sc.pl.embedding(adata, color=present, basis="spatial", s=point_size, show=False, return_fig=True)
        fig.savefig(os.path.join(out_dir, f"{sample_id}_multigrate_spatial.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Spatial embedding saved to {os.path.join(out_dir, f'{sample_id}_multigrate_spatial.png')}")

#%%
def main():
    # setup runtime
    bootstrap_runtime()
    notebook_mode = is_notebook()
    args = parse_args(notebook=notebook_mode)
    validate_args(args)

    # load and split the processed SMA joint adata
    rna, msi, joint = load_sma_joint_adata(args.sample_id, args.msi_input)
    n_genes = rna.n_vars

    # Organize the two modalities into a multigrate multiome AnnData. organize_* copies the
    # named layer into .X per modality and records uns["modality_lengths"].
    mvi = multigrate.data.organize_multiome_anndatas(
        adatas=[[rna], [msi]],
        layers=[["counts"], ["counts"]],
    )
    # Carry obs annotations + spatial coords from the original joint_adata (organize_* keeps
    # obs but not obsm); align by obs_names in case ordering changed.
    pos = joint.obs_names.get_indexer(mvi.obs_names)
    assert (pos >= 0).all(), "organized AnnData has obs not present in joint_adata"
    for col in ("RNA_clusters", "MSI_clusters", "region", "lesion", "section"):
        if col in joint.obs.columns:
            mvi.obs[col] = joint.obs[col].to_numpy()[pos]
    if "spatial" in joint.obsm:
        mvi.obsm["spatial"] = np.asarray(joint.obsm["spatial"])[pos]
    print(f"[INFO] organized multiome: {mvi.shape}; modality_lengths={mvi.uns['modality_lengths']}")

    # setup + model: RNA -> nb (size factors from the first n_genes RNA features),
    # MSI -> mse (continuous intensities, modelled directly).
    multigrate.model.MultiVAE.setup_anndata(
        mvi,
        rna_indices_end=n_genes,
        batch_key=args.batch_key,
    )
    model = multigrate.model.MultiVAE(
        mvi,
        z_dim=args.n_latent,
        losses=["nb", "mse"],
        integrate_on=args.batch_key,
    )
    model.view_anndata_setup()
    _patch_multigrate_scvi_compat()
    model.train(max_epochs=args.max_epochs)

    # save model
    model_dir = os.path.join(SPATIALJEPA_OUT, "multigrate_mouse_sma", args.sample_id)
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir, overwrite=True)
    print(f"Model saved to {model_dir}")

    # deterministic joint + per-modality (RNA-only / MSI-only) latent means
    joint_emb, (rna_emb, msi_emb) = get_latent_means(model, mvi)
    for name, emb in (("joint", joint_emb), ("rna", rna_emb), ("msi", msi_emb)):
        assert not np.isnan(emb).any(), f"NaNs in {name} latent"
    mvi.obsm[MULTIGRATE_LATENT_KEY] = joint_emb
    mvi.obsm[MULTIGRATE_RNA_LATENT_KEY] = rna_emb
    mvi.obsm[MULTIGRATE_MSI_LATENT_KEY] = msi_emb
    print(f"[INFO] latents: joint{joint_emb.shape} rna{rna_emb.shape} msi{msi_emb.shape}")

    # UMAP / Leiden / spatial visualization (on the joint latent)
    colors = ["RNA_clusters", "MSI_clusters", "region", "lesion"]
    for color in colors:
        if color in mvi.obs.columns:
            mvi.obs[color] = mvi.obs[color].astype(str)
    compute_umap_and_leiden(
        mvi,
        color_keys=colors,
        out_dir=model_dir,
        sample_id=args.sample_id,
    )

    # export to disk: full embedded AnnData + a compact npz of the latents
    # anndata cannot serialize uns dicts with integer keys; stringify modality_lengths.
    if "modality_lengths" in mvi.uns:
        mvi.uns["modality_lengths"] = {
            str(k): int(v) for k, v in mvi.uns["modality_lengths"].items()
        }
    adata_path = os.path.join(model_dir, "multigrate_adata.h5ad")
    mvi.write_h5ad(adata_path)
    print(f"Embedded AnnData saved to {adata_path}")

    npz_path = os.path.join(model_dir, "multigrate_modality_latents.npz")
    np.savez(
        npz_path,
        obs_names=np.asarray(mvi.obs_names, dtype=object),
        joint=joint_emb,
        rna=rna_emb,
        msi=msi_emb,
    )
    print(f"Modality latents saved to {npz_path}")


if __name__ == "__main__":
    main()

# %%
