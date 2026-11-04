try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%%
import os

import muon as mu
import scanpy as sc
from pathlib import Path

CAH_MOFA_N_FACTORS = 5
CAH_MOFA_GPU_MODE = True
CAH_MOFA_GPU_DEVICE = int(os.environ.get("CAH_MOFA_GPU_DEVICE", "2"))
CAH_MOFA_MODEL_PATH = Path(__import__("os").environ["SPATIALJEPA_HOME"], "outputs/sea_ad_lipid_gps_minimal/cah/snrna_mofa.h5mu")

def ensure_group_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o2775)
    except (PermissionError, OSError):
        # Non-owner collaborators can write via group perms but cannot chmod.
        pass

def canonicalize_factor_sign(
    model_path: Path,
    *,
    factor_index: int = 0,
    anchor_view: str = "msi",
    anchor_feature: str = "msi:Dopamine",
) -> None:
    """Flip a MOFA factor's sign in place so `anchor_feature` loads positively.

    MOFA factors are identified only up to a joint sign flip of Z and W
    (Z @ W.T, the ELBO, and every downstream fit statistic are invariant
    under negating both), so this is a relabelling of an arbitrary
    convention the optimizer happens to land on, not a change to the fit.
    Without it, Factor1's polarity (and hence "positive"/"negative" labels
    downstream) flips unpredictably between training runs.
    """
    import h5py

    with h5py.File(model_path, "r+") as f:
        feature_names = [x.decode() for x in f[f"features/{anchor_view}"][:]]
        anchor_idx = feature_names.index(anchor_feature)
        weight = f[f"expectations/W/{anchor_view}"][factor_index, anchor_idx]
        if weight >= 0:
            print(f"Factor index {factor_index} already has {anchor_feature} >= 0; no flip needed.")
            return
        for view in [v.decode() for v in f["views/views"][:]]:
            w = f[f"expectations/W/{view}"]
            w[factor_index, :] = -w[factor_index, :]
        for group in [g.decode() for g in f["groups/groups"][:]]:
            z = f[f"expectations/Z/{group}"]
            z[factor_index, :] = -z[factor_index, :]
    print(f"Flipped sign of factor index {factor_index} so {anchor_feature} loads positively.")

def train_cah_mofa_model(
    mudata: mu.MuData,
    mofa_model_path: Path = CAH_MOFA_MODEL_PATH,
) -> Path:
    """Train CaH RNA/MSI MOFA and write the mofapy2 HDF5 model."""

    ensure_group_writable_dir(mofa_model_path.parent)
    mu.tl.mofa(
        mudata,
        use_var=None,
        likelihoods=["gaussian"] * len(mudata.mod),
        n_factors=CAH_MOFA_N_FACTORS,
        scale_views=True,
        center_groups=True,
        ard_weights=True,
        ard_factors=True,
        spikeslab_weights=True,
        spikeslab_factors=False,
        convergence_mode="fast",
        n_iterations=100,
        gpu_mode=CAH_MOFA_GPU_MODE,
        gpu_device=CAH_MOFA_GPU_DEVICE,
        use_float32=True,
        seed=42,
        outfile=str(mofa_model_path),
        verbose=True,
        quiet=False,
    )
    print(f"Trained CaH MOFA model: {mofa_model_path}")
    canonicalize_factor_sign(mofa_model_path, factor_index=0, anchor_feature="msi:Dopamine")
    # Load the MOFA model to ensure it was saved correctly, then return the path.
    import mofax as mfx

    try:
        mofa = mfx.mofa_model(str(mofa_model_path))
        print("MOFA model loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load MOFA model after training: {e}")
    return mofa_model_path, mofa

#%% load data

scrna_msi_mudata = mu.read_h5mu(f"{__import__('os').environ['SPATIALJEPA_HOME']}/outputs/sea_ad_lipid_gps_minimal/cah/snrna_spatialmeta_aligned.h5mu")

# %%

mofa_model_path, mofa = train_cah_mofa_model(scrna_msi_mudata)

# %%
