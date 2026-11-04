"""Zero-shot MultiVI latents for ALL target nuclei (same nuclei as the other methods).

baseline_multivi.py applied the source-trained MultiVI model
to the target 'train' split only (50%, 11,637 nuclei). This re-applies the same
saved model, without retraining, to all 23,275 target nuclei, using the exact
gene/peak set of the saved target MuData and raw counts in X (as the script does).
Validation: latents for the 11,637 previously embedded nuclei must match the
saved X_multivi_{rna,atac}_zero_shot before anything is written.

Outputs (new folder; earlier results untouched):
  metrics_bootstrap_v3/metric_inputs/{specs.json, target_multivi_zero_shot__{rna,atac}.h5ad}
  metrics_bootstrap_v3/multivi_zero_shot_full_target_latents.npz
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import json
import os
import warnings

warnings.filterwarnings("ignore")
import anndata as ad
import mudata as md
import numpy as np
import scvi

D = f"{os.environ['SPATIALJEPA_HOME']}/data/aligned_data"
MODEL = f"{os.environ['SPATIALJEPA_HOME']}/outputs/multivi_mouse_brain_spatial_rna_atac"
OLD_INPUTS = f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap/metric_inputs"
V3 = f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap_v3"
MODALITIES = {"rna_layer": "rna", "atac_layer": "atac"}

# saved MuDatas: source (to load the model) and zero-shot target (feature set + reference latents)
src = md.read_h5mu(os.path.join(D, "multivi_mdata.h5mu"))
zs = md.read_h5mu(os.path.join(D, "multivi_mdata_zero_shot.h5mu"))
model = scvi.model.MULTIVI.load(MODEL, adata=src)
print("model loaded", flush=True)

# full target, restricted to the saved feature set, raw counts in X (as in the MultiVI script)
rna = ad.read_h5ad(os.path.join(D, "target_rna_aligned_SCT.h5ad"))
atac = ad.read_h5ad(os.path.join(D, "target_atac_aligned.h5ad"))
assert rna.obs_names.equals(atac.obs_names)
rna = rna[:, zs.mod["rna"].var_names].copy()
atac = atac[:, zs.mod["atac"].var_names].copy()
rna.X = rna.layers["counts"].tocsr() if hasattr(rna.layers["counts"], "tocsr") else rna.layers["counts"]
atac.X = atac.layers["counts"].tocsr() if hasattr(atac.layers["counts"], "tocsr") else atac.layers["counts"]
full = md.MuData({"rna": rna, "atac": atac})
scvi.model.MULTIVI.setup_mudata(full, modalities=MODALITIES)
print("target:", full.n_obs, "nuclei;", rna.n_vars, "genes;", atac.n_vars, "peaks", flush=True)

lat = {m: model.get_latent_representation(adata=full, modality=m)
       for m in ["expression", "accessibility", "joint"]}

# validation against the saved 50% zero-shot latents
pos = full.obs_names.get_indexer(zs.obs_names)
assert (pos >= 0).all()
for m, key in [("expression", ("rna", "X_multivi_rna_zero_shot")),
               ("accessibility", ("atac", "X_multivi_atac_zero_shot"))]:
    ref = np.asarray(zs.mod[key[0]].obsm[key[1]])
    diff = np.abs(lat[m][pos] - ref).max()
    print(f"validation {m}: max |new - saved| = {diff:.2e} over {len(pos)} nuclei", flush=True)
    assert diff < 1e-3, "recomputed latents do not match the saved ones"

# write new metric inputs in the metrics_bootstrap format, same labels as the other target models
os.makedirs(os.path.join(V3, "metric_inputs"), exist_ok=True)
old_specs = {s["domain"]: s for s in json.load(open(os.path.join(OLD_INPUTS, "specs.json")))}
spec = dict(old_specs["target_multivi_zero_shot"])
label_col = spec["label_key"]["rna"]
labels = ad.read_h5ad(os.path.join(OLD_INPUTS, "target__rna.h5ad")).obs[label_col]
labels = labels.reindex(full.obs_names)
assert labels.notna().all(), "some target nuclei lack a label"
for mod, m in [("rna", "expression"), ("atac", "accessibility")]:
    slim = ad.AnnData(X=np.zeros((full.n_obs, 1), dtype=np.float32),
                      obs=labels.to_frame(label_col).set_index(full.obs_names))
    slim.obsm[spec["embedding_key"]] = lat[m].astype(np.float32)
    slim.write_h5ad(os.path.join(V3, "metric_inputs", spec[f"{mod}_file"]))
with open(os.path.join(V3, "metric_inputs", "specs.json"), "w") as fh:
    json.dump([spec], fh, indent=2)
np.savez_compressed(os.path.join(V3, "multivi_zero_shot_full_target_latents.npz"),
                    obs_names=np.asarray(full.obs_names), **{m: v for m, v in lat.items()})
print("wrote", V3, flush=True)
