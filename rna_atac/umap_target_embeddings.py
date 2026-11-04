"""Target UMAPs for Supplementary Fig. S1 computed as main Fig. 2d.

Fig. 2d (analyze_rna_atac.py, run 20260505_190037): concat RNA+ATAC student
embedding, keep the target 'test' split of the run's domain_splits.json, then
sc.pp.neighbors(n_neighbors=30) and sc.tl.umap(min_dist=0.1, spread=1.5), all
other arguments at scanpy defaults. The zero-shot student and MultiVI use these
settings; the OT-aligned student uses the source (Fig. 2b/c) settings on the
same nuclei (see SETTINGS). MultiVI's zero-shot embeddings exist only for the
run's 'train' split (0 test nuclei), so MultiVI uses its own nuclei.
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
import numpy as np
import pandas as pd
import scanpy as sc

INPUTS = f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap/metric_inputs"
SPLITS = f"{os.environ['SPATIALJEPA_HOME']}/outputs/rna_atac_models/20260505_190037/artifacts/splits/domain_splits.json"
OUT = f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap_v2/s1_umaps"
specs = {s["domain"]: s for s in json.load(open(os.path.join(INPUTS, "specs.json")))}
test = set(map(str, json.load(open(SPLITS))["domains"]["target"]["splits"]["test"]["obs_names"]))

def fig2d_umap(a):  # Fig. 2d settings
    sc.pp.neighbors(a, n_neighbors=30)
    sc.tl.umap(a, min_dist=0.1, spread=1.5)


def source_umap(a):  # Fig. 2b/c settings: compute_concat_umap(deterministic=True, n_neighbors=10)
    sc.pp.neighbors(a, n_neighbors=10, use_rep="X", knn=True, method="umap",
                    metric="euclidean", random_state=0)
    sc.tl.umap(a, random_state=0, init_pos="spectral")


# OT-aligned embeddings are barycentric projections onto source embeddings (each
# nucleus lies ~0.006 from a source spot vs ~0.3 spot spacing), i.e. they live
# in the source embedding space, so they get the source UMAP settings.
SETTINGS = {"target": (fig2d_umap, "__fig2d"),
            "target_ot": (source_umap, "__subset_src_settings"),
            "target_multivi_zero_shot": (fig2d_umap, "__fig2d")}
ONLY = os.environ.get("ONLY")  # e.g. ONLY=target_ot to recompute a single model
# MultiVI zero-shot latents for all target nuclei (encode_multivi_zero_shot.py),
# replacing the original 50%-split inputs; written under a distinct cache name
V3_INPUTS = os.path.join(os.path.dirname(OUT), "..", "metrics_bootstrap_v3", "metric_inputs")
if os.environ.get("MULTIVI_FULL"):
    SETTINGS["target_multivi_zero_shot"] = (fig2d_umap, "__fig2d_full_target")
    specs["target_multivi_zero_shot"] = {
        **json.load(open(os.path.join(V3_INPUTS, "specs.json")))[0], "_dir": V3_INPUTS}

# ALL_SOURCE=1: preview with every target model on the source settings
# (writes separate "__subset_src_settings" caches; the S1 caches are untouched)
if os.environ.get("ALL_SOURCE"):
    SETTINGS = {d: (source_umap, "__subset_src_settings") for d in SETTINGS}
    if os.environ.get("MULTIVI_FULL"):
        SETTINGS["target_multivi_zero_shot"] = (source_umap, "__subset_src_settings_full_target")

for domain, (run_umap, suffix) in SETTINGS.items():
    if ONLY and domain != ONLY:
        continue
    spec = specs[domain]
    INPUTS_D = spec.get("_dir", INPUTS)
    key, lab = spec["embedding_key"], spec["label_key"]
    rna = ad.read_h5ad(os.path.join(INPUTS_D, spec["rna_file"]))
    atac = ad.read_h5ad(os.path.join(INPUTS_D, spec["atac_file"]))
    obs = pd.concat([
        pd.DataFrame({"modality": "rna", "label": rna.obs[lab["rna"]].astype(str).values},
                     index=rna.obs_names.astype(str) + "_rna"),
        pd.DataFrame({"modality": "atac", "label": atac.obs[lab["atac"]].astype(str).values},
                     index=atac.obs_names.astype(str) + "_atac"),
    ])
    concat = ad.AnnData(X=np.concatenate([rna.obsm[key], atac.obsm[key]], axis=0), obs=obs)
    in_test = concat.obs_names.str.split("_").str[0].isin(test)
    subset = "test" if in_test.any() else "own_nuclei"
    if in_test.any():
        concat = concat[in_test].copy()
    run_umap(concat)
    df = concat.obs.copy()
    df["umap1"], df["umap2"] = concat.obsm["X_umap"][:, 0], concat.obsm["X_umap"][:, 1]
    df["subset"] = subset
    df.to_csv(os.path.join(OUT, f"{domain}{suffix}.csv.gz"))
    print("done:", domain, subset, concat.n_obs, flush=True)
