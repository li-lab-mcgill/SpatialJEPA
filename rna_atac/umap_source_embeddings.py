"""Source UMAPs (all spots) for Supplementary Fig. S1, with the Fig. 2d target settings.

sc.pp.neighbors(n_neighbors=30) + sc.tl.umap(min_dist=0.1, spread=1.5), other
arguments at scanpy defaults. Writes "<domain>__fig2d_settings.csv.gz" caches;
the S1 caches are untouched. As analyze_rna_atac.py flips the student's UMAP
to face the teacher's, the student's axis signs are chosen here to best match
the teacher's layout (same spots; sign flips only).
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
OUT = f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap_v2/s1_umaps"
specs = {s["domain"]: s for s in json.load(open(os.path.join(INPUTS, "specs.json")))}

res = {}
for domain in ["teacher_source", "source", "source_nonspatial", "source_multivi"]:
    spec = specs[domain]
    key, lab = spec["embedding_key"], spec["label_key"]
    rna = ad.read_h5ad(os.path.join(INPUTS, spec["rna_file"]))
    atac = ad.read_h5ad(os.path.join(INPUTS, spec["atac_file"]))
    obs = pd.concat([
        pd.DataFrame({"modality": "rna", "label": rna.obs[lab["rna"]].astype(str).values},
                     index=rna.obs_names.astype(str) + "_rna"),
        pd.DataFrame({"modality": "atac", "label": atac.obs[lab["atac"]].astype(str).values},
                     index=atac.obs_names.astype(str) + "_atac"),
    ])
    concat = ad.AnnData(X=np.concatenate([rna.obsm[key], atac.obsm[key]], axis=0), obs=obs)
    sc.pp.neighbors(concat, n_neighbors=30)
    sc.tl.umap(concat, min_dist=0.1, spread=1.5)
    df = concat.obs.copy()
    df["label"] = df["label"].str.replace(r"^c_", "", regex=True)
    df["umap1"], df["umap2"] = concat.obsm["X_umap"][:, 0], concat.obsm["X_umap"][:, 1]
    res[domain] = df
    print("done:", domain, concat.n_obs, flush=True)

t, s = res["teacher_source"], res["source"]
assert t.index.equals(s.index)
for ax in ["umap1", "umap2"]:
    if np.corrcoef(t[ax], s[ax])[0, 1] < 0:
        s[ax] *= -1
        print("flipped student", ax)
for domain, df in res.items():
    df.to_csv(os.path.join(OUT, f"{domain}__fig2d_settings.csv.gz"))
