"""Astrocyte C2 (seed0) carried to Xenium, as astrocyte C1 was for Fig. 7 panel g (astrocyte_c1_ridge.py).

A ridge from the 464 Xenium panel genes (full snRNA UMIs, normalise_total + log1p, z-scored within
snRNA astrocytes) to latent C2, fitted within snRNA astrocytes; donor-grouped 5-fold CV for per-nucleus
and donor-mean agreement. Applied to Xenium astrocytes (log-normalised over the panel, z-scored over
astrocytes of all sections), then count-adjusted per section (residual of a regression on log
transcript counts, section mean kept), as for panel g. Writes astro_c2_xenium.pkl next to this script.
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import pickle
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer"
BASE = Path(os.environ['SPATIALJEPA_HOME'])
XENIUM_PATH = BASE / "data/SEA_AD/sea_ad_cah/CaH_Xenium_final.2026-01-07.h5ad"
SNRNA_PATH = BASE / "data/SEA_AD/sea_ad_cah/CaH_final_nuclei.2025-12-29.h5ad"
ALPHAS = np.logspace(0, 5, 11)
zs = lambda M: (M - M.mean(0)) / M.std(0).clip(1e-6)

cells = pd.read_parquet(HERE.parent / "astro_c1_adnc" / "cells.parquet", columns=["Donor ID", "Subclass"])
R = pickle.load(open(BASE / "outputs/human_cah_transfer/seed0/cah/pls_msi_ot_reg0.001_sec/results.pkl", "rb"))
assert R["scores"].index.astype(str).equals(cells.index.astype(str))
ia = np.flatnonzero((cells["Subclass"].astype(str) == "Astrocyte").to_numpy())
c2 = R["scores"]["PLS C2"].to_numpy()[ia]
donor = cells["Donor ID"].astype(str).to_numpy()[ia]
xen_genes = pd.Index(ad.read_h5ad(XENIUM_PATH, backed="r").var_names.astype(str))

with h5py.File(SNRNA_PATH, "r") as h:                       # astrocyte rows only, by indptr slices
    sn_obs = pd.Index(h["obs"][h["obs"].attrs["_index"]].asstr()[:])
    sn_var = pd.Index(h["var"][h["var"].attrs["_index"]].asstr()[:]).str.upper()
    rows = sn_obs.get_indexer(pd.Index(cells.index.astype(str)).take(ia))
    assert (rows >= 0).all()
    gpan = xen_genes[xen_genes.str.upper().isin(sn_var)]
    colmap = np.full(len(sn_var), -1)
    colmap[sn_var.get_indexer(gpan.str.upper())] = np.arange(len(gpan))
    U, ip = h["layers/UMIs"], h["layers/UMIs/indptr"][:]
    Ua, lib = np.zeros((len(ia), len(gpan))), np.zeros(len(ia))
    for k, i in enumerate(rows):
        d, j = U["data"][ip[i]:ip[i + 1]], U["indices"][ip[i]:ip[i + 1]]
        lib[k] = d.sum()
        m = colmap[j]
        Ua[k, m[m >= 0]] = d[m >= 0]
Xpan = zs(np.log1p(Ua / lib[:, None] * 1e4))
del Ua
print(f"astrocytes {len(ia):,}; {len(gpan)} Xenium panel genes", flush=True)

oof = np.zeros(len(c2))
for tr, te in GroupKFold(5).split(Xpan, c2, donor):
    oof[te] = RidgeCV(alphas=ALPHAS).fit(Xpan[tr], c2[tr]).predict(Xpan[te])
fit = RidgeCV(alphas=ALPHAS).fit(Xpan, c2)
dm = pd.DataFrame({"c2": c2, "oof": oof, "d": donor}).groupby("d").agg(["mean", "size"])
dm = dm[dm[("c2", "size")] >= 50]
coef = pd.Series(fit.coef_, index=gpan.str.upper())
stats = dict(alpha=fit.alpha_, cell_r=pearsonr(c2, oof)[0], donor_r=pearsonr(dm[("c2", "mean")], dm[("oof", "mean")])[0])
print(f"ridge alpha {fit.alpha_:g}; donor-grouped CV r per nucleus {stats['cell_r']:.3f}, donor means {stats['donor_r']:.3f}")
print("  top -:", ", ".join(f"{k} {v:.2f}" for k, v in coef.nsmallest(10).items()))
print("  top +:", ", ".join(f"{k} {v:.2f}" for k, v in coef.nlargest(10).items()), flush=True)

x = ad.read_h5ad(XENIUM_PATH)
xa_idx = np.flatnonzero(x.obs["Subclass"].astype(str).str.startswith("Astrocyte").to_numpy())
Xc = x.X.tocsr()[xa_idx].astype(np.float64)
tot = np.asarray(Xc.sum(1)).ravel()
L = np.log1p(Xc.toarray() / np.where(tot > 0, tot, 1)[:, None] * 1e4)
xg = pd.Index(x.var_names.astype(str)).str.upper()
val = zs(L[:, xg.get_indexer(coef.index)]) @ coef.to_numpy()
xa = pd.DataFrame({"c2": val, "log_counts": np.log(np.where(tot > 0, tot, 1)),
                   "Specimen_ID": x.obs["Specimen_ID"].astype(str).to_numpy()[xa_idx]}, index=x.obs_names[xa_idx])
xa["c2_adj"] = np.nan
for sid, g in xa.groupby("Specimen_ID"):
    xa.loc[g.index, "c2_adj"] = smf.ols("c2 ~ log_counts", g).fit().resid + g["c2"].mean()
stats["xenium_r_counts"] = pearsonr(xa["c2"], xa["log_counts"])[0]
print(f"Xenium astrocytes {len(xa):,}; r(carried C2, log counts) {stats['xenium_r_counts']:+.3f}")
pickle.dump(dict(xa=xa, coef=coef, stats=stats), open(HERE / "astro_c2_xenium.pkl", "wb"))
print("wrote astro_c2_xenium.pkl")
