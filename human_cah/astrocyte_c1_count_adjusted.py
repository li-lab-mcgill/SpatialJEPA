"""Follow-up to astro_c1_checks.py: Xenium location in microns, and transcript-count dependence.

Xenium obsm['spatial'] is in morphology-image pixels (0.2125 um; median nearest-cell spacing 45 px).
1  snRNA: within-astrocyte correlation of latent C1 with log UMIs, and donor level.
2  Xenium: ridge-carried astrocyte C1 vs log transcript counts; a count-adjusted C1 (residual of a
   per-section regression on log counts); location effects (tissue edge < 100 um, ventricle wall =
   < 200 um from an ependymal cell, oligodendrocyte density) before and after count adjustment;
   donor means vs snRNA.
3  Fig. 7 section: does the panel-g pattern survive count adjustment (smoothed maps compared)?
Adds the count-adjusted value to astro_c1_checks2.pkl for an alternative panel g.
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

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/astro_c1_adnc"
UM = 0.2125
A = "Overall AD neuropathological Change"
ADNC = ["Not AD", "Low", "Intermediate", "High"]
SEC7 = "H21.33.002.CX18.CaH.02.01"

# 1. snRNA
cells = pd.read_parquet(HERE / "cells.parquet", columns=["Donor ID", "Subclass", "Number of UMIs", "C1_seed0"])
a = cells[cells["Subclass"].astype(str) == "Astrocyte"].copy()
a["lu"] = np.log(pd.to_numeric(a["Number of UMIs"].astype(str), errors="coerce"))
print(f"1. snRNA astrocytes: r(C1, log UMI) = {pearsonr(a['C1_seed0'], a['lu'])[0]:+.3f} per nucleus;",
      f"donor means r = {pearsonr(*a.groupby('Donor ID', observed=True)[['C1_seed0', 'lu']].mean().to_numpy().T)[0]:+.3f}")

# 2. Xenium
CK = pickle.load(open(HERE / "astro_c1_checks.pkl", "rb"))
X = pickle.load(open(HERE / "astro_c1_cache.pkl", "rb"))["xenium"]
xa = CK["xa"].copy()
xa["log_counts"] = np.log(xa["total_counts"])
xa["d_edge_um"], xa["d_epend_um"] = np.nan, np.nan
for sid, idx in X.groupby("Specimen_ID", observed=True).indices.items():
    xy = X[["x", "y"]].to_numpy()[idx] * UM
    ij = np.floor((xy - xy.min(0)) / 20.0).astype(int) + 3                 # 20-um tissue mask
    mask = np.zeros(ij.max(0) + 4, bool); mask[ij[:, 0], ij[:, 1]] = True
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    dist = ndimage.distance_transform_edt(mask) * 20.0
    obs = X.index[idx]
    ina = obs.isin(xa.index)
    xa.loc[obs[ina], "d_edge_um"] = dist[ij[ina, 0], ij[ina, 1]]
    ep = X["Subclass"].to_numpy()[idx] == "Ependymal"
    if ep.sum() >= 20:
        d, _ = NearestNeighbors(n_neighbors=1).fit(xy[ep]).kneighbors(xy[ina])
        xa.loc[obs[ina], "d_epend_um"] = d[:, 0]
xa["edge"] = (xa["d_edge_um"] < 100).astype(float)
xa["vwall"] = (xa["d_epend_um"] < 200).astype(float)                      # NaN-free: sections without ependyma -> False
xa["c1_adj"] = np.nan
for sid, g in xa.groupby("Specimen_ID", observed=True):
    res = smf.ols("c1 ~ log_counts", g).fit()
    xa.loc[g.index, "c1_adj"] = res.resid + g["c1"].mean()                 # keep the section mean
print(f"\n2. Xenium astrocytes: rho(C1, log counts) {spearmanr(xa['c1'], xa['log_counts'])[0]:+.3f};",
      f"within-section mean r {xa.groupby('Specimen_ID', observed=True).apply(lambda g: pearsonr(g['c1'], g['log_counts'])[0]).mean():+.3f}")
for lab, m in (("< 100 um from the tissue edge", xa["edge"] == 1), ("< 200 um from an ependymal cell", xa["vwall"] == 1),
               ("rest", (xa["edge"] == 0) & (xa["vwall"] == 0))):
    print(f"  {lab:34s} {m.mean():6.1%} of astrocytes; mean C1 {xa.loc[m, 'c1'].mean():+.3f}, count-adjusted {xa.loc[m, 'c1_adj'].mean():+.3f}, "
          f"median counts {xa.loc[m, 'total_counts'].median():.0f}")
for y in ("c1", "c1_adj"):
    f = smf.ols(f"{y} ~ edge + vwall + dens_oligo + C(Specimen_ID)", xa).fit()
    print(f"  {y}: edge {f.params['edge']:+.3f} (t {f.tvalues['edge']:.0f}), ventricle wall {f.params['vwall']:+.3f} (t {f.tvalues['vwall']:.0f}), "
          f"oligodendrocyte density {f.params['dens_oligo']:+.3f} (t {f.tvalues['dens_oligo']:.0f})")
snr = pickle.load(open(HERE / "astro_c1_cache.pkl", "rb"))["snrna_donors"]["C1_seed0"]; snr.index = snr.index.astype(str)
adnc = xa.groupby("Donor ID", observed=True)[A].first(); adnc.index = adnc.index.astype(str)
cnt = xa.groupby("Donor ID", observed=True)["log_counts"].mean(); cnt.index = cnt.index.astype(str)
for y, m, lab in (("c1", slice(None), "all"), ("c1_adj", slice(None), "count-adjusted"),
                  ("c1_adj", (xa["edge"] == 0) & (xa["vwall"] == 0), "count-adjusted, no edge/ventricle wall")):
    dm = xa.loc[m].groupby("Donor ID", observed=True)[y].mean(); dm.index = dm.index.astype(str)
    print(f"  donor means ({lab}): rho vs snRNA {spearmanr(dm, snr.reindex(dm.index))[0]:+.2f} (p {spearmanr(dm, snr.reindex(dm.index))[1]:.3f}), "
          f"vs ADNC {spearmanr(dm, adnc.reindex(dm.index).map({l: i for i, l in enumerate(ADNC)}))[0]:+.2f}, vs mean log counts {spearmanr(dm, cnt.reindex(dm.index))[0]:+.2f}")

# 3. Fig. 7 section
s = xa[xa["Specimen_ID"] == SEC7]
xy = X.loc[s.index, ["x", "y"]].to_numpy()
_, nb = NearestNeighbors(n_neighbors=31).fit(xy).kneighbors(xy)
sm = {y: s[y].to_numpy()[nb].mean(1) for y in ("c1", "c1_adj", "log_counts", "dens_oligo")}
print(f"\n3. Fig. 7 section ({len(s):,} astrocytes, {int(s['vwall'].sum())} within 200 um of ependymal cells, {int(s['edge'].sum())} within 100 um of the edge):")
print(f"  smoothed maps: r(C1, count-adjusted C1) {pearsonr(sm['c1'], sm['c1_adj'])[0]:+.2f}; r(C1, log counts) {pearsonr(sm['c1'], sm['log_counts'])[0]:+.2f}; "
      f"r(C1, oligodendrocyte density) {pearsonr(sm['c1'], sm['dens_oligo'])[0]:+.2f}; r(count-adjusted C1, oligodendrocyte density) {pearsonr(sm['c1_adj'], sm['dens_oligo'])[0]:+.2f}")
pickle.dump(xa, open(HERE / "astro_c1_checks2.pkl", "wb"))
