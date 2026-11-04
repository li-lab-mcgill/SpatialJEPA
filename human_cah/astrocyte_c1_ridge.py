"""Carry astrocyte C1 (seed0) to Xenium through shared genes, as Fig. 7e carries dopamine.

The Fig. 7f gene-weight projection over the 241 shared genes tracks latent C1 within astrocytes
poorly (donor-mean r 0.15) and loses the ADNC association, so here a ridge map from the shared
genes to C1 is fitted within snRNA astrocytes, with donor-grouped CV, and checked for whether its
out-of-fold donor means reproduce the snRNA ADNC association before it is applied to Xenium.
Genes: the 241 shared by the 1,057-gene model panel and Xenium (aligned h5mu, log-normalised), and
all Xenium panel genes found in the full snRNA (UMIs, normalise_total + log1p). z-scored within
astrocytes of each platform. Adds astro_c1_ridge.pkl next to this script.
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
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/astro_c1_adnc"
BASE = Path(os.environ['SPATIALJEPA_HOME'])
CAH = BASE / "outputs/human_cah_transfer/seed0/cah"
XENIUM_PATH = BASE / "data/SEA_AD/sea_ad_cah/CaH_Xenium_final.2026-01-07.h5ad"
SNRNA_PATH = BASE / "data/SEA_AD/sea_ad_cah/CaH_final_nuclei.2025-12-29.h5ad"
ALPHAS = np.logspace(0, 5, 11)
D = pickle.load(open(HERE / "astro_c1_cache.pkl", "rb"))

cells = pd.read_parquet(HERE / "cells.parquet", columns=["Donor ID", "Subclass", "C1_seed0"])
ia = np.flatnonzero((cells["Subclass"].astype(str) == "Astrocyte").to_numpy())
c1 = cells["C1_seed0"].to_numpy()[ia]
donor = cells["Donor ID"].astype(str).to_numpy()[ia]
xen_genes = pd.Index(ad.read_h5ad(XENIUM_PATH, backed="r").var_names.astype(str))
zs = lambda M: (M - M.mean(0)) / M.std(0).clip(1e-6)

# snRNA astrocyte expression: (i) model panel genes from the aligned h5mu, (ii) full snRNA
with h5py.File(CAH / "snrna_spatialmeta_aligned.h5mu", "r") as h:
    r = h["mod/rna"]
    genes = pd.Index(r["var/_index"].asstr()[:]).str.replace("rna:", "", regex=False)
    Xs = sp.csr_matrix((r["X/data"][:], r["X/indices"][:], r["X/indptr"][:]), shape=(len(cells), len(genes)))[ia]
sh = genes.str.upper().isin(xen_genes.str.upper())
X241 = zs(Xs[:, np.flatnonzero(sh)].toarray().astype(np.float64))
g241 = genes[sh]
del Xs

with h5py.File(SNRNA_PATH, "r") as h:                 # astrocyte rows only, by indptr slices
    sn_obs = pd.Index(h["obs"][h["obs"].attrs["_index"]].asstr()[:])
    sn_var = pd.Index(h["var"][h["var"].attrs["_index"]].asstr()[:]).str.upper()
    rows = sn_obs.get_indexer(pd.Index(cells.index.astype(str)).take(ia))   # nuclei are stored in another order
    assert (rows >= 0).all(), "astrocyte barcodes missing from the snRNA file"
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
print(f"astrocytes {len(ia):,}; shared genes: model panel {len(g241)}, full Xenium panel {len(gpan)}")


def ordinal(dm):
    tab = D["snrna_donors"].copy()
    tab["x"] = dm.reindex(tab.index.astype(str)).to_numpy()
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    from scipy.stats import spearmanr
    A = "Overall AD neuropathological Change"; ADNC = ["Not AD", "Low", "Intermediate", "High"]
    f = tab.dropna(subset=["x", "Age at Death", "PMI"])
    z = lambda s: (s - s.mean()) / s.std()
    X = pd.DataFrame({"x": z(f["x"]), "age": z(f["Age at Death"]), "sex_m": (f["Sex"] == "Male").astype(float),
                      "pmi": z(f["PMI"])}, index=f.index)
    y = pd.Series(pd.Categorical(f[A], categories=ADNC, ordered=True), index=f.index)
    res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    b, se = res.params["x"], res.bse["x"]
    return dict(OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=res.pvalues["x"],
                rho=spearmanr(f["x"], f[A].map({l: i for i, l in enumerate(ADNC)}))[0])


res = {}
keep = pd.Series(donor).value_counts().ge(50)
for name, X, g in (("model panel (241)", X241, g241), (f"Xenium panel ({len(gpan)})", Xpan, gpan)):
    oof = np.zeros(len(c1))
    for tr, te in GroupKFold(5).split(X, c1, donor):
        oof[te] = RidgeCV(alphas=ALPHAS).fit(X[tr], c1[tr]).predict(X[te])
    full = RidgeCV(alphas=ALPHAS).fit(X, c1)
    dm = pd.DataFrame({"c1": c1, "oof": oof, "d": donor}).groupby("d").mean()
    dm = dm[keep.reindex(dm.index).fillna(False).to_numpy()]
    res[name] = dict(alpha=full.alpha_, cell_r=pearsonr(c1, oof)[0], donor_r=pearsonr(dm.c1, dm.oof)[0],
                     **{f"or_{k}": v for k, v in ordinal(dm["oof"]).items()},
                     coef=pd.Series(full.coef_, index=g), intercept=full.intercept_, oof_donor=dm["oof"])
    c = res[name]["coef"].sort_values()
    print(f"\n[{name}] alpha {full.alpha_:g}; donor-grouped CV r cell {res[name]['cell_r']:.3f}, donor {res[name]['donor_r']:.3f}; "
          f"OOF donor means vs ADNC OR {res[name]['or_OR']:.2f} [{res[name]['or_lo']:.2f}, {res[name]['or_hi']:.2f}] "
          f"p {res[name]['or_p']:.3f} rho {res[name]['or_rho']:.2f}")
    print("  top -:", ", ".join(f"{k} {v:.2f}" for k, v in c.head(10).items()))
    print("  top +:", ", ".join(f"{k} {v:.2f}" for k, v in c.tail(10)[::-1].items()))

# Xenium astrocytes: same genes, log-normalised over the panel, z-scored within Xenium astrocytes
x = ad.read_h5ad(XENIUM_PATH)
xo = x.obs
xa_idx = np.flatnonzero(xo["Subclass"].astype(str).str.startswith("Astrocyte").to_numpy())
Xx = x.X.tocsr()[xa_idx].astype(np.float64)
tot = np.asarray(x.X.tocsr()[xa_idx].sum(1)).ravel()
Xx = np.log1p(Xx.toarray() / np.where(tot > 0, tot, 1)[:, None] * 1e4)
xg = pd.Index(x.var_names.astype(str)).str.upper()
xen = pd.DataFrame(index=xo.index[xa_idx])
for name in res:
    coef = res[name]["coef"]
    cols = xg.get_indexer(coef.index.str.upper())
    xen[name] = zs(Xx[:, cols]) @ coef.to_numpy()
res["xenium_astro"] = xen
pickle.dump(res, open(HERE / "astro_c1_ridge.pkl", "wb"))
cache = D["xenium"]
xa = cache.loc[xen.index]
for name in xen:
    m = xa.assign(v=xen[name]).groupby(["Donor ID", "Overall AD neuropathological Change"], observed=True)["v"].mean()
    print(f"\n[Xenium, {name}] donor means:\n{m.round(3).to_string()}")
    print("  r with oligodendrocyte density:", round(pearsonr(xen[name], xa["dens_oligo"])[0], 3))
print("wrote", HERE / "astro_c1_ridge.pkl")
