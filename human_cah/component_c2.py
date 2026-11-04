"""CPU part of the supplementary figure on PLS C2 (seed0; Fig. 7 recipe, nothing in Fig. 7 changes).

1  Seed robustness: in seed1/s2 and fix_s0, the component matching seed0's C2 (largest |r| of the
   OT metabolite weights), its RNA/MSI weight agreement, OT-vs-source recovery and rotation null.
2  snRNA: C2 by subclass (mean, median, IQR), AUROC astrocyte + ependymal vs all other nuclei.
3  Xenium: the snRNA C2 gene weights applied to every section's expression, z-scored per section
   (the Fig. 7f recipe); subclass means vs snRNA; local oligodendrocyte density kept for the map.
4  Donor level: donor-mean C2 within each subclass (>= 50 nuclei) vs ADNC (cumulative logit; age, sex,
   PMI), BH over subclasses; the same for C1 for reference.
Writes c2.pkl next to this script.
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
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer"
BASE = Path(os.environ['SPATIALJEPA_HOME'])
ROOT = BASE / "outputs/human_cah_transfer"
A = "Overall AD neuropathological Change"
ADNC = ["Not AD", "Low", "Intermediate", "High"]
z = lambda s: (s - s.mean()) / s.std()
load = lambda r: pickle.load(open(ROOT / f"{r}/cah/pls_msi_ot_reg0.001_sec/results.pkl", "rb"))

R0 = load("seed0")
C2 = "PLS C2"
out = {"W_rna": R0["W"]["target RNA"][C2], "W_msi_ot": R0["W"]["target MSI (OT)"][C2],
       "W_msi_src": R0["W"]["source MSI"][C2], "summary": R0["summary"]}

# 1 seeds
rows = []
for run in ("seed0", "seed1", "seed2", "fix_s0", "fix_s1", "fix_s2"):
    Rr = load(run)
    Wm = Rr["W"]["target MSI (OT)"]
    r = Wm.corrwith(out["W_msi_ot"])
    c = r.abs().idxmax(); sgn = np.sign(r[c])
    wr = Rr["W"]["target RNA"][c] * sgn
    s = Rr["summary"].set_index("component").loc[c]
    rows.append(dict(run=run, matched=c, r_msi_ot=abs(r[c]), r_rna=wr.corr(out["W_rna"].reindex(wr.index)),
                     share_of_stack=s["share_of_stack"], r_ot_vs_source=s["r_ot_vs_source"],
                     null_max=s["r_ot_rotation_null_max"],
                     top_msi_ot=", ".join(m.replace("msi:", "") for m in (Wm[c] * sgn).nlargest(5).index),
                     top_rna_neg=", ".join((wr).nsmallest(5).index)))
out["seeds"] = pd.DataFrame(rows)
print("1. C2 across seeds\n", out["seeds"].round(3).to_string(index=False), flush=True)

# 2 snRNA by subclass
cells = pd.read_parquet(HERE.parent / "astro_c1_adnc" / "cells.parquet", columns=["Donor ID", "Subclass"])
sub = cells["Subclass"].astype(str).to_numpy()
sc = R0["scores"]
assert sc.index.astype(str).equals(cells.index.astype(str))
c2 = sc[C2].to_numpy()
g = pd.DataFrame({"C2": c2, "C1": sc["PLS C1"].to_numpy(), "sub": sub})
out["by_subclass"] = g.groupby("sub")["C2"].describe()[["count", "mean", "25%", "50%", "75%"]].sort_values("mean")
ae = np.isin(sub, ["Astrocyte", "Ependymal"])
out["auroc_astro_epend_low"] = roc_auc_score(ae, -c2)
out["auroc_astro_low"] = roc_auc_score(sub == "Astrocyte", -c2)
out["r_c1_c2"] = np.corrcoef(c2, g["C1"])[0, 1]
print("\n2. C2 by subclass\n", out["by_subclass"].round(2).to_string(),
      f"\nAUROC low C2 for astrocyte+ependymal {out['auroc_astro_epend_low']:.3f}, astrocyte {out['auroc_astro_low']:.3f}; "
      f"r(C1, C2) over nuclei {out['r_c1_c2']:+.3f}", flush=True)

# 3 Xenium projection (per-section z, as Fig. 7f)
x = ad.read_h5ad(BASE / "data/SEA_AD/sea_ad_cah/CaH_Xenium_final.2026-01-07.h5ad")
genes = pd.Index(x.var_names.astype(str)).str.upper()
w = out["W_rna"][~out["W_rna"].index.duplicated()]; w = w.groupby(w.index.str.upper()).first()
on = genes.isin(w.index); wv = w.reindex(genes[on]).to_numpy()
X = x.X.tocsr().astype(np.float64); tot = np.asarray(X.sum(1)).ravel()
X = sp.diags(1e4 / np.where(tot > 0, tot, 1)) @ X; X.data = np.log1p(X.data)
L = X[:, np.flatnonzero(on)].toarray(); del X
secs = x.obs["Specimen_ID"].astype(str).to_numpy()
proj = np.empty(len(L))
for s_ in np.unique(secs):
    m = secs == s_
    proj[m] = np.clip((L[m] - L[m].mean(0)) / L[m].std(0).clip(1e-6), -10, 10) @ wv / np.abs(wv).sum()
xs = x.obs["Subclass"].astype(str).str.replace("_Subclass", "", regex=False).to_numpy()
cache = pickle.load(open(HERE.parent / "astro_c1_adnc" / "astro_c1_cache.pkl", "rb"))["xenium"]
xen = pd.DataFrame({"C2_proj": proj, "Subclass": xs, "Specimen_ID": secs}, index=x.obs_names)
xen = xen.join(cache[["x", "y", "dens_oligo"]])
xm = xen.groupby("Subclass")["C2_proj"].mean()
sm = out["by_subclass"]["mean"]
both = xm.index.intersection(sm.index)
out["xenium"] = xen
out["xenium_subclass_mean"] = xm
out["xenium_vs_snrna_subclass_rho"] = spearmanr(xm[both], sm[both])[0]
print(f"\n3. Xenium C2 projection over {on.sum()} genes; subclass means vs snRNA rho {out['xenium_vs_snrna_subclass_rho']:.2f} "
      f"({len(both)} subclasses)\n", xm.sort_values().round(3).to_string(), flush=True)

# 4 donor level
don = pickle.load(open(HERE.parent / "astro_c1_adnc" / "astro_c1_cache.pkl", "rb"))["snrna_donors"]; don.index = don.index.astype(str)
d = cells["Donor ID"].astype(str).to_numpy()
res = []
for comp in ("PLS C2", "PLS C1"):
    v = sc[comp].to_numpy()
    tab = pd.DataFrame({"v": v, "d": d, "s": sub}).groupby(["s", "d"])["v"].agg(["mean", "size"])
    for s_ in np.unique(sub):
        t = tab.loc[s_]; t = t[t["size"] >= 50]["mean"]
        f = don.join(t.rename("x"), how="inner").dropna(subset=["Age at Death", "PMI"])
        if len(f) < 30:
            continue
        X0 = pd.DataFrame({"x": z(f["x"]), "age": z(f["Age at Death"]), "sex": (f["Sex"] == "Male").astype(float), "pmi": z(f["PMI"])})
        fit = OrderedModel(pd.Series(pd.Categorical(f[A], categories=ADNC, ordered=True), index=f.index), X0, distr="logit").fit(method="bfgs", disp=False)
        res.append(dict(component=comp, subclass=s_, n=len(f), OR=np.exp(fit.params["x"]), p=fit.pvalues["x"]))
res = pd.DataFrame(res)
res["q"] = res.groupby("component")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
out["donor_adnc"] = res
print("\n4. donor-mean component within subclass vs ADNC\n", res.sort_values(["component", "p"]).round(3).to_string(index=False), flush=True)
pickle.dump(out, open(HERE / "c2.pkl", "wb"))
print("wrote c2.pkl")
