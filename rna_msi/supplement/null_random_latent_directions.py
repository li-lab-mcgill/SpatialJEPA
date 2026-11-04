try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
# Supplementary (dopamine dose-response). Usage: null_random_latent_directions.py "<run>" <out_dir>; the RNA-PCA
# control null_rna_pca_directions.py "<run>" <out_dir>; panels plot_dopamine_dose_response.py <out_dir> <out_dir>.
# Random latent directions vs the six MSI components: source dopamine weight (measured MSI,
# intact hemisphere) against NMF-22 convergence of the target gene weights, for the teacher on
# P22, the student on P22 and the student on multiome nuclei. Also an identity-adjusted version
# (target gene weights partial on the MSN module score).
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import anndata as ad
from scipy.special import softmax
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import msi_components_common as P
import shared_msi_components as SC

RUN = sys.argv[1] if len(sys.argv) > 1 else "fixed s3"
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
tag = P.RUNS[RUN]
root = P.OUT / "rna_msi_runs/shared_components/msi_svd_intact"
if RUN != "fixed s3":
    root = root / "runs" / RUN.replace(" ", "_")
R = pickle.load(open(root / "spatial" / "results.pkl", "rb"))
U, comps, c_da = R["U"], R["comps"], R["c_da"]
N = 5000
rng = np.random.default_rng(0)
Ur = rng.standard_normal((U.shape[0], N)); Ur /= np.linalg.norm(Ur, axis=0)

# NMF-22 reference (panel j): softmax specificity, top 50
nmf = joblib.load(P.PROJ / P.BASE / "nmf_dopamine_spatial_xcorr_no_panel_model.joblib")
with np.load(P.PROJ / P.BASE / "nmf_dopamine_spatial_xcorr_no_panel_metrics.npz") as m:
    best = int(np.argmax(m["bivariate_moran_I"]))
joint = ad.read_h5ad(P.MODELS / P.BASE / "joint_adata_no_panel.h5ad", backed="r")
st = joint.var_names[joint.var["type"].eq("ST")]
H = nmf.components_
spec = pd.Series(softmax(H / H.max(axis=1, keepdims=True) / 0.1, axis=0)[best],
                 index=[g.split(":", 1)[1].upper() for g in st]).groupby(level=0).max()
nmf_top = set(spec.nlargest(50).index)


def weights(C, Z0, V):
    sigma = Z0.T @ Z0 / (len(Z0) - 1)
    return C.T @ V / np.sqrt(np.einsum("dk,de,ek->k", V, sigma, V))


def auroc_cols(W, member):
    r = rankdata(W, axis=0)
    n1, n0 = member.sum(), (~member).sum()
    return (r[member].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


# source dopamine weight per direction
S = SC.load_source(tag)
S = SC.subset_source(S, S["obs"]["lesion"].eq("intact").to_numpy())
Zs = SC.standardize(S["Z"])
Cm = SC.corr_matrix(Zs, S["msi"], P.column_stats(S["msi"]))
da = S["metabolites"].get_loc(P.DA)
Ur *= np.sign(weights(Cm[:, [da]], Zs, Ur)[0])        # orient like the components: dopamine >= 0
dw_rand = weights(Cm[:, [da]], Zs, Ur)[0]
dw_comp = weights(Cm[:, [da]], Zs, U)[0]
# dopamine weight beyond striatal identity in the source: partial on the source MSN score
msn_s = P.module_scores(S["rna"].tocsr(), S["genes"])["MSN"].to_numpy()
y_da = SC._standardize_cols(S["msi"][:, [da]])[:, 0]
r_dm = np.corrcoef(y_da, msn_s)[0, 1]
def src_partial(V):
    s = Zs @ V
    r_sm = np.array([np.corrcoef(s[:, j], msn_s)[0, 1] for j in range(V.shape[1])])
    return (weights(Cm[:, [da]], Zs, V)[0] - r_dm * r_sm) / np.sqrt((1 - r_dm ** 2) * (1 - r_sm ** 2))
pw_rand, pw_comp = src_partial(Ur), src_partial(U)
print(f"source r(dopamine, MSN score) {r_dm:.2f}; MSN-partial dopamine weight of components {np.round(pw_comp, 3)}")
dw_max = float(np.sqrt(Cm[:, da] @ np.linalg.solve(Zs.T @ Zs / (len(Zs) - 1), Cm[:, da])))

rows, pts = [], {}
for target, keys in (("spatial", ("teacher", "student")), ("multiome", ("student",))):
    T = P.load_target(target)
    lat = SC.target_latents(tag, target, T["cells"])
    genes = pd.Index(T["genes"]).str.upper()
    keep = ~genes.duplicated() & genes.isin(spec.index)
    member = np.asarray(genes[keep].isin(nmf_top))
    stats = P.column_stats(T["rna"])
    msn = P.module_scores(T["rna"], T["genes"])["MSN"].to_numpy()
    for key in keys:
        Z0 = SC.standardize(lat[key])
        C = SC.corr_matrix(Z0, T["rna"], stats)[:, keep]
        Wr, Wc = weights(C, Z0, Ur), weights(C, Z0, U)
        a_r, a_c = auroc_cols(Wr, member), auroc_cols(Wc, member)
        # identity-adjusted: partial correlation of each gene with the score given the MSN score
        g_msn = np.nan_to_num(P.plain_corr(T["rna"], stats, msn)[keep])
        def partial(V):
            s = Z0 @ V
            r_sm = np.array([np.corrcoef(s[:, j], msn)[0, 1] for j in range(V.shape[1])])
            Wv = weights(C, Z0, V)
            return (Wv - np.outer(g_msn, r_sm)) / np.sqrt(np.outer(1 - g_msn ** 2, 1 - r_sm ** 2))
        p_r, p_c = auroc_cols(partial(Ur), member), auroc_cols(partial(U), member)
        name = f"{key} @ {'P22' if target == 'spatial' else 'multiome'}"
        pts[name] = dict(rand=a_r, comp=a_c, rand_adj=p_r, comp_adj=p_c)
        for k, c in enumerate(comps):
            rows.append(dict(model=name, component=c, is_da=c == c_da, src_dopamine_weight=dw_comp[k],
                             src_dopamine_weight_msn_adj=pw_comp[k],
                             nmf_auroc=a_c[k], pct_vs_random=(a_r < a_c[k]).mean(),
                             nmf_auroc_msn_adj=p_c[k], pct_vs_random_msn_adj=(p_r < p_c[k]).mean()))
        rho = np.corrcoef(dw_rand, a_r)[0, 1]; rho_adj = np.corrcoef(pw_rand, p_r)[0, 1]
        print(f"[{name}] over {N} random directions: r(source dopamine weight, NMF AUROC) {rho:.2f}; "
              f"MSN-adjusted both sides {rho_adj:.2f}; random AUROC q95 {np.quantile(a_r, .95):.3f}, "
              f"max {a_r.max():.3f}; adjusted q95 {np.quantile(p_r, .95):.3f}", flush=True)

T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"source dopamine weight: components {np.round(dw_comp, 3)}, max attainable (OLS) {dw_max:.3f}, "
      f"random q95 {np.quantile(np.abs(dw_rand), .95):.3f}")
print(T.round(3).to_string(index=False))
T.to_csv(OUT / f"direction_null_{RUN.replace(' ', '_')}.csv", index=False)
with open(OUT / f"direction_null_{RUN.replace(' ', '_')}.pkl", "wb") as h:
    pickle.dump(dict(dw_rand=dw_rand, dw_comp=dw_comp, pw_rand=pw_rand, pw_comp=pw_comp, dw_max=dw_max, pts=pts, comps=comps, c_da=c_da), h)
