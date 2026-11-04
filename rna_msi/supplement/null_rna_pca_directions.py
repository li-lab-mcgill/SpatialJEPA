try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
# No-teacher control for null_random_latent_directions.py: the same analysis on an RNA-only latent
# (10 PCs of the student's 1,806 panel genes, fit on intact source spots, projected onto each
# target with per-dataset gene standardisation). Components = SVD of the source MSI block on it.
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import joblib
from scipy.special import softmax
from scipy.stats import rankdata
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import msi_components_common as P
import shared_msi_components as SC

src = open(Path(__file__).parent / "null_random_latent_directions.py").read()
helpers = src[src.index("def weights"):src.index("# source dopamine weight per direction")]
exec(src[src.index("# NMF-22 reference"):src.index("def weights")])
exec(helpers)

RUN = sys.argv[1] if len(sys.argv) > 1 else "fixed s3"
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
tag = P.RUNS[RUN]
N, K = 5000, 6

S = SC.load_source(tag)
S = SC.subset_source(S, S["obs"]["lesion"].eq("intact").to_numpy())
joint = ad.read_h5ad(P.MODELS / (P.BASE + tag) / "joint_adata.h5ad", backed="r")
panel = [g.split(":", 1)[-1] for g in joint.var_names[joint.var["type"].eq("ST")]]
gi = S["genes"].get_indexer(panel)
panel = [g for g, i in zip(panel, gi) if i >= 0]


def zgenes(X):
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
    sd = X.std(0); return (X - X.mean(0)) / np.where(sd > 0, sd, 1)


Xs = zgenes(S["rna"].tocsr()[:, S["genes"].get_indexer(panel)])
_, _, Vt = np.linalg.svd(Xs, full_matrices=False)
L = Vt[:10].T                                              # genes x 10 loadings
Zs = SC.standardize(Xs @ L)
Cm = SC.corr_matrix(Zs, S["msi"], P.column_stats(S["msi"]))
da = S["metabolites"].get_loc(P.DA)
U = np.linalg.svd(Cm / np.linalg.norm(Cm), full_matrices=False)[0][:, :K]
U = U * np.sign(weights(Cm[:, [da]], Zs, U)[0])
dw_comp = weights(Cm[:, [da]], Zs, U)[0]
k_da = int(np.argmax(dw_comp))
rng = np.random.default_rng(0)
Ur = rng.standard_normal((10, N)); Ur /= np.linalg.norm(Ur, axis=0)
Ur *= np.sign(weights(Cm[:, [da]], Zs, Ur)[0])
dw_rand = weights(Cm[:, [da]], Zs, Ur)[0]
print(f"[pca] {len(panel)} panel genes; source dopamine weight of components {np.round(dw_comp, 3)} "
      f"(dopamine component C{k_da + 1}); max attainable "
      f"{np.sqrt(Cm[:, da] @ np.linalg.solve(Zs.T @ Zs / (len(Zs) - 1), Cm[:, da])):.3f}")

rows, pts = [], {}
for target in ("spatial", "multiome"):
    T = P.load_target(target)
    tg = T["genes"].get_indexer(panel)
    ok = tg >= 0
    Zt = SC.standardize(zgenes(T["rna"].tocsr()[:, tg[ok]]) @ L[ok])
    genes = pd.Index(T["genes"]).str.upper()
    keep = ~genes.duplicated() & genes.isin(spec.index)
    member = np.asarray(genes[keep].isin(nmf_top))
    C = SC.corr_matrix(Zt, T["rna"], P.column_stats(T["rna"]))[:, keep]
    a_r, a_c = auroc_cols(weights(C, Zt, Ur), member), auroc_cols(weights(C, Zt, U), member)
    name = f"rnaPCA @ {'P22' if target == 'spatial' else 'multiome'}"
    pts[name] = dict(rand=a_r, comp=a_c)
    print(f"[{name}] {ok.sum()} panel genes in target; r(source dopamine weight, NMF AUROC) over {N} "
          f"random directions {np.corrcoef(dw_rand, a_r)[0, 1]:.2f}; random q95 {np.quantile(a_r, .95):.3f}, "
          f"max {a_r.max():.3f}", flush=True)
    for k in range(K):
        rows.append(dict(model=name, component=f"C{k + 1}", is_da=k == k_da, src_dopamine_weight=dw_comp[k],
                         nmf_auroc=a_c[k], pct_vs_random=(a_r < a_c[k]).mean()))
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))
T.to_csv(OUT / f"direction_null_rnaPCA_{RUN.replace(' ', '_')}.csv", index=False)
with open(OUT / f"direction_null_rnaPCA_{RUN.replace(' ', '_')}.pkl", "wb") as h:
    pickle.dump(dict(dw_rand=dw_rand, dw_comp=dw_comp, pts=pts, comps=[f"C{k + 1}" for k in range(K)],
                     c_da=f"C{k_da + 1}"), h)
