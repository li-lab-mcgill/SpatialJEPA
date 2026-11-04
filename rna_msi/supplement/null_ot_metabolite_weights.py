try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
# Supplementary (replaces the metabolite-weight recovery and held-out-animal figures): the main-text
# recipe for metabolite weights in RNA-only targets, applied to random latent directions, with the
# target latent randomly rotated before pairing as the control. Laid out like the dose-response figure.
# Usage: null_ot_metabolite_weights.py <out_dir> [--draw-only]
#   directions  the 5,000 random unit directions of null_random_latent_directions.py (same seed, same
#               orientation: measured source dopamine weight >= 0) and the six MSI components
#   x           dopamine weight of the direction in the source (measured MSI, intact hemisphere)
#   y           dopamine weight recovered in the target: entropic OT pairing of target cells with
#               source spots (reg 0.01), PLS decoder fitted on the target, R_m x corr(decoded, score)
#               (fig567 ot_msi_weights; ot_msi_recovery.py `recovered`)
#   control     the same with the target latent randomly rotated before pairing only (alignment to
#               the source broken, target distribution kept); 20 rotations, each scoring 250 of the
#               random directions and all six components
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
p = argparse.ArgumentParser()
p.add_argument("out_dir", type=Path)
p.add_argument("--draw-only", action="store_true")
p.add_argument("--run", default="fixed s3", help="model run (msi_components_common.RUNS)")
args = p.parse_args()
OUT = args.out_dir; OUT.mkdir(parents=True, exist_ok=True)
N, N_ROT = 5000, 20
MODELS = (("teacher @ P22", "spatial", "teacher", "Teacher latent, P22"),
          ("student @ multiome", "multiome", "student", "Student latent, multiome"))


def weights(C, Z0, V):
    """corr(feature, Z0 v) for every feature and every column v of V (dopamine_direction_null.weights)."""
    sigma = Z0.T @ Z0 / (len(Z0) - 1)
    return C.T @ V / np.sqrt(np.einsum("dk,de,ek->k", V, sigma, V))


def compute():
    import msi_components_common as P
    import shared_msi_components as SC

    tag = P.RUNS[args.run]
    comp_root = P.OUT / "rna_msi_runs/shared_components/msi_svd_intact"
    if args.run != "fixed s3":
        comp_root = comp_root / "runs" / args.run.replace(" ", "_")
    R = pickle.load(open(comp_root / "spatial" / "results.pkl", "rb"))
    U, comps, c_da = R["U"], R["comps"], R["c_da"]
    rng = np.random.default_rng(0)                       # as null_random_latent_directions.py
    Ur = rng.standard_normal((U.shape[0], N)); Ur /= np.linalg.norm(Ur, axis=0)
    S = SC.load_source(tag)
    S = SC.subset_source(S, S["obs"]["lesion"].eq("intact").to_numpy())
    Ys, Zs = SC._standardize_cols(S["msi"]), SC.standardize(S["Z"])
    Cm = SC.corr_matrix(Zs, S["msi"], P.column_stats(S["msi"]))
    da = S["metabolites"].get_loc(P.DA)
    Ur *= np.sign(weights(Cm[:, [da]], Zs, Ur)[0])
    V = np.column_stack([U, Ur])                         # components first
    src = weights(Cm, Zs, V)                             # metabolites x directions
    res = dict(comps=comps, c_da=c_da, src_da=src[da], models={})
    rows = []
    rot_rng = np.random.default_rng(1)
    for name, target, key, _ in MODELS:
        T = P.load_target(target)
        Zt = SC.standardize(SC.target_latents(tag, target, T["cells"])[key])
        groups = (SC.spatial_blocks(SC.F.coordinates(tag, target, T["cells"])) if target == "spatial"
                  else np.random.default_rng(0).integers(0, 5, len(T["cells"])))

        def recover(Zpair):
            Yhat = SC.ot_barycentre(Zs, Ys, Zpair, reg=SC.OT_REG)
            dec = SC.fit_msi_predictor(Zt, Yhat, groups, method="simpls")
            return SC.ported_msi_weights(dec, Zt, V)[0]  # metabolites x directions

        W = recover(Zt)
        r_all = SC.col_corr(W, src) if hasattr(SC, "col_corr") else np.array(
            [np.corrcoef(W[:, j], src[:, j])[0, 1] for j in range(V.shape[1])])
        rot_da, rot_comp, rot_r = np.full(N, np.nan), [], np.full(N, np.nan)
        for q in range(N_ROT):
            Q = np.linalg.qr(rot_rng.standard_normal((Zt.shape[1], Zt.shape[1])))[0]
            Wq = recover(Zt @ Q)
            sel = np.arange(q, N, N_ROT)
            rot_da[sel] = Wq[da, len(comps) + sel]
            rot_r[sel] = [np.corrcoef(Wq[:, len(comps) + j], src[:, len(comps) + j])[0, 1] for j in sel]
            rot_comp.append(Wq[da, :len(comps)])
            print(f"[{name}] rotation {q + 1}/{N_ROT}", flush=True)
        rot_comp = np.array(rot_comp)
        res["models"][name] = dict(da=W[da], r_all=r_all, rot_da=rot_da, rot_r=rot_r, rot_comp=rot_comp)
        x = res["src_da"][len(comps):]
        for kind, y, rr in (("aligned", W[da, len(comps):], r_all[len(comps):]), ("rotated", rot_da, rot_r)):
            rows.append(dict(model=name, latent=kind, component="random (5,000)",
                             r_dopamine_weight_over_directions=np.corrcoef(x, y)[0, 1],
                             median_r_all_metabolites=np.median(rr)))
        for j, c in enumerate(comps):
            rows.append(dict(model=name, latent="aligned", component=c, src_dopamine_weight=res["src_da"][j],
                             recovered_dopamine_weight=W[da, j], r_all_metabolites=r_all[j]))
            rows.append(dict(model=name, latent="rotated (median of 20)", component=c,
                             src_dopamine_weight=res["src_da"][j], recovered_dopamine_weight=np.median(rot_comp[:, j])))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "ot_direction_null.csv", index=False)
    with open(OUT / "ot_direction_null.pkl", "wb") as h:
        pickle.dump(res, h)
    pd.set_option("display.width", 220)
    print(D.round(3).to_string(index=False))


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = pickle.load(open(OUT / "ot_direction_null.pkl", "rb"))
    comps, c_da = res["comps"], res["c_da"]
    nc = len(comps)
    x, xc = res["src_da"][nc:], res["src_da"][:nc]
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), sharex=True, sharey=True)
    for col, (name, _t, _k, title) in enumerate(MODELS):
        m = res["models"][name]
        for row, (y, yc, sub) in enumerate(((m["da"][nc:], m["da"][:nc], "aligned latent (main-text recipe)"),
                                            (m["rot_da"], np.median(m["rot_comp"], 0),
                                             "target latent rotated before pairing"))):
            ax = axes[row, col]
            ax.scatter(x, y, s=2, color="0.75", lw=0, rasterized=True)
            lo = min(x.min(), y.min()) - 0.05; hi = max(x.max(), y.max()) + 0.05
            ax.plot([-1, 1], [-1, 1], ls="--", color="0.45", lw=0.7, zorder=0)
            ax.axhline(0, color="0.6", lw=0.5, zorder=0)
            for j, c in enumerate(comps):
                red = c == c_da
                ax.scatter(xc[j], yc[j], s=40 if red else 26, color="#b2182b" if red else "0.1",
                           edgecolor="white", lw=0.8, zorder=3)
                ax.annotate(c, (xc[j], yc[j]), xytext=(5, 0), textcoords="offset points", va="center",
                            fontsize=8, fontweight="bold" if red else "normal")
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.97, 0.04, f"5,000 random directions (grey)\nr = {r:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color="0.3")
            ax.set_title(f"{title}\n{sub}", fontsize=8)
            if row == 1:
                ax.set_xlabel("dopamine weight in source (measured MSI)")
            if col == 0:
                ax.set_ylabel("dopamine weight recovered in target")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_xlim(-0.05, max(x.max(), xc.max()) + 0.08)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"S_Fig5_ot_direction_null.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}/S_Fig5_ot_direction_null.png")


if not args.draw_only:
    compute()
draw()
