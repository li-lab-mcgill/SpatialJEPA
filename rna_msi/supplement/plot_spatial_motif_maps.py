try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Supplementary to Fig. 6d: the Fig. 6b motifs placed on the source tissue through the student latent.
# (Column and stats labels still say "6c", the panel's letter when this was written; the figure says 6b.)
#
# Only RNA is aligned: the student latent of the multiome nuclei (from their RNA) is paired with
# the source Visium spots (RNA + MSI, V11L12-109_B1, both hemispheres) by the entropic OT of Fig. 5i
# (fig567 ot_barycentre, reg OT_REG, standardised latents, uniform marginals). The coupling places
# each nucleus on the source tissue, and its ATAC follows because RNA and ATAC come from the same
# nucleus: every spot gets the coupling-weighted mean motif accessibility of its nuclei, as pychromVAR
# deviation z-scores (all peaks; fig6_chromvar_multiome) residualised on log ATAC depth. The same recipe
# carries the source MSI to the nuclei in 5i (there on the intact hemisphere's spots), so the
# measured dopamine map and the motif maps share one spatial reference.
#
#   motifs     the top 15 of the GC-adjusted motif panel (Fig. 6b) by odds ratio (--ranking, fig56's
#              motifs_gc/pls_C2_gmm_d2msn_motif_ranking.csv;
#              JASPAR versions of one motif averaged), their mean, EGR1 (RNA-ATAC concordant at FDR on
#              both sides; Fig. 6d of fig56_pls_manuscript draws ZEB1, PBX3, MEIS1, EGR1), and
#              SOX10 / NEUROD2 as controls
#   stats      per motif map: AUROC striatum vs other spots (all, and within each hemisphere), Moran's I
#              (6-NN spot graph); null: the nuclei latent randomly rotated before OT (N_ROT);
#              robustness: each spot the mean of its K_NN nearest nuclei instead of OT
#   alignment  share of each spot's coupling mass from MSN-cluster nuclei
#
# Usage: plot_spatial_motif_maps.py --stage compute|draw <out_dir> [--ranking <motif_ranking.csv>]
# Env: $PY_ANALYSIS (GPU for OT)

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import msi_components_common as P  # noqa: E402
import shared_msi_components as SC  # noqa: E402

CHROMVAR = SC.OUT_DEFAULT / "chromvar_multiome" / "p5e-05_pychromvar"
RANKING = P.OUT / "figure_bundle_fig567_2026-10-20" / "work" / "fig56" / "motifs_gc" / "pls_C2_gmm_d2msn_motif_ranking.csv"
N_TOP, N_ROT, K_NN = 15, 20, 50
CONCORDANT = ["EGR1"]
CONTROLS = {"SOX10": "oligodendrocyte", "Neurod2": "excitatory neuron"}
SHOW = ["MEIS3", "JUNB"]          # leading 6b motifs not already mapped in main Fig. 6d (ZEB1, PBX3, MEIS1, EGR1)
SET = f"6c top {N_TOP} (mean)"
INK, MUTED, GREY = "#222222", "#6b6b6b", "#b5b5b5"


def base_name(names):
    return pd.Series(names).str.replace(r" \(MA[0-9.]+\)$", "", regex=True).to_numpy()


def random_rotation(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def morans_i(xy, V, k=6):
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    Vc = V - V.mean(0)
    return (Vc * Vc[idx].mean(1)).sum(0) / (Vc ** 2).sum(0)


def aurocs(V, labels):
    from scipy.stats import rankdata

    r = rankdata(V, axis=0)
    n1 = labels.sum()
    return (r[labels].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(labels) - n1))


# --------------------------------------------------------------------------------- compute
def compute(out, ranking_csv, run="fixed s3"):
    from sklearn.neighbors import NearestNeighbors

    tag = P.RUNS[run]
    S = SC.load_source(tag)
    T = P.load_target("multiome")
    Z0s = SC.standardize(S["Z"])
    Z0t = SC.standardize(SC.target_latents(tag, "multiome", T["cells"])["student"])
    xy = S["xy"]
    striatum = S["obs"]["region"].eq("striatum").to_numpy()
    intact = S["obs"]["lesion"].eq("intact").to_numpy()

    # motif deviations, depth-residualised; the motifs of 6c (versions averaged) and the controls
    z = np.load(CHROMVAR / "deviation_z.npy")
    cols = pd.read_csv(CHROMVAR / "deviation_columns.csv")["annotation"]
    keep = cols.str.startswith("all|").to_numpy()
    z, motifs = z[:, keep].astype(float), cols[keep].str.split("|", n=1).str[1].to_numpy()
    ld = np.log(pd.read_csv(CHROMVAR / "nucleus_depth.csv")["depth"].to_numpy())
    A = np.column_stack([np.ones(len(ld)), ld])
    z = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    z /= z.std(0, ddof=1)
    base = base_name(motifs)
    ranking = pd.read_csv(ranking_csv).dropna(subset=["rank_by_odds_ratio_among_robust"])
    top = (ranking.sort_values("rank_by_odds_ratio_among_robust").drop_duplicates("motif_name")   # as drawn in 6c
           ["motif_name"].head(N_TOP).tolist())
    named = {}
    for name in top + CONCORDANT + list(CONTROLS):
        sel = base == name
        if not sel.any():
            raise ValueError(f"{name}: not among the chromVAR motifs")
        named[name] = z[:, sel].mean(1)
    named[SET] = np.mean([named[n] for n in top], axis=0)
    cl = pd.Series(T["clusters"]).astype(str).to_numpy()
    msn = P.module_scores(T["rna"], T["genes"])["MSN"].to_numpy()
    cm = pd.Series(msn).groupby(cl).mean()
    is_msn = np.isin(cl, cm.index[cm > 0.5 * msn.std() + msn.mean()]).astype(float)
    V = np.column_stack([z] + list(named.values()) + [is_msn])
    names = list(motifs) + list(named) + ["MSN-cluster share"]
    print(f"[data] {len(Z0t):,} nuclei -> {len(Z0s):,} spots; {len(motifs)} motifs; 6c top {N_TOP}: {', '.join(top)}",
          flush=True)

    # spot maps: OT (Fig. 5i plan, spots x nuclei), rotated-latent OT, kNN
    maps = SC.ot_barycentre(Z0t, V, Z0s, reg=SC.OT_REG)
    rng = np.random.default_rng(0)
    rot_auc, rot_moran = [], []
    for _ in range(N_ROT):
        R = SC.ot_barycentre(Z0t @ random_rotation(Z0t.shape[1], rng), V, Z0s, reg=SC.OT_REG)
        rot_auc.append(aurocs(R, striatum)); rot_moran.append(morans_i(xy, R))
    rot_auc, rot_moran = np.array(rot_auc), np.array(rot_moran)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(Z0t).kneighbors(Z0s, return_distance=False)
    knn = V[nn].mean(1)

    stats = pd.DataFrame({"map": names,
                          "auroc_striatum": aurocs(maps, striatum),
                          "auroc_striatum_intact": aurocs(maps[intact], striatum[intact]),
                          "auroc_striatum_lesioned": aurocs(maps[~intact], striatum[~intact]),
                          "moran_I": morans_i(xy, maps),
                          "auroc_striatum_rotation_mean": rot_auc.mean(0),
                          "auroc_striatum_rotation_q975": np.quantile(rot_auc, 0.975, axis=0),
                          "pct_rotation_auroc": (rot_auc < aurocs(maps, striatum)).mean(0),
                          "moran_I_rotation_mean": rot_moran.mean(0),
                          "auroc_striatum_knn": aurocs(knn, striatum)})
    stats["kind"] = (["motif"] * len(motifs) + ["6c"] * len(top) + ["concordant"] * len(CONCORDANT)
                     + ["control"] * len(CONTROLS) + ["6c set", "alignment"])
    stats.to_csv(out / "spatial_motif_map_stats.csv", index=False)
    show = list(named) + ["MSN-cluster share"]
    pd.DataFrame(maps[:, len(motifs):], columns=show, index=S["cells"]).assign(
        x=xy[:, 0], y=xy[:, 1], striatum=striatum, intact=intact,
        dopamine_measured=S["msi"][:, S["metabolites"].get_loc(P.DA)],
        msn_score_source=P.module_scores(S["rna"].tocsr(), S["genes"])["MSN"].to_numpy()).to_csv(
        out / "spatial_motif_maps.csv.gz")
    pd.DataFrame(knn[:, len(motifs):], columns=show, index=S["cells"]).to_csv(out / "spatial_motif_maps_knn.csv.gz")

    pd.set_option("display.width", 220)
    print(stats[stats.kind != "motif"].round(3).to_string(index=False))
    m = stats[stats.kind == "motif"].sort_values("auroc_striatum", ascending=False).reset_index(drop=True)
    print("all motifs: median striatum AUROC", round(m.auroc_striatum.median(), 3), "| top 15:")
    print(m.head(15)[["map", "auroc_striatum", "moran_I", "auroc_striatum_rotation_q975", "auroc_striatum_knn"]]
          .round(3).to_string(index=False))
    for n in top:
        hit = m.index[base_name(m["map"]) == n]
        print(f"  {n}: striatum-AUROC rank {', '.join(str(i + 1) for i in hit)} of {len(m)}")


# ------------------------------------------------------------------------------------ draw
def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def _map(ax, D, v, title, cmap="viridis", colors=None):
    """Spots as overlapping discs (1.7x the 24-unit pitch, as Fig. 5a's large markers), which also
    covers the runs of spots missing from the source section."""
    w, h = (np.ptp(D[c]) * 1.04 for c in ("x", "y"))
    box = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    pt_per_unit = min(box.width * fw / w, box.height * fh / h) * 72
    size = (1.7 * 24 * pt_per_unit) ** 2
    if colors is None:
        lo, hi = np.nanpercentile(v, [2, 98])
        sc = ax.scatter(D["x"], D["y"], c=np.clip(v, lo, hi), s=size, cmap=cmap, lw=0, rasterized=True)
    else:
        sc = ax.scatter(D["x"], D["y"], c=colors, s=size, lw=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=8, color=INK)
    return sc


def draw(out, ranking_csv=None, run=None):
    D = pd.read_csv(out / "spatial_motif_maps.csv.gz", index_col=0)
    St = pd.read_csv(out / "spatial_motif_map_stats.csv")
    st = St[St.kind.ne("motif")].set_index("map")                 # named maps (6c motifs recur among all)
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "font.family": "DejaVu Sans", "axes.linewidth": 0.7})
    fig = plt.figure(figsize=(11, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)
    lab = lambda m: f"striatum AUROC {st.loc[m, 'auroc_striatum']:.2f} (rotated {st.loc[m, 'auroc_striatum_rotation_mean']:.2f})"

    # row 1: reference maps, the alignment, and the 6c set
    r1 = gs[0].subgridspec(1, 4, wspace=0.05).subplots()
    region_colors = {"intact striatum": "#1f77b4", "lesioned striatum": "#ff7f0e", "other": "#d9d9d9"}
    region = np.where(D["striatum"], np.where(D["intact"], 0, 1), 2)
    _map(r1[0], D, None, "source section\n(Visium RNA + MSI)", colors=np.array(list(region_colors.values()))[region])
    from matplotlib.patches import Patch
    r1[0].legend([Patch(color=c) for c in region_colors.values()], region_colors, frameon=False, fontsize=7,
                 loc="lower right", bbox_to_anchor=(1.08, 0.0), handlelength=1.0)
    _map(r1[1], D, np.log1p(D["dopamine_measured"]), "measured dopamine\n(MSI, log1p)")
    _map(r1[2], D, D["MSN-cluster share"], "MSN-cluster share of mapped nuclei\n"
         f"striatum AUROC {st.loc['MSN-cluster share', 'auroc_striatum']:.2f}", cmap="magma")
    _map(r1[3], D, D[SET], f"Fig. 6b top-{N_TOP} motifs (mean)\n{lab(SET)}")

    # row 2: (b) the leading 6b motifs not mapped in main Fig. 6d, beside (c) the control motifs
    r2 = gs[1].subgridspec(1, 4, wspace=0.05).subplots()
    for ax, m in zip(r2[:2], SHOW):
        _map(ax, D, D[m], f"{m}\nstriatum AUROC {st.loc[m, 'auroc_striatum']:.2f}")
    for ax, (m, cell) in zip(r2[2:], CONTROLS.items()):
        _map(ax, D, D[m], f"{m.upper()} ({cell} control)\nstriatum AUROC {st.loc[m, 'auroc_striatum']:.2f}")

    fig.text(0.08, gs[0].get_position(fig).y1 + 0.01, "a", fontsize=16, fontweight="bold")
    for ax, letter in ((r2[0], "b"), (r2[2], "c")):
        fig.text(ax.get_position().x0 - 0.01, gs[1].get_position(fig).y1 + 0.01, letter, fontsize=16,
                 fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(out / f"Fig6_draft_spatial_motif_maps.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", out / "Fig6_draft_spatial_motif_maps.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("compute", "draw"), required=True)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--ranking", type=Path, default=RANKING, help="Fig. 6b motif ranking (compute)")
    p.add_argument("--run", default="fixed s3", help="model run (msi_components_common.RUNS) for compute")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    {"compute": compute, "draw": draw}[a.stage](a.out_dir, a.ranking, a.run)


if __name__ == "__main__":
    main()
