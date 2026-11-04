try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Manuscript Figs. 5 and 6 drawn from the shared PLS components instead of MOFA+.
#
# The figures are produced by the manuscript's own code (5__'s FIG_dopamine_overview and
# dopamine_overview_cells.fig6_board, with the Fig. 6 cells of 5__ computing the cCRE GSEA,
# leading edges and TF-regulon enrichment). Only the two MOFA+ models are replaced, by
# PLSModel: a mofax-like view of shared_msi_components.py's results for one run.
#
#   teacher model (P22 spatial target, "msi_teacher")
#       factors   teacher RNA-expert latent on P22 (z-scored) x the P22 analysis' U
#       weights   rna/atac: corr(P22 feature, teacher component score);
#                 msi_teacher: metabolite weights on P22 (teacher latent), OT-recovered with a
#                 target-fitted PLS decoder (`--msi-weights ot`, default) or ported
#   student model (multiome target, "msi_student")
#       factors   student RNA-expert latent on the nuclei (z-scored) x the multiome U
#       weights   rna/atac: corr(nucleus feature, student score); msi_student: OT-recovered
#                 (default) or ported weights on the nuclei
#
# `--anchor msi --source-spots intact` (defaults): U is the SVD of the source MSI block (intact hemisphere) (no dopamine label, no target
# data; identical latent directions in both analyses) and the figures follow the component
# with the largest |dopamine weight| (FACTOR, read from the results; C3 in the mouse run).
# Ported weights: shared_msi_components.ported_msi_weights. `--anchor dopamine` redraws
# the earlier dopamine-anchored C1 version (source-measured metabolite weights).
# Each view carries the PLS analysis' own feature set (all 17,069 target genes, all
# 66,520 target peaks, the 779 decoder metabolites), not the MOFA models' HVG subsets, so
# panel j and every Fig. 6 test (cCRE GSEA, leading edges, motifs, regulons) rank all
# features. Main Fig. 6 shows D2-MSN on its full cCRE set; MXD, whose cCRE peaks are ~90%
# shared with D2-MSN, moves to a supplementary figure on MXD-only cCREs. Fig. 6 is a GSEA,
# b motifs, c concordance, d ZEB1 / PBX3 / MEIS1 / EGR1 accessibility of the multiome nuclei
# placed on the source section through the student latent (--motif-maps, from
# fig567_supplement/plot_spatial_motif_maps.py); a-c sit in one row and the board is drawn at
# print width (fig6_d2msn_board), from fig6_panels.pkl when `--redraw-fig6`. Panel j's NMF reference is ranked by component-22 specificity rather than raw
# loading (nmf_specificity). Panel 6b shows the FDR-robust motifs ordered by GC-adjusted odds ratio
# (draw_top_motifs_by_effect; both orderings in motifs_gc/*_motif_ranking.csv). Motif panels use the GC-adjusted test
# (gc_matched_motif_enrichment.py) on the GMM-significant part of 5__'s leading edges. Labels "Factor Ck" become "PLS Ck".
#
# Env: $PY_ANALYSIS

import argparse
import ast
import functools
import os
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msi_components_common as P  # noqa: E402
import shared_msi_components as SC  # noqa: E402
import dopamine_overview_cells as FD  # noqa: E402

SCENICPLUS_PY = os.environ['PY_MOTIF']
FACTOR = "C1"                       # the dopamine component; main() sets it from the results


def relabels():
    return (("MOFA factor RNA loading", f"PLS {FACTOR} RNA weight"),
            ("Dopamine-factor RNA loading", f"PLS {FACTOR} RNA weight"), ("loading", "weight"),
            (f"Factor {FACTOR}", f"PLS {FACTOR}"),
            ("factor\nscore", "component\nscore"), ("factor score", "component score"),
            ("NMF dopamine program", "specificity-ranked NMF program"))


NMF_TAU = 0.1


def parse_args():
    p = argparse.ArgumentParser(description="Manuscript Figs. 5-6 from shared PLS components.")
    p.add_argument("--run", default="fixed s3", choices=list(P.RUNS))
    p.add_argument("--anchor", default="msi", choices=["msi", "dopamine"],
                   help="components of shared_msi_components.py --anchor <this>")
    p.add_argument("--source-spots", default="intact", choices=["intact", "all"],
                   help="components of shared_msi_components.py --source-spots <this>")
    p.add_argument("--msi-weights", choices=("ot", "ported"), default="ot",
                   help="metabolite weights of the target views: OT-recovered (fig567 'target MSI (OT)') or ported")
    p.add_argument("--top-loadings", type=int, default=10,
                   help="genes / metabolites per Fig. 5i dot panel (colour key only)")
    p.add_argument("--nmf-top-k", type=int, default=50,
                   help="NMF program genes defining the positive set of the Fig. 5j convergence panel")
    p.add_argument("--size-key", action="store_true",
                   help="Fig. 5i dot panels with the |weight| size key (the bundle's top-6 layout)")
    p.add_argument("--motif-maps", type=Path, default=SC.OUT_DEFAULT / "spatial_motif_maps",
                   help="fig567_supplement/plot_spatial_motif_maps.py output (Fig. 6d)")
    p.add_argument("--redraw-fig6", action="store_true",
                   help="only redraw Fig. 6 from <out>/fig6_panels.pkl, cached by a full run")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: manuscript_layout_<components dir> (manuscript_layout_full for the "
                        "dopamine anchor on all spots)")
    args = p.parse_args()
    if args.out_dir is None:
        sub = SC.components_subdir(args.anchor, args.source_spots)
        args.out_dir = SC.OUT_DEFAULT / ((f"manuscript_layout_{sub}" if sub else "manuscript_layout_full")
                                         + ("_ot" if args.msi_weights == "ot" else ""))
    return args


class PLSModel:
    """The slice of mofax.mofa_model that 5__'s figure and Fig. 6 cells use."""

    def __init__(self, scores, weights, samples_metadata=None):
        self.scores = scores                      # samples x components
        self.weights = weights                    # view -> features x components
        self.views = list(weights)
        self.samples_metadata = samples_metadata

    def _select(self, frame, factors):
        if factors is None:
            return frame
        factors = [factors] if isinstance(factors, (str, int, np.integer)) else list(factors)
        names = [frame.columns[f] if isinstance(f, (int, np.integer)) else f for f in factors]
        return frame[names]

    def get_factors(self, factors=None, df=False, **_):
        Z = self._select(self.scores, factors)
        return Z if df else Z.to_numpy()

    def get_weights(self, views=None, factors=None, df=False, absolute_values=False, **_):
        views = self.views if views is None else ([views] if isinstance(views, str) else list(views))
        W = self._select(pd.concat([self.weights[v] for v in views]), factors)
        W = W.abs() if absolute_values else W
        return W if df else W.to_numpy()

    def get_top_features(self, factors=None, n_features=10, views=None, **_):
        W = self.get_weights(views=views, factors=factors, df=True).abs().max(axis=1)
        return W.sort_values(ascending=False).head(n_features).index.to_numpy()

    def get_features(self):
        return pd.concat([pd.DataFrame({"feature": w.index, "view": v}) for v, w in self.weights.items()],
                         ignore_index=True)

    def get_cells(self):
        return pd.DataFrame({"cell": self.scores.index})



def unique(frame):
    return frame[~frame.index.duplicated()]


MSI_BLOCK = {"ot": "target MSI (OT)", "ported": "target MSI (ported)"}
MSI_WEIGHTS = "ot"                  # main() sets it from --msi-weights


def pls_model(real, results, latent, msi_view, teacher=False):
    """PLSModel standing in for `real` (the run's MOFA model of the same target)."""
    cells = real.get_factors(df=True).index.astype(str)
    Z0 = SC.standardize(latent.reindex(cells).to_numpy())
    scores = pd.DataFrame(Z0 @ results["U"], index=cells, columns=results["comps"])
    W = results["teacher"] if teacher else results["W"]
    msi = W.get(MSI_BLOCK[MSI_WEIGHTS], results["W"]["source MSI"])     # dopamine-anchored runs: measured
    weights = {"rna": unique(W["target RNA"]), "atac": unique(W["target ATAC"]), msi_view: unique(msi)}
    return PLSModel(scores, weights, real.samples_metadata)


def nmf_specificity(ns, tau=NMF_TAU):
    """Softmax over components of each gene's loading (components scaled to max 1), taken
    at the dopamine component: high for genes that load strongly *and* mostly on it, so
    housekeeping / mitochondrial genes shared by every component drop out."""
    from scipy.special import softmax

    best = ns["best_nmf_component"]
    H = ns["nmf"].components_
    row = int(np.flatnonzero((H == best.to_numpy()).all(axis=1))[0])
    spec = softmax(H / H.max(axis=1, keepdims=True) / tau, axis=0)[row]
    print(f"[pls] NMF reference: component {row} by softmax specificity (tau={tau}); top-10: "
          f"{', '.join(pd.Series(spec, index=best.index).nlargest(10).index)}", flush=True)
    return pd.Series(spec, index=best.index)


def ccre_partition(ns):
    """Target peaks split by D2-MSN / MXD cCRE overlap into D2-only, shared and MXD-only.
    ~90% of MXD cCRE peaks are also D2-MSN cCRE peaks, so the MXD-specific question
    (Supplementary MXD figure) can only be asked of the MXD-only part."""
    peaks = ns["multiome_mofa"].get_weights(views=["atac"], df=True).index
    overlap = ns["atac_peaks_overlapping_regions"]
    d2, mxd = (overlap(peaks.to_numpy(), ns[f"{ct}_ccre_regions"]) for ct in ("d2msn", "mxd"))
    parts = {"D2-MSN-only": peaks[d2 & ~mxd], "shared": peaks[d2 & mxd], "MXD-only": peaks[mxd & ~d2]}
    print("[pls] cCRE partition: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()), flush=True)
    return parts


FIG6_MAPS = (("ZEB1", "top motif"), ("PBX3", "top motif"), ("MEIS1", "top motif"), ("EGR1", "RNA-ATAC concordant"))
FIG6_N_MOTIFS, FIG6_N_REGULONS = 8, 18
# Fig. 6 is drawn at the manuscript's \linewidth (16 cm) with every axes placed in inches, so
# these are the printed font sizes and nothing is rescaled after drawing.
FIG6_WIDTH = 6.3
FIG6_RC = {"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8,
           "ytick.labelsize": 8, "legend.fontsize": 7.5, "legend.title_fontsize": 8,
           "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
           "xtick.major.size": 2.5, "ytick.major.size": 2.5}


def inch_axes(fig, x, y, w, h):
    """Axes w x h inches with its lower-left corner (x, y) inches from the figure's."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def thin(n, keep=(), points=4000):
    """About `points` evenly spaced of n indices plus `keep` and the last: the 66,520-rank GSEA
    curves stay vector in the SVG with no visible change."""
    return np.unique(np.r_[np.arange(0, n, max(1, n // points)), np.asarray(keep, dtype=int), n - 1])


def draw_gsea(fig, box, p, title):
    """Fig. 6a in `box` (x, y, w, h inches): running enrichment score, the fraction of peaks
    that are cCREs along the ranking (Gaussian-kernel hit rate; at ~38,000 hits the usual
    one-tick-per-hit strip is a solid slab), the rank-metric strip and the ranked weights."""
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator
    from scipy.ndimage import gaussian_filter1d

    x0, y0, w, h = box
    axes, top = [], y0 + h
    # ES, gap, cCRE fraction, gap, colour strip, ranked weight (shares of h); the gaps keep the
    # stacked axes' end tick labels apart
    for i, share in enumerate((0.5, 0.03, 0.17, 0.025, 0.035, 0.24)):
        top -= share * h
        if i not in (1, 3):
            axes.append(inch_axes(fig, x0, top, w, share * h))
    ax_es, ax_kde, ax_bar, ax_rank = axes
    rank, res, n = p["ranking"], p["res"], len(p["ranking"])
    xs = np.arange(n)
    zero = int(np.abs(rank).argmin())               # gseapy's "zero score" rank
    idx = thin(n, keep=(np.argmax(np.abs(res)), zero))
    for ax in axes:
        ax.set_xlim(0, n - 1)
    for ax in axes[:-1]:
        ax.set_xticks([])

    ax_es.plot(xs[idx], res[idx], color="#88C544", lw=1.6)
    ax_es.axhline(0, color="0.5", lw=0.6)
    ax_es.set_ylim(min(res.min(), 0) - 0.06 * np.ptp(res), max(res.max(), 0) + 0.06 * np.ptp(res))
    ax_es.yaxis.set_major_locator(MaxNLocator(4))
    ax_es.set_ylabel("Enrichment\nscore")
    ax_es.set_title(title)
    floor = 1 / p["n_perm"]
    ax_es.text(0.42, 0.08, f"NES = {p['nes']:.2f}\nFDR " + (f"< {floor:g}" if p["fdr"] < floor else f"= {p['fdr']:.2g}"),
               transform=ax_es.transAxes, ha="center", va="bottom")      # under the curve

    hit = np.zeros(n)
    hit[p["hits"]] = 1.0
    # mirrored ends: "nearest" would pad with the last peak's 0/1 for 4 sigma and bend the ends
    rate, base = gaussian_filter1d(hit, sigma=n / 100, mode="reflect")[idx], hit.mean()
    ax_kde.fill_between(xs[idx], rate, base, where=rate >= base, interpolate=True, color="0.2", lw=0)
    ax_kde.fill_between(xs[idx], rate, base, where=rate < base, interpolate=True, color="0.72", lw=0)
    ax_kde.axhline(base, color="k", lw=0.6, ls=(0, (2, 1.5)))
    ax_kde.text(0.985 * n, base + 0.04, "all peaks", ha="right", va="bottom", fontsize=7.5)
    ax_kde.set_ylim(0, 1)
    ax_kde.set_yticks([0, 1])
    ax_kde.set_ylabel("cCRE\nfraction")

    lo, hi = np.percentile(rank, [1, 99])
    ax_bar.imshow(np.clip(rank[idx], lo, hi)[None, :], aspect="auto", cmap="seismic", norm=TwoSlopeNorm(0, lo, hi),
                  extent=(0, n - 1, 0, 1), interpolation="nearest")
    ax_bar.set_yticks([])

    ax_rank.fill_between(xs[idx], rank[idx], 0, color="#C3CDD9", lw=0)
    ax_rank.axvline(zero, color="0.35", lw=0.7, ls=":")
    ax_rank.text(zero + 0.03 * n, 0.55 * rank.max(), f"zero at\nrank {zero:,}", ha="left", va="center", fontsize=7.5)
    ax_rank.set_ylim(rank.min() - 0.08 * np.ptp(rank), rank.max() + 0.08 * np.ptp(rank))
    ax_rank.yaxis.set_major_locator(MaxNLocator(3))
    ax_rank.set_ylabel("Ranked\nweight")
    ax_rank.set_xticks([0, 20000, 40000, 60000], ["0", "20k", "40k", "60k"])
    ax_rank.set_xlabel("Peak rank")
    fig.align_ylabels([ax_es, ax_kde, ax_rank])


def draw_top_motifs_by_effect(ax, enr, n_top=FIG6_N_MOTIFS):
    """Fig. 6b: the motifs robust at FDR < 0.05 (robust_motif_tier), ordered and drawn by their
    GC-adjusted odds ratio (foreground vs the cell type's other cCRE peaks). q-value order mixes
    enrichment with motif frequency: a common motif reaches a small q at a modest odds ratio."""
    robust = enr[enr["padj"] < 0.05].sort_values("odds_ratio", ascending=False)
    top = robust.drop_duplicates("motif_name").head(n_top)
    pos = np.arange(len(top))
    ax.bar(pos, top["odds_ratio"], width=0.72, color="#B2182B")
    ax.axhline(1.0, color="k", ls="--", lw=0.7, zorder=0)
    ax.set_ylim(max(0.0, min(1.0, top["odds_ratio"].min()) - 0.05), top["odds_ratio"].max() * 1.04)
    ax.set_xlim(-0.6, len(top) - 0.4)
    ax.set_xticks(pos, top["motif_name"], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_ylabel("GC-adjusted odds ratio\n(FDR q < 0.05)")
    ax.set_title("D2-MSN: top TF motifs")


def draw_regulon_concordance(ax, regulons, tiers, n_top=FIG6_N_REGULONS):
    """Fig. 6c: 5__'s RNA-regulon x ATAC-motif concordance dotplot (_fig_draw_regulon_concordance)
    redrawn at print size from fig6_panels' table; the colour key lists only the tiers shown."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colours = dict(tiers)
    top = regulons.head(n_top).iloc[::-1]
    tier = top["tier"].where(top["tier"].isin(colours), "not concordant")
    n_genes = top["n_genes"].to_numpy(dtype=float)
    span, (smin, smax) = np.ptp(n_genes), (10.0, 90.0)

    def size(v):
        v = np.asarray(v, dtype=float)
        return smin + (v - n_genes.min()) / span * (smax - smin) if span > 0 else np.full(v.shape, (smin + smax) / 2)

    x = -np.log10(top["padj"].clip(lower=1e-300)).to_numpy()
    y = np.arange(len(top))
    ax.scatter(x, y, s=size(n_genes), c=tier.map(colours).tolist(), edgecolor="k", linewidth=0.4, zorder=3)
    ax.axvline(-np.log10(0.05), color="k", ls="--", lw=0.7, zorder=0)
    ax.set_yticks(y, top["Term"])
    ax.set_ylim(-0.7, len(top) - 0.3)
    ax.set_xlim(0, x.max() * 1.12)
    ax.set_xlabel(r"$-\log_{10}$(RNA regulon FDR)")
    ax.set_title("motif concordance (D2-MSN)")
    # both keys stacked in the empty lower-right corner (rows run from most to least significant),
    # the size key left-aligned on top of the colour key
    frame = dict(frameon=False, borderaxespad=0.15, borderpad=0.2)
    colour_key = ax.legend(handles=[Patch(facecolor=c, edgecolor="k", lw=0.4,
                                          label=t.replace("RNA-", "").replace("ATAC-", "").replace(" x ", " × "))
                                    for t, c in tiers if (tier == t).any()],
                           loc="lower right", title="RNA × ATAC", handlelength=1.0, handleheight=0.9, **frame)
    colour_key._legend_box.align = "left"
    ax.add_artist(colour_key)
    ax.figure.canvas.draw()
    corner = colour_key.get_window_extent().transformed(ax.transAxes.inverted())
    refs = np.unique(np.quantile(n_genes, [0.0, 0.5, 1.0]).round().astype(int))
    size_key = ax.legend(handles=[Line2D([0], [0], marker="o", ls="", markerfacecolor="#888888", markeredgecolor="k",
                                         markeredgewidth=0.4, markersize=np.sqrt(size(v)), label=f"{v} genes")
                                  for v in refs],
                         loc="lower left", bbox_to_anchor=(corner.x0, corner.y1), title="overlap",
                         handlelength=1.0, labelspacing=0.35, **frame)
    size_key._legend_box.align = "left"


def spot_outline(ax, x, y, inside, pitch=24.0, **kw):
    """Contour of a spot set (Visium pitch 24 units); triangles longer than two pitches, i.e.
    across the section's outer edge, are masked."""
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(x, y)
    p = np.stack([x[tri.triangles], y[tri.triangles]], -1)
    tri.set_mask(np.linalg.norm(p - np.roll(p, 1, axis=1), axis=-1).max(1) > 2 * pitch)
    ax.tricontour(tri, inside.astype(float), levels=[0.5], **kw)


def map_limits(D, pad=0.02):
    return [(v.min() - pad * np.ptp(v), v.max() + pad * np.ptp(v)) for v in (D["x"].to_numpy(), D["y"].to_numpy())]


def draw_motif_maps(fig, D, st, x0, y0, map_w, gap):
    """Fig. 6d: motif accessibility of the multiome nuclei placed on the source section by the
    student latent (fig567_supplement/plot_spatial_motif_maps.py), striatum outlined; one
    map_w-inch map per FIG6_MAPS entry from (x0, y0) inches, then the colour bar."""
    import matplotlib.pyplot as plt

    x, y = D["x"].to_numpy(), D["y"].to_numpy()
    (xl, yl) = map_limits(D)
    map_h = map_w * np.ptp(yl) / np.ptp(xl)
    disc = 2.0 * 24 * map_w / np.ptp(xl) * 72           # two spot pitches, in points
    for i, (motif, origin) in enumerate(FIG6_MAPS):
        ax = inch_axes(fig, x0 + i * (map_w + gap), y0, map_w, map_h)
        v = D[motif].to_numpy()
        lo, hi = np.nanpercentile(v, [2, 98])
        sc = ax.scatter(x, y, c=np.clip(v, lo, hi), s=disc ** 2, cmap="viridis", lw=0,
                        rasterized=True)          # discs over the runs of missing spots, as in Fig. 5a
        spot_outline(ax, x, y, D["striatum"].to_numpy(), colors="white", linewidths=0.8)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"$\\bf{{{motif}}}$\n{origin}\nstriatum AUROC {st.loc[motif, 'auroc_striatum']:.2f}",
                     fontsize=8, linespacing=1.15, pad=3)
    cax = inch_axes(fig, x0 + len(FIG6_MAPS) * (map_w + gap) + 0.04, y0 + 0.08 * map_h, 0.07, 0.84 * map_h)
    cb = plt.colorbar(sc, cax=cax, ticks=[sc.norm.vmin, sc.norm.vmax])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("motif accessibility\n(chromVAR, mapped)")
    cb.outline.set_visible(False)


def fig6_panels(ns, term="D2MSN_cCRE", n_regulons=20):
    """Everything Fig. 6a-c draw, as plain arrays and tables. main() caches it as
    fig6_panels.pkl, so `--redraw-fig6` redraws the board without rerunning the pipeline."""
    pre, reg, enr = ns["d2msn_multiome_gsea"], ns["regulon_res"], ns["d2msn_motif_enrichment"]
    r = pre.results[term]
    rna_fdr, rna_nom = (set(reg.loc[reg[col] < 0.05, "Term"]) for col in ("padj", "pval"))
    atac_fdr, atac_nom = (ns["motif_hits_to_tf_symbols"](enr, use=use, alpha=0.05) for use in ("padj", "pvalue"))
    top = reg.sort_values("padj").head(n_regulons).copy()
    top["tier"] = [ns["assign_concordance_tier"](t, rna_fdr, rna_nom, atac_fdr, atac_nom) for t in top["Term"]]
    top["n_genes"] = (top["Overlap"].astype(str).str.split("/").str[0].astype(float)
                      if "Overlap" in top.columns else 1.0)
    return dict(factor=FACTOR, ranking=np.asarray(pre.ranking, dtype=float), hits=np.asarray(r["hits"], dtype=int),
                res=np.asarray(r["RES"], dtype=float), nes=float(r["nes"]), pval=float(r["pval"]),
                fdr=float(r["fdr"]), n_perm=int(getattr(pre, "permutation_num", 1000)), motifs=enr,
                regulons=top, tiers=list(ns["CONCORDANCE_TIERS"]) + [("not concordant", ns["NOT_CONCORDANT_COLOR"])],
                rcparams=dict(ns["FIG_RCPARAMS"]))


def fig6_d2msn_board(p, output_file, maps_dir):
    """Main Fig. 6, D2-MSN only, at print width: (a) cCRE GSEA on all peaks, (b) top TF motifs
    of the GMM-significant leading edge and (c) RNA-regulon x ATAC-motif concordance in one row,
    over (d) motif accessibility of the multiome nuclei mapped onto the source section.
    `p` is fig6_panels' output."""
    import matplotlib.pyplot as plt

    D = pd.read_csv(maps_dir / "spatial_motif_maps.csv.gz", index_col=0)
    st = pd.read_csv(maps_dir / "spatial_motif_map_stats.csv").query("kind != 'motif'").set_index("map")
    W, margin, gap = FIG6_WIDTH, 0.05, 0.1
    map_w = (W - 2 * margin - 0.95 - gap * (len(FIG6_MAPS) - 1)) / len(FIG6_MAPS)   # 0.95: colour bar + label
    (xl, yl) = map_limits(D)
    d_top = 0.03 + map_w * np.ptp(yl) / np.ptp(xl) + 0.48       # maps + three-line titles
    row_y, row_h = d_top + 0.85, 2.3                             # 0.85: b's slanted motif names
    H = row_y + row_h + 0.32
    with plt.rc_context({**p["rcparams"], **FIG6_RC}):
        fig = plt.figure(figsize=(W, H))
        draw_gsea(fig, (0.62, row_y, 1.36, row_h), p, "D2-MSN cCREs")
        draw_top_motifs_by_effect(inch_axes(fig, 2.66, row_y, 1.12, row_h), p["motifs"])
        draw_regulon_concordance(inch_axes(fig, 4.42, row_y, W - 4.42 - margin, row_h), p["regulons"], p["tiers"])
        draw_motif_maps(fig, D, st, margin, 0.03, map_w, gap)
        for x, y, letter in ((0.0, H, "a"), (2.03, H, "b"), (3.78, H, "c"), (0.0, d_top + 0.12, "d")):
            fig.text(x / W, y / H, letter, fontsize=12, fontweight="bold", va="top", ha="left")
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
    print(f"[pls] wrote {output_file}", flush=True)


MXD_MODELS = (("teacher, P22", "spatial", "teacher", "#1b9e77", "o", True),
              ("student, P22", "spatial", "W", "#d95f02", "o", False),
              ("student, multiome", "multiome", "W", "#d95f02", "o", True))


def fig_mxd_supplement(ns, results, parts, output_file):
    """Supplementary MXD figure: GC-stratified AUROC of FACTOR peak weights for D2-MSN-only,
    shared and MXD-only cCRE peaks against non-cCRE peaks, for the teacher on P22, the
    student on P22 and the student on multiome. Asks whether the MXD node of the teacher's
    P22 map (Fig. 5g) has MXD-specific chromatin behind it, and whether that is a teacher
    or a target-dataset property."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import mannwhitneyu

    weights = {name: unique(results[t]["teacher" if key == "teacher" else "W"]["target ATAC"])[FACTOR]
               for name, t, key, *_ in MXD_MODELS}
    peaks = next(iter(weights.values())).index
    gc = peak_gc(peaks, Path(output_file).parent / "peak_gc.csv")
    any_ccre = np.concatenate([v.to_numpy() for v in parts.values()])
    non = ~peaks.isin(any_ccre)
    rows = []
    for name, w in weights.items():
        w = w.reindex(peaks).to_numpy()
        for part, idx in parts.items():
            m = peaks.isin(idx)
            rows.append(dict(model=name, part=part, n=int(m.sum()), gc_member=gc[m].mean(), gc_background=gc[non].mean(),
                             auroc_raw=mannwhitneyu(w[m], w[non]).statistic / (m.sum() * non.sum()),
                             auroc_gc_stratified=gc_stratified_auroc(w, m, non, gc)))
    table = pd.DataFrame(rows)
    table.to_csv(Path(output_file).with_suffix(".csv"), index=False)
    print(table.round(3).to_string(index=False), flush=True)

    with plt.rc_context(ns["FIG_RCPARAMS"]):
        fig, ax = plt.subplots(figsize=(9, 4))
        order = list(parts)
        for j, (name, _t, _k, color, marker, filled) in enumerate(MXD_MODELS):
            sub = table[table.model == name].set_index("part").reindex(order)
            y = np.arange(len(order)) + (j - 1) * 0.22
            ax.scatter(sub["auroc_gc_stratified"], y, s=80, marker=marker, linewidths=2, zorder=3,
                       facecolors=color if filled else "white", edgecolors=color)
        ax.axvline(0.5, color="0.5", linestyle="--", linewidth=1, zorder=1)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([f"{k} cCREs (n={len(parts[k]):,})" for k in order])
        ax.invert_yaxis()
        ax.set_xlabel(f"GC-stratified AUROC of {FACTOR} peak weight vs non-cCRE peaks")
        ax.grid(axis="x", color="0.9", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=[Line2D([0], [0], marker=m, linestyle="none", markersize=9, markeredgewidth=2,
                                  markeredgecolor=c, markerfacecolor=c if f else "white", label=n)
                           for n, _t, _k, c, m, f in MXD_MODELS],
                  loc="lower right", frameon=False)
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[pls] wrote {output_file}", flush=True)


def latent_frame(tag, target, key):
    import anndata as ad

    emb = ad.read_h5ad(P.PROJ / (P.BASE + tag) / P.TARGETS[target]["emb"], backed="r")
    return pd.DataFrame(np.asarray(emb.obsm[key]), index=emb.obs_names.astype(str))


def run_handoff_defs(ns):
    """5__'s motif hand-off cell without the scenicplus runs: its defs, the leading edges
    and the cell-type cCRE backgrounds (the GC-adjusted test replaces the scan)."""
    for header, code in FD.split_cells(FD.SCRIPT):
        if header.startswith("TF motif enrichment hand-off"):
            tree = ast.parse(code)
            skipped_calls = {"run_tf_motif_enrichment", "plot_top_motifs"}
            tree.body = [n for n in tree.body if not any(
                isinstance(c, ast.Call) and getattr(c.func, "id", "") in skipped_calls
                for c in ast.walk(n))]
            print("[pls] running cell: TF motif enrichment hand-off (defs + leading edges)", flush=True)
            exec(compile(ast.Module(body=tree.body, type_ignores=[]), str(FD.SCRIPT), "exec"), ns)
            return
    raise RuntimeError("motif hand-off cell not found")


def gmm_slab(weights, seed=0):
    """Peaks whose |weight| falls in the higher-mean component of a 2-component Gaussian
    mixture (slab posterior >= 0.5): 5__'s fit_weight_gmm rule, here on peaks."""
    from sklearn.mixture import GaussianMixture

    X = np.abs(weights.to_numpy(dtype=float)).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=seed, n_init=5).fit(X)
    return weights.index[gm.predict_proba(X)[:, int(np.argmax(gm.means_.ravel()))] >= 0.5]


def gc_motifs(ns, out):
    """GC-adjusted motif test. Foreground: the GMM-significant part of 5__'s GSEA leading
    edge (the slab of |FACTOR weight| fitted within the leading edge); background: the cell
    type's other cCRE peaks. On all 66,520 peaks the leading edge is ~half of each cCRE
    set, so testing it whole contrasts the upper vs lower half of the cCREs."""
    inputs, motifs_dir, label = out / "motif_inputs_gmm", out / "motifs_gc", f"pls_{FACTOR}_gmm"
    inputs.mkdir(parents=True, exist_ok=True)
    w = ns["multiome_mofa"].get_weights(views=["atac"], factors=FACTOR, df=True).iloc[:, 0]
    for ct in ("d2msn", "mxd"):
        lead = w.index.intersection(pd.Index(ns[f"{ct}_leading_edge_peaks"]))
        fg = gmm_slab(w.loc[lead])
        bg = pd.Index(ns[f"{ct}_ccre_background_peaks"]).difference(fg)
        print(f"[pls] {ct}: motif foreground {len(fg)} GMM-significant of {len(lead)} leading-edge "
              f"peaks (|w| >= {w.loc[fg].abs().min():.3f}); background {len(bg)} cCRE peaks", flush=True)
        P.peaks_bed(fg).to_csv(inputs / f"{ct}_foreground.bed", sep="\t", header=False, index=False)
        P.peaks_bed(bg).to_csv(inputs / f"{ct}_background.bed", sep="\t", header=False, index=False)
    if not (motifs_dir / f"{label}_mxd_gc_controlled_motifs.tsv").exists():
        print("[pls] GC-adjusted motif test (scenicplus env)", flush=True)
        subprocess.run([SCENICPLUS_PY, str(HERE / "gc_matched_motif_enrichment.py"), "--processes", "16",
                        "--out-dir", str(motifs_dir), "--inputs", f"{label}={inputs}"],
                       check=True, stdout=subprocess.DEVNULL)
    FD.use_gc_adjusted_motifs(ns, motifs_dir / label)
    robust_motif_tier(ns, motifs_dir / label)


def robust_motif_tier(ns, spec):
    """Panel c-d q-value = max(GC-adjusted logit FDR, median FDR of the three GC-matched
    background draws), so q < 0.05 exactly when a motif is robust (logit FDR < 0.05 and
    FDR < 0.05 in >= 2 of 3 matched draws). The nominal tier keeps the logit p-value."""
    for ct in ("d2msn", "mxd"):
        table = pd.read_csv(spec.parent / f"{spec.name}_{ct}_gc_controlled_motifs.tsv", sep="\t")
        enr = pd.DataFrame({
            "motif_name": table["motif_name"].str.replace(r" \(MA[0-9.]+\)$", "", regex=True),
            "pvalue": table["logit_pvalue"],
            "padj": np.maximum(table["logit_padj"], table["matched_median_padj"]),
            "odds_ratio": np.exp(table["logit_log_odds_fg_adj_gc"]),
        }).sort_values(["padj", "pvalue"]).reset_index(drop=True)
        ns[f"{ct}_motif_enrichment"] = enr
        robust = enr[enr.padj < 0.05].sort_values("odds_ratio", ascending=False)
        order = enr.assign(rank_by_q=np.arange(1, len(enr) + 1))
        order["rank_by_odds_ratio_among_robust"] = order["motif_name"].map(
            pd.Series(np.arange(1, len(robust) + 1), index=robust["motif_name"]).groupby(level=0).min())
        order.to_csv(spec.parent / f"{spec.name}_{ct}_motif_ranking.csv", index=False)
        print(f"[pls] {ct}: robust motif tier, {len(robust)} robust at FDR<0.05; by GC-adjusted odds ratio: "
              f"{', '.join(robust.motif_name[:6])} (by q: {', '.join(enr.motif_name[:6])})", flush=True)


def peak_gc(peaks, cache):
    """GC fraction of each 'chr:start-end' peak in mm10 (bedtools nuc), cached as csv."""
    if cache.exists():
        gc = pd.read_csv(cache, index_col=0).iloc[:, 0]
        if gc.index.equals(pd.Index(peaks)):
            return gc
    import pybedtools

    os.environ["PATH"] = f"{Path(sys.executable).parent}:{os.environ['PATH']}"
    bed = P.peaks_bed(pd.Index(peaks))[["chrom", "start", "end", "name"]]
    sizes = pd.read_csv(f"{ns_fasta()}.fai", sep="\t", header=None, index_col=0)[1]
    bed = bed[bed["end"] <= bed["chrom"].map(sizes)]        # scaffolds absent from mm10.fa -> GC NaN
    nuc = pybedtools.BedTool.from_dataframe(bed).nucleotide_content(fi=str(ns_fasta())).to_dataframe(
        disable_auto_names=True, header=0)
    gc = nuc.set_index(nuc.columns[3])[nuc.columns[5]].astype(float).reindex(peaks).rename("gc")
    gc.to_csv(cache)
    return gc


def ns_fasta():
    return Path.home() / ".local/share/genomes/mm10/mm10.fa"


def gc_stratified_auroc(w, member, background, gc, width=0.02):
    """Mann-Whitney AUROC of w (member vs background) pooled over GC bins: only pairs in
    the same `width`-wide GC bin are compared."""
    from scipy.stats import mannwhitneyu

    bins = np.floor(gc.to_numpy() / width)
    num = den = 0.0
    for b in np.unique(bins[np.isfinite(bins)]):
        m, n = member & (bins == b), background & (bins == b)
        if m.any() and n.any():
            num += mannwhitneyu(w[m], w[n]).statistic
            den += m.sum() * n.sum()
    return num / den


def manuscript_row_order(ns):
    """Redefine FIG_dopamine_overview with the manuscript's row order: the embedding
    benchmark as row 2 (panels d-f) and the transferred program as row 3 (g-j). 5__ draws
    them the other way round and letters panels in drawing order, so the two row blocks
    are swapped in its source rather than relettered afterwards."""
    text = FD.SCRIPT.read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "FIG_dopamine_overview")
    src = ast.get_source_segment(text, node)
    swaps = (("height_ratios = [5.9, 4.4] + ([4.2] if bench is not None else [])",
              "height_ratios = [5.9] + ([4.2] if bench is not None else []) + [4.4]"),
             ("sf_transfer = rows[1]", "sf_transfer = rows[-1]"),
             ("sf_bench = rows[2]", "sf_bench = rows[1]"))
    for old, new in swaps:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    marks = [src.index(m) for m in ("        # ── row 2:", "        # ── row 3:", "        # Last, because")]
    src = src[:marks[0]] + src[marks[1]:marks[2]] + src[marks[0]:marks[1]] + src[marks[2]:]
    exec(compile(src, str(FD.SCRIPT), "exec"), ns)


def relabelled(finish):
    def wrapper(fig, output_file):
        for text in fig.findobj(lambda a: hasattr(a, "get_text") and hasattr(a, "set_text")):
            s = text.get_text()
            for old, new in relabels():
                s = s.replace(old, new)
            if s != text.get_text():
                text.set_text(s)
        return finish(fig, output_file)
    return wrapper


def main():
    global FACTOR, MSI_WEIGHTS
    args = parse_args()
    MSI_WEIGHTS = args.msi_weights
    tag = P.RUNS[args.run]
    sub = SC.components_subdir(args.anchor, args.source_spots)
    comp_root = SC.OUT_DEFAULT / sub if sub else SC.OUT_DEFAULT
    if args.run != "fixed s3":
        comp_root = comp_root / "runs" / args.run.replace(" ", "_")
    out = args.out_dir if args.run == "fixed s3" else args.out_dir / args.run.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    if args.redraw_fig6:
        with open(out / "fig6_panels.pkl", "rb") as handle:
            panels = pickle.load(handle)
        for suffix in ("png", "svg"):
            fig6_d2msn_board(panels, out / f"SMA_mouse_pls_enrichment.{suffix}", args.motif_maps)
        return 0
    results = {}
    for target in ("spatial", "multiome"):
        with open(comp_root / target / "results.pkl", "rb") as handle:
            results[target] = pickle.load(handle)
    FACTOR = results["multiome"]["c_da"]
    assert results["spatial"]["c_da"] == FACTOR, (results["spatial"]["c_da"], FACTOR)
    print(f"[pls] components from {comp_root} (anchor {args.anchor}); dopamine component {FACTOR}", flush=True)

    os.environ.update(SPATIALJEPA_RUN_TAG=tag, SPATIALJEPA_NMF_CACHE_DIR=str(P.PROJ / P.BASE),
                      SPATIALJEPA_FIG_DIR=str(out))
    ns = {"__name__": "fig56_pls_manuscript", "__file__": str(FD.SCRIPT)}
    FD.run_cells(FD.FIG5_CELLS[:8], ns)                     # data, NMF, targets, MOFA models

    ns["spatial_mofa"] = pls_model(ns["spatial_mofa"], results["spatial"],
                                   latent_frame(tag, "spatial", "teacher_X_emb_st"), "msi_teacher", teacher=True)
    ns["multiome_mofa"] = pls_model(ns["multiome_mofa"], results["multiome"],
                                    latent_frame(tag, "multiome", "student_X_emb_st"), "msi_student")
    ns["spatial_dopamine_factor_n"] = ns["multiome_dopamine_factor_n"] = FACTOR
    ns["best_nmf_component"] = nmf_specificity(ns)
    for name in ("spatial_mofa", "multiome_mofa"):
        view = "msi_teacher" if name == "spatial_mofa" else "msi_student"
        w = ns[name].get_weights(views=[view], df=True).loc["msi:Dopamine", FACTOR]
        print(f"[pls] {name}: {FACTOR} dopamine weight {w:+.3f}", flush=True)

    FD.run_cells(FD.FIG6_CELLS[:3], ns)                      # cCRE overlap, top features, GSEA
    run_handoff_defs(ns)
    gc_motifs(ns, out)
    FD.run_cells(FD.FIG6_CELLS[4:], ns)                      # RNA-side regulon enrichment
    FD.run_cells(FD.FIG5_CELLS[-1:], ns)                     # figure functions
    manuscript_row_order(ns)
    ns["_fig_finish"] = relabelled(ns["_fig_finish"])
    ns["_fig_loadings_dotplot"] = functools.partial(ns["_fig_loadings_dotplot"], size_key=args.size_key)

    for suffix in ("png", "svg"):
        ns["FIG_dopamine_overview"](output_file=str(out / f"SMA_mouse_dopamine_nmf_pls.{suffix}"),
                                    n_top_loadings=args.top_loadings, top_k=args.nmf_top_k)
    for k in (10, 20, 30, 50, 100):                          # Fig. 5j sensitivity to top-k (logged)
        ns["_fig_nmf_mofa_enrichment_stats"](ns["best_nmf_component"], ns["_fig_default_enrichment_models"](), k)
    FD.repair_degenerate_gsea(ns["d2msn_multiome_gsea"], "D2MSN_cCRE")
    panels = fig6_panels(ns)
    with open(out / "fig6_panels.pkl", "wb") as handle:
        pickle.dump(panels, handle)
    parts = ccre_partition(ns)
    for suffix in ("png", "svg"):
        fig6_d2msn_board(panels, out / f"SMA_mouse_pls_enrichment.{suffix}", args.motif_maps)
        fig_mxd_supplement(ns, results, parts, out / f"SMA_mouse_pls_mxd_supplement.{suffix}")
    print(f"[pls] wrote {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
