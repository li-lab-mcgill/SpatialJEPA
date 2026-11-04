try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Fig. 5 and Fig. 6 drawn from the PLS decoders instead of MOFA+.
#
# Same layouts and drawing helpers as the manuscript figures (5__'s FIG_* helpers);
# only the MOFA-derived inputs are replaced (see msi_components_common.py for the
# rationale). A figure is drawn for one latent direction of one SpatialJEPA run:
#
#   dopamine axis   the student MSI decoder's dopamine direction (imputed dopamine)
#   MSI-PLS Ck      the decoder's k-th component, oriented so dopamine loads positively
#
# Fig. 5   rows 1 and 3 unchanged (measured source; embedding benchmark)
#          g, h  axis score in the P22 spatial target / multiome nuclei (student latent)
#          i     axis gene loadings (latent->RNA PLS decoder fitted in the multiome
#                target) and metabolite loadings (dopamine axis: correlation with the
#                axis in the source's MEASURED MSI when the run's source latents exist,
#                else the decoded target MSI; components: the decoder's own loadings,
#                fitted on measured source MSI)
#          j     gene-level convergence with the NMF program; same 2,000 genes as Fig. 5j
# Fig. 6   a     cell-type cCRE maps (adapted from Zhang et al., 2023; unchanged)
#          b     weighted GSEA of D2-MSN / MXD cCRE peaks over Fig. 6's 2,000 peaks,
#                ranked by the axis's ATAC loadings (latent->ATAC PLS decoder);
#                permutation NES, sign-specific null
#          c     TF motifs of the leading-edge cCREs vs the remaining cCREs of the type,
#                GC-adjusted (hit ~ foreground + GC + GC^2; gc_matched_motif_enrichment.py)
#          d     peak loading vs the linked gene's RNA loading (10x peak annotation),
#                promoter and distal links; Spearman, shuffled-gene null
#          e     peaks linked to the NMF top-50 genes vs other linked peaks; AUROC, with
#                the MSN-identity peak ranking as baseline
#
# Env: $PY_ANALYSIS

import argparse
import ast
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import msi_components_common as P  # noqa: E402
from dopamine_overview_cells import FIG6_PANEL_A_BOX, FIG6_PANEL_A_SOURCE  # noqa: E402

SCRIPT5 = HERE / "dopamine_nmf_signature.py"
OUT_DEFAULT = P.OUT / "rna_msi_runs" / "pls_figures"
MOTIFS_DEFAULT = P.OUT / "rna_msi_runs" / "pls_dopamine_axis" / "motifs_gc"
SCENICPLUS_PY = os.environ['PY_MOTIF']
VARIANTS = {
    "fixed_s3_dopamine_axis": ("fixed s3", "dopamine axis"),
    "fixed_s3_C2": ("fixed s3", "MSI-PLS C2"),
    "manuscript_dopamine_axis": ("manuscript", "dopamine axis"),
}
# in file order; from the two Fig. 6 cells only the drawing helpers are taken (their other
# functions have defaults that reference objects of cells not run here)
SCRIPT5_CELLS = ("Load data", "Load joint data", "Run NMF on spatial Visium data",
                 "plot cCRE enrichment of the top C12 features", "TF motif enrichment hand-off",
                 "figure functions")
DEFS_ONLY = {"plot cCRE enrichment of the top C12 features": {"_fig_draw_gsea"},
             "TF motif enrichment hand-off": {"_fig_draw_top_motifs"}}
TOP_K = 50


def parse_args():
    p = argparse.ArgumentParser(description="PLS versions of Figs. 5 and 6.")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--motifs-dir", type=Path, default=MOTIFS_DEFAULT)
    p.add_argument("--redraw", action="store_true", help="redraw from each variant's cached panels.pkl")
    return p.parse_args()


# ------------------------------------------------------------------ script-5 helpers
def script5_namespace(tag):
    """Run only the light script-5 cells (source data, cached NMF, figure helpers)."""
    os.environ["SPATIALJEPA_RUN_TAG"] = tag
    os.environ["SPATIALJEPA_NMF_CACHE_DIR"] = str(P.PROJ / P.BASE)
    lines = SCRIPT5.read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.match(r"^#\s?%%", line)] + [len(lines)]
    ns = {"__name__": "fig56_pls", "__file__": str(SCRIPT5)}
    wanted = list(SCRIPT5_CELLS)
    for a, b in zip(starts, starts[1:]):
        header = re.sub(r"^#\s?%%\s*", "", lines[a]).strip()
        if not wanted or not header.startswith(wanted[0]):
            continue
        key = wanted.pop(0)
        code = "\n" * a + "".join(lines[a:b])
        if key in DEFS_ONLY:
            tree = ast.parse(code)
            tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                         or (isinstance(n, ast.FunctionDef) and n.name in DEFS_ONLY[key])]
            code = ast.unparse(tree)
        print(f"[fig] script 5 ({tag or 'manuscript'}): {header}", flush=True)
        exec(compile(code, str(SCRIPT5), "exec"), ns)
    if wanted:
        raise RuntimeError(f"script-5 cells not found: {wanted}")
    plt.close("all")
    return ns


# ------------------------------------------------------------------ PLS quantities
def axis_beta(R, axis):
    """Latent filter (raw latent units), metabolite loadings or None, score label."""
    if axis == "dopamine axis":
        return R["beta_da"], None
    if R["decoder"] is None:
        raise ValueError(f"{axis}: this run's decoder file does not reproduce its projection")
    comp = {c["name"]: c for c in P.decoder_components(R["decoder"], R["features"])}[axis]
    return comp["beta"], comp["q"]


def gene_axis_loadings(R, beta, T, stats):
    pls = P.pls_crosscov(R["Z"], T["rna"], P.K_FEATURE, y_stats=stats)
    decoded, _ = P.axis_loadings(pls, beta * pls["xs"], T["rna"], stats)
    return pd.Series(decoded, index=T["genes"]).dropna()


def peak_axis_loadings(R, beta, T, stats):
    pls = P.pls_crosscov(R["Z"], T["atac"], P.K_FEATURE, y_stats=stats)
    decoded, _ = P.axis_loadings(pls, beta * pls["xs"], T["atac"], stats)
    return pd.Series(decoded, index=T["peaks"])


def source_metabolite_loadings(tag, beta, features):
    """corr(measured metabolite, axis score) over the source spots (P0 latents)."""
    import anndata as ad
    import scipy.sparse as sp

    path = P.MODELS / (P.BASE + tag) / "joint_adata.h5ad"
    joint = ad.read_h5ad(path)
    if "student_X_emb" not in joint.obsm:
        return None
    feats = [f for f in features if f in joint.var_names]
    X = joint[:, feats].X
    Y = np.log1p(X.toarray() if sp.issparse(X) else np.asarray(X))
    s = np.asarray(joint.obsm["student_X_emb"]) @ beta
    Yc, sc = Y - Y.mean(0), s - s.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Yc * sc[:, None]).sum(0) / np.sqrt((Yc ** 2).sum(0) * (sc ** 2).sum())
    return pd.Series(r, index=feats).dropna()


def convergence_stats(ns, label, rna_loadings, universe, top_k=TOP_K):
    """Panel j statistics, as 5__'s _fig_nmf_mofa_enrichment_stats, for PLS loadings."""
    from scipy.stats import hypergeom, spearmanr
    from sklearn.metrics import roc_auc_score

    nmf = ns["_fig_bare_gene_index"](ns["best_nmf_component"], "NMF")
    rna = ns["_fig_bare_gene_index"](rna_loadings, label)
    rna = rna[rna.index.isin(set(universe))]
    shared = nmf.index.intersection(rna.index)
    nmf_s, rna_s = nmf.reindex(shared), rna.reindex(shared)
    nmf_top = set(nmf_s.sort_values(ascending=False).head(top_k).index)
    labels = np.fromiter((g in nmf_top for g in shared), dtype=int, count=len(shared))
    auroc = roc_auc_score(labels, rna_s.values)
    rho, rho_p = spearmanr(nmf_s.values, rna_s.values)
    overlap = len(nmf_top & set(rna_s.sort_values(ascending=False).head(top_k).index))
    print(f"[fig] {label}: |shared|={len(shared)} AUROC={auroc:.3f} overlap@{top_k}={overlap}", flush=True)
    return {"name": label, "auroc": float(auroc), "spearman": float(rho), "spearman_p": float(rho_p),
            "overlap": overlap, "expected": top_k ** 2 / len(shared),
            "hyper_p": float(hypergeom.sf(overlap - 1, len(shared), top_k, top_k)), "n_shared": len(shared),
            "top": rna_s.reindex(nmf_top).dropna().values,
            "rest": rna_s.drop(index=[g for g in nmf_top if g in rna_s.index]).values}


def gsea_object(scores, member, names, term):
    """gseapy-style result (ranking, hits, running ES) for 5__'s _fig_draw_gsea."""
    res, leading = P.weighted_gsea(scores, member)
    order = np.argsort(-scores)
    s, m = scores[order], member[order]
    hit = np.where(m, np.abs(s), 0.0)
    running = np.cumsum(hit / hit.sum() - (~m) / (~m).sum())
    pre = types.SimpleNamespace(
        ranking=pd.Series(s, index=names[order]),
        results={term: dict(hits=list(np.flatnonzero(m)), RES=running, nes=res["nes"],
                            pval=res["p"], fdr=res["p"])},   # one gene set: BH q = p
    )
    print(f"[fig] GSEA {term}: ES={res['es']:.3f} NES={res['nes']:.2f} p={res['p']:.4f} "
          f"leading edge {res['n_leading']}/{res['n_set']}", flush=True)
    return pre, leading, res


def motif_table(path):
    t = pd.read_csv(path, sep="\t")
    return pd.DataFrame({
        "motif_name": t["motif_name"].str.replace(r" \(MA[0-9.]+\)$", "", regex=True),
        "pvalue": t["logit_pvalue"], "padj": t["logit_padj"],
    }).sort_values(["padj", "pvalue"]).reset_index(drop=True)


# --------------------------------------------------------------------------- Fig. 5
def draw_fig5(ns, panel, output_file, *, lattice_fill=4.5, factor_map_clip=98, factor_map_gamma=2.0):
    """FIG_dopamine_overview's layout with the row-2 inputs taken from PLS."""
    from matplotlib.lines import Line2D

    bdf = ns["bivariate_moran_I_df"]
    nmf_cmp = bdf.iloc[bdf["bivariate_moran_I"].argmax()].name
    joint_adata, nmf_adata = ns["joint_adata"], ns["nmf_adata"]
    bench = ns["_fig_load_benchmark_metrics"](None)
    bench_order = ns["_fig_benchmark_order"](bench)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    import scanpy as sc

    panel = dict(panel, axis_title=panel["axis_title"].replace("MSI-PLS component ", "MSI-PLS "))
    with plt.rc_context(ns["FIG_RCPARAMS"]):
        heights = [5.9, 4.4, 4.2]
        fig = plt.figure(figsize=(18, sum(heights)), constrained_layout=True)
        rows = fig.subfigures(3, 1, height_ratios=heights)

        # row 1: measured source (unchanged)
        ax_lesion, ax_dopa, ax_nmf = rows[0].subplots(1, 3)
        key = ns["_fig_add_lesion_striatum"](joint_adata)
        palette = {"intact": "tab:blue", "lesioned": "tab:orange"}
        sc.pl.embedding(joint_adata, basis="spatial", color=key, palette=palette, na_color="lightgray",
                        na_in_legend=False, size=100, ax=ax_lesion, show=False, legend_loc="none")
        ax_lesion.legend(handles=[Line2D([0], [0], marker="o", linestyle="none", label=n, markerfacecolor=c,
                                         markeredgecolor="none", markersize=14) for n, c in palette.items()],
                         loc="lower right", frameon=False, fontsize=14, handletextpad=0.4)
        ax_lesion.set_title("Striatum: intact vs lesioned")
        sc.pl.embedding(joint_adata, basis="spatial", color="msi:Dopamine", size=100, ax=ax_dopa, show=False)
        sc.pl.embedding(nmf_adata, basis="spatial", color=nmf_cmp, size=100, ax=ax_nmf, show=False)
        ax_dopa.set_title("Measured dopamine (MALDI-MSI)")
        ax_nmf.set_title(f"NMF {nmf_cmp} — dopamine-associated\nlesion-response program "
                         f"(biv. $I$ = {bdf.loc[nmf_cmp, 'bivariate_moran_I']:.3f})", fontsize=11)
        row1 = [ax_lesion, ax_dopa, ax_nmf]
        for ax in row1:
            if ax is not ax_lesion:
                ax.set_facecolor(ns["FIG_MISSING_COLOR"])
            ns["_fig_rasterize_scatters"](ax)
            ax.set_box_aspect(1)
            ns["_fig_panel_letter"](ax, next(letters))

        # row 2: the transferred program, from the PLS decoders
        axes = rows[1].subplots(1, 5, gridspec_kw={"width_ratios": [1.5, 1.5, 0.55, 0.62, 1.45]})
        ns["_fig_spatial_scatter"](axes[0], panel["spatial_xy"] * np.array([1, -1]), panel["spatial_score"],
                                   f"{panel['axis_title']} — student\n(P22 spatial target)", s=10,
                                   clip_percentile=factor_map_clip, gamma=factor_map_gamma,
                                   cbar_label=panel["score_label"])
        ns["_fig_spatial_scatter"](axes[1], panel["multiome_xy"] * np.array([-1, -1]), panel["multiome_score"],
                                   f"{panel['axis_title']} — student\n(dissociated multiome target)", s=10,
                                   clip_percentile=factor_map_clip, gamma=factor_map_gamma,
                                   cbar_label=panel["score_label"])
        ns["_fig_loadings_dotplot"](axes[2], ns["_fig_top_loadings"](panel["genes"], 6),
                                    title=f"{panel['axis_title']}\ntop-6 genes", italic=True)
        ns["_fig_loadings_dotplot"](axes[3], ns["_fig_top_loadings"](panel["metabolites"], 6),
                                    title=f"{panel['axis_title']}\ntop-6 metabolites\n({panel['metab_source']})")
        ns["_fig_draw_enrichment_convergence"](
            axes[4], panel["convergence"], TOP_K,
            title=f"Gene-level convergence with the\nNMF dopamine program (top {TOP_K})")
        axes[4].set_ylabel(f"{panel['axis_title']} RNA loading (PLS)")
        for slot, ax in enumerate(axes):
            if slot == 3:
                continue
            ns["_fig_panel_letter"](ax, next(letters), x=0.09 if slot < 2 else -0.06)

        # row 3: within-source embedding benchmark (unchanged)
        ab = rows[2].subplots(1, 3, gridspec_kw={"width_ratios": [1.25, 1.55, 1.2]})
        ns["_fig_benchmark_bars"](ab[0], bench, "delta_auroc", bench_order, group="striatum",
                                  ylabel="$\\Delta$AUROC (smoothed − raw dopamine)", title="Dopamine structural gain")
        ns["_fig_benchmark_smoothness"](ab[1], bench, bench_order)
        ns["_fig_benchmark_bars"](ab[2], bench, "enrich_log_odds", bench_order, group="all",
                                  ylabel="log-odds (intact-striatal)", title="Intact-striatal specificity")
        for ax in ab:
            ns["_fig_panel_letter"](ax, next(letters))
        ns["_fig_fit_markers_to_lattice"](row1, fill=lattice_fill)
        for suffix in (".png", ".svg"):
            fig.savefig(output_file.with_suffix(suffix), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[fig] wrote {output_file.with_suffix('.png')}", flush=True)


# --------------------------------------------------------------------------- Fig. 6
def draw_fig6(ns, panel, output_file):
    from PIL import Image

    cell_types = ("D2MSN", "MXD")
    with plt.rc_context(ns["FIG_RCPARAMS"]):
        fig = plt.figure(figsize=(14.4, 12.5))
        sf_left, sf_right = fig.subfigures(1, 2, width_ratios=[1.0, 0.82], wspace=0.03)
        sf_a, sf_gsea, sf_motif = sf_left.subfigures(3, 1, height_ratios=[0.8, 1.0, 1.0], hspace=0.04)
        ax_a = sf_a.add_subplot(111)
        ax_a.imshow(np.asarray(Image.open(FIG6_PANEL_A_SOURCE).convert("RGB").crop(FIG6_PANEL_A_BOX)))
        ax_a.set_axis_off()
        for sf_cell, label in zip(np.atleast_1d(sf_gsea.subfigures(1, 2, wspace=0.05)), cell_types):
            host = sf_cell.add_subplot(111)
            host.set_axis_off()
            ns["_fig_draw_gsea"](host, panel["gsea"][label], label)
        motif_axes = np.atleast_1d(sf_motif.subplots(1, 2))
        sf_motif.subplots_adjust(left=0.2, right=0.98, wspace=0.75, bottom=0.2)
        for ax, label in zip(motif_axes, cell_types):
            ns["_fig_draw_top_motifs"](ax, panel["motifs"][label], label, n_top=15)
            ax.set_title(f"{label}: top TF motifs\n(GC-adjusted)")
            ax.set_xlabel("-log10(FDR q, GC-adjusted)")
            legend = ax.get_legend()
            handles, labels = legend.legend_handles, [t.get_text() for t in legend.texts]
            legend.remove()
        motif_axes[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.25, -0.17), ncol=3,
                             fontsize=8, frameon=False, handlelength=1.2, columnspacing=1.0)

        sf_d, sf_e = sf_right.subfigures(2, 1, height_ratios=[1.0, 0.62], hspace=0.06)
        ax_d = np.atleast_1d(sf_d.subplots(2, 1, sharex=True))
        sf_d.subplots_adjust(hspace=0.3, top=0.93, bottom=0.08)
        for ax, row in zip(ax_d, panel["concordance"]):
            hb = ax.hexbin(row["x"], row["y"], gridsize=55, bins="log", cmap="Blues", mincnt=1,
                           linewidths=0, rasterized=True)
            ax.axhline(0, color="#999999", lw=0.6, zorder=0)
            ax.axvline(0, color="#999999", lw=0.6, zorder=0)
            if ax is ax_d[-1]:
                ax.set_xlabel("linked gene: RNA loading (PLS)")
            ax.set_ylabel("peak: ATAC loading (PLS)")
            ax.set_title(f"{row['peak_type'].capitalize()} peaks ({row['n_links']:,} peak-gene links)")
            ax.text(0.03, 0.95, f"Spearman $\\rho$ = {row['rho']:.2f}\n"
                                f"shuffled genes: {row['null_mean']:+.3f} ± {row['null_sd']:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=10)
            cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
            cb.set_label("links per bin", fontsize=9)
            cb.ax.tick_params(labelsize=8)
        ax_e = sf_e.add_subplot(111)
        e = panel["nmf_peaks"]
        box = ax_e.boxplot([e["rest"], e["top"]], positions=[1, 1.7], widths=0.5, patch_artist=True,
                           showfliers=False)
        for patch, color in zip(box["boxes"], ["#bdbdbd", "#c0392b"]):
            patch.set_facecolor(color)
        ax_e.set_xticks([1, 1.7])
        ax_e.set_xticklabels([f"other linked peaks\n(n = {len(e['rest']):,})",
                              f"linked to NMF top-{TOP_K}\n(n = {len(e['top']):,})"])
        low, high = ax_e.get_ylim()
        ax_e.set_ylim(low, high + 0.35 * (high - low))
        ax_e.set_ylabel("peak: ATAC loading (PLS)")
        ax_e.set_title(f"Peaks of the NMF dopamine program's top-{TOP_K} genes")
        ax_e.text(0.5, 0.95, f"AUROC = {e['auroc']:.2f} ({ns['_fig_fmt_p'](e['p'])})\n"
                              f"MSN-identity ranking: AUROC = {e['baseline_auroc']:.2f}",
                  transform=ax_e.transAxes, ha="center", va="top", fontsize=10)
        for sf, letter in ((sf_a, "a"), (sf_gsea, "b"), (sf_motif, "c"), (sf_d, "d"), (sf_e, "e")):
            sf.text(0.0, 1.0, letter, fontsize=20, fontweight="bold", va="top", ha="left")
        fig.suptitle(f"{panel['axis_title']} {panel.get('subtitle', '(student PLS decoders), multiome target')}",
                     fontsize=13, y=1.04)
        for suffix in (".png", ".svg"):
            fig.savefig(output_file.with_suffix(suffix), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[fig] wrote {output_file.with_suffix('.png')}", flush=True)


# ------------------------------------------------------------------------------ main
def coordinates(tag, target, cells):
    import anndata as ad

    if target == "spatial":
        a = ad.read_h5ad(P.PROJ / (P.BASE + tag) / P.TARGETS["spatial"]["emb"], backed="r")
    else:
        a = ad.read_h5ad(Path(os.environ.get("SPATIALJEPA_DATA", f"{os.environ['SPATIALJEPA_HOME']}/data"))
                         / "aligned_data" / "target_rna_aligned_with_latents.h5ad", backed="r")
    xy = pd.DataFrame(np.asarray(a.obsm["spatial"])[:, :2], index=a.obs_names).reindex(cells)
    if xy.isna().any().any():
        raise ValueError(f"{target}: coordinates missing for some cells")
    return xy.to_numpy()


def redraw(args):
    import pickle

    namespaces = {}
    for variant in args.variants:
        run, _ = VARIANTS[variant]
        tag = P.RUNS[run]
        if tag not in namespaces:
            namespaces[tag] = script5_namespace(tag)
        out = args.out_dir / variant
        with open(out / "panels.pkl", "rb") as handle:
            panel5, panel6 = pickle.load(handle)
        draw_fig5(namespaces[tag], panel5, out / "SMA_mouse_dopamine_pls")
        draw_fig6(namespaces[tag], panel6, out / "SMA_mouse_pls_enrichment")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.redraw:
        return redraw(args)
    targets = {t: P.load_target(t) for t in ("multiome", "spatial")}
    stats = {t: (P.column_stats(T["rna"]), P.column_stats(T["atac"])) for t, T in targets.items()}
    Tm = targets["multiome"]
    ccre = P.ccre_membership(Tm["peaks"])
    fig_mask = np.asarray(Tm["peaks"].isin(Tm["fig_peaks"]))
    links = P.peak_gene_links(Tm["peaks"])
    prog = P.nmf_program()
    nmf_linked = np.asarray(Tm["peaks"].isin(links.loc[links["gene"].str.upper().isin(set(prog.nlargest(TOP_K).index)), "peak"]))
    any_linked = np.asarray(Tm["peaks"].isin(links["peak"]))
    msn_peak = P.plain_corr(Tm["atac"], stats["multiome"][1], P.module_scores(Tm["rna"], Tm["genes"])["MSN"].to_numpy())
    baseline_auroc, _ = P.auroc_stats(msn_peak[any_linked], nmf_linked[any_linked])

    namespaces, runs = {}, {}
    for variant in args.variants:
        run, axis = VARIANTS[variant]
        tag = P.RUNS[run]
        if run not in runs:
            runs[run] = {t: P.load_run(tag, t, targets[t]["cells"]) for t in targets}
        if tag not in namespaces:
            namespaces[tag] = script5_namespace(tag)
        ns, R = namespaces[tag], runs[run]
        title = "Dopamine axis" if axis == "dopamine axis" else axis.replace("MSI-PLS ", "MSI-PLS component ")
        print(f"[fig] === {variant}: {run}, {axis}", flush=True)

        beta, q = axis_beta(R["multiome"], axis)
        genes_m = gene_axis_loadings(R["multiome"], beta, Tm, stats["multiome"][0])
        genes_s = gene_axis_loadings(R["spatial"], beta, targets["spatial"], stats["spatial"][0])
        peaks = peak_axis_loadings(R["multiome"], beta, Tm, stats["multiome"][1])
        if q is not None:
            metabolites, metab_source = q, "decoder, measured MSI"
        else:
            metabolites = (None if run == "manuscript"
                           else source_metabolite_loadings(tag, beta, R["multiome"]["features"]))
            metab_source = "measured source MSI"
            if metabolites is None:
                # the June model behind the manuscript projection has no surviving source
                # latents: fall back to the decoded (imputed) target MSI
                msi = R["multiome"]["msi"]
                s = R["multiome"]["Z"] @ beta
                metabolites = msi.apply(lambda col: np.corrcoef(col, s)[0, 1])
                metab_source = "decoded target MSI"
        universe = Tm["fig_genes"]
        panel5 = dict(
            axis_title=title,
            score_label="axis score (imputed dopamine)" if axis == "dopamine axis" else "component score",
            spatial_xy=coordinates(tag, "spatial", targets["spatial"]["cells"]),
            multiome_xy=coordinates(tag, "multiome", Tm["cells"]),
            spatial_score=R["spatial"]["Z"] @ beta, multiome_score=R["multiome"]["Z"] @ beta,
            genes=genes_m[genes_m.index.isin(set(universe))], metabolites=metabolites, metab_source=metab_source,
            convergence=[convergence_stats(ns, "P22 spatial", genes_s, targets["spatial"]["fig_genes"]),
                         convergence_stats(ns, "Multiome", genes_m, universe)],
        )
        out = args.out_dir / variant
        out.mkdir(parents=True, exist_ok=True)
        draw_fig5(ns, panel5, out / "SMA_mouse_dopamine_pls")

        # Fig. 6 inputs
        fig_peaks = Tm["peaks"][fig_mask]
        scores = peaks.to_numpy()[fig_mask]
        gsea, motif_inputs = {}, out / "motif_inputs"
        for label in ("D2MSN", "MXD"):
            pre, leading, _ = gsea_object(scores, ccre[label][fig_mask], fig_peaks, f"{label}_cCRE")
            gsea[label] = pre
            fg = fig_peaks[leading]
            bg = fig_peaks[ccre[label][fig_mask]].difference(fg)
            motif_inputs.mkdir(parents=True, exist_ok=True)
            P.peaks_bed(fg).to_csv(motif_inputs / f"{label.lower()}_foreground.bed", sep="\t", header=False, index=False)
            P.peaks_bed(bg).to_csv(motif_inputs / f"{label.lower()}_background.bed", sep="\t", header=False, index=False)
        run_label = run.replace(" ", "_")
        motifs_dir = args.motifs_dir
        if axis != "dopamine axis":
            motifs_dir, run_label = out / "motifs_gc", variant
            if not (motifs_dir / f"{run_label}_mxd_gc_controlled_motifs.tsv").exists():
                print(f"[fig] GC-adjusted motif test for {variant} (scenicplus env)", flush=True)
                subprocess.run([SCENICPLUS_PY, str(HERE / "gc_matched_motif_enrichment.py"), "--processes", "16",
                                "--out-dir", str(motifs_dir), "--inputs", f"{run_label}={motif_inputs}"],
                               check=True, stdout=subprocess.DEVNULL)
        motifs = {label: motif_table(motifs_dir / f"{run_label}_{label.lower()}_gc_controlled_motifs.tsv")
                  for label in ("D2MSN", "MXD")}
        gene_load = genes_m[~genes_m.index.duplicated()]
        conc = []
        for row in P.link_concordance(peaks, gene_load, links):
            sub = links[links["peak_type"].eq(row["peak_type"]) & links["gene"].isin(gene_load.index)]
            x, y = gene_load.reindex(sub["gene"]).to_numpy(), peaks.reindex(sub["peak"]).to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            conc.append(dict(row, x=x[ok], y=y[ok]))
        pv = peaks.to_numpy()
        auroc, p = P.auroc_stats(pv[any_linked], nmf_linked[any_linked])
        panel6 = dict(axis_title=title, gsea=gsea, motifs=motifs, concordance=conc,
                      nmf_peaks=dict(top=pv[any_linked & nmf_linked], rest=pv[any_linked & ~nmf_linked],
                                     auroc=auroc, p=p, baseline_auroc=baseline_auroc))
        import pickle

        with open(out / "panels.pkl", "wb") as handle:
            pickle.dump((panel5, panel6), handle)
        draw_fig6(ns, panel6, out / "SMA_mouse_pls_enrichment")


if __name__ == "__main__":
    main()
