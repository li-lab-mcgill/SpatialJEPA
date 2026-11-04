"""Supplementary figures for held-out evaluation (RNA-MSI held-out animals; RNA-ATAC held-out Multiome).

  S_heldout_rnamsi_agreement   (a) validation loss of the Fig. 5 teacher's early-stopping pass;
                               (b) teacher self-consistency and (c) student-teacher agreement, full
                               schedule vs early-stopped teacher; (d-f) cluster agreement on held-out
                               striatum (cluster_agreement.py), early-stopped student vs controls.
  S_heldout_rnamsi_fig5def     the Fig. 5d-f metrics (9__) in-sample, on the held-out half of the
                               training section and on held-out animals.
  S_heldout_multiome_p22       10x Multiome nuclei placed on the P22 section through each model's
                               embedding (heldout_multiome_alignment.py): teacher-cluster recovery,
                               placed cell types, known-pairing W1, and agreement maps.

Reads the benchmark tables under $SPATIALJEPA_OUT/spatialjepa_benchmark/animal_transfer_p2000{,_split}/evaluation,
the Fig. 5 model's early_stopping.json and $SPATIALJEPA_OUT/MultiGATE/rna_atac_analysis/20260505_190037/heldout_alignment.
Writes <out_dir>/<name>.{png,svg} and <out_dir>/<name>_values.csv. Env: nichecompass_liana.
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

OUT = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs")
BENCH = [OUT / "spatialjepa_benchmark" / r / "evaluation" for r in ("animal_transfer_p2000", "animal_transfer_p2000_split")]
ES_JSON = OUT / "spatialjepa_models" / "V11L12-109_B1_seed3" / "early_stopping.json"
ALIGN = OUT / "MultiGATE" / "rna_atac_analysis" / "20260505_190037" / "heldout_alignment"

# main-text Fig. 5d-f colours for teacher / student / non-spatial; controls in greys
TEACHER, STUDENT, NONSPATIAL = "#66c2a5", "#fc8d62", "#8da0cb"
GREY, DARK = "#9a9a9a", "#4d4d4d"
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
      "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
      "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False}

MSI_MODELS = {  # benchmark name -> (label, colour)
    "A_early_teacher": ("teacher", TEACHER),
    "A_early": ("student", STUDENT),
    "A": ("student, full-\nschedule teacher", "#f4c3ad"),
    "B": ("student of id.-\ngraph teacher", GREY),
    "E": ("identity-\ngraph VAE", GREY),
    "G": ("non-spatial\nSpatialMETA", NONSPATIAL),
    "input_pca10": ("input PCA", DARK),
}
RELATIONS = {("pairwise", "other_animal"): "held-out animals", ("within_section", "same_section"): "held-out half",
             ("within_section", "other_animal"): "held-out animals (half-trained)"}


def panel_label(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, out, name, values=None):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if values is not None:
        values.to_csv(out / f"{name}_values.csv", index=False)
    print("wrote", out / f"{name}.png", flush=True)


# ------------------------------------------------------------------------------- RNA-MSI
def unit_means(frame, keys):
    """Mean over seeds within each held-out unit (fold x section)."""
    return frame.groupby(keys + ["fold", "section"], as_index=False)["value"].mean()


def cluster_agreement():
    L = pd.concat([pd.read_csv(d / "cluster_agreement_long.csv") for d in BENCH])
    L = L[L.model.isin(MSI_MODELS)]
    metrics = {("ari", "A_early_teacher"): "ARI vs teacher clusters", ("ari", "anatomy"): "ARI vs region x lesion",
               ("morans_i", "none"): "Moran's I of the latent"}
    L = L[[(m, r) in metrics for m, r in zip(L.metric, L.reference)]].copy()
    L["metric_name"] = [metrics[(m, r)] for m, r in zip(L.metric, L.reference)]
    L = L[[(d, r) in RELATIONS for d, r in zip(L.design, L.relation)]]
    L["held_out"] = [RELATIONS[(d, r)] for d, r in zip(L.design, L.relation)]
    return unit_means(L, ["held_out", "metric_name", "model"]), list(metrics.values())


def teacher_consistency():
    s = pd.read_csv(BENCH[0] / "teacher_consistency_summary.csv")
    keep = {("full", "in_sample x in_sample", True): "two seeds,\nsame\nsection",
            ("full", "heldout_animal x in_sample", False): "own vs\nother-animal\nteacher",
            ("full", "heldout_animal x heldout_animal", False): "two other-\nanimal\nteachers"}
    s = s[[(k, p, u) in keep for k, p, u in zip(s.kind, s["pair"], s.same_unit)]
          & s.family.isin(["A_teacher", "A_early_teacher"])].copy()
    s["comparison"] = [keep[(k, p, u)] for k, p, u in zip(s.kind, s["pair"], s.same_unit)]
    return s, list(keep.values())


def student_teacher():
    d = pd.concat([pd.read_csv(b / "student_teacher_diagnostics_long.csv") for b in BENCH])
    d = d[d.config.isin(["A", "A_early"]) & ((d.candidate == d.config) | (d.candidate == d.config + "_oracle_ridge"))]
    d = d[[(g, r) in {("pairwise", "in_sample"), ("pairwise", "other_animal"), ("within_section", "same_section")}
           for g, r in zip(d.design, d.relation)]].copy()
    d["held_out"] = d.relation.map({"in_sample": "training spots", "same_section": "held-out half",
                                    "other_animal": "held-out animals"})
    d["who"] = np.where(d.candidate.str.endswith("oracle_ridge"), "oracle", "student")
    d = d.rename(columns={"linear_r2": "value"})
    return unit_means(d, ["held_out", "config", "who"])


def draw_agreement(out):
    with open(ES_JSON) as handle:
        es = json.load(handle)
    curve = pd.DataFrame(es["curve"])
    ca, metric_names = cluster_agreement()
    tc, comparisons = teacher_consistency()
    st = student_teacher()
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.2, 5.9))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0], hspace=0.6, wspace=0.5)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(curve.epoch, curve.val_loss / 1e3, color=TEACHER, lw=1.6)
        best = es["best_epoch"]
        ax.axvline(best, color=DARK, lw=0.8, ls="--")
        ax.text(best + 4, curve.val_loss.max() / 1e3 * 0.97, f"best epoch {best}\n(of 1,000)", fontsize=7, va="top")
        ax.set(xlabel="teacher epoch", ylabel="loss on held-out blocks (x10$^3$)", title="Teacher early stopping")
        panel_label(ax, "a", x=-0.3, y=1.04)

        ax = fig.add_subplot(gs[0, 1])
        x = np.arange(len(comparisons))
        for k, (fam, col, lab) in enumerate((("A_teacher", "#cfeee3", "full schedule"),
                                             ("A_early_teacher", TEACHER, "early-stopped"))):
            v = tc[tc.family.eq(fam)].set_index("comparison").reindex(comparisons)
            ax.bar(x + (k - 0.5) * 0.36, v.linear_r2_mean, 0.34, color=col, edgecolor=DARK, lw=0.4, label=lab)
            ax.errorbar(x + (k - 0.5) * 0.36, v.linear_r2_mean,
                        yerr=[v.linear_r2_mean - v.linear_r2_min, v.linear_r2_max - v.linear_r2_mean],
                        fmt="none", color=DARK, lw=0.6, capsize=1.5)
        ax.set_xticks(x, comparisons, fontsize=6)
        ax.set(ylim=(0, 1.15), yticks=np.arange(0, 1.01, 0.2), ylabel="linear $R^2$ between teacher latents",
               title="Teacher self-consistency")
        ax.legend(loc="upper left", fontsize=6, ncol=2, handlelength=1, columnspacing=0.8)
        panel_label(ax, "b", x=-0.3, y=1.04)

        ax = fig.add_subplot(gs[0, 2])
        settings = ["training\nspots", "held-out\nhalf", "held-out\nanimals"]
        x = np.arange(len(settings))
        for k, (cfg, who, col, lab) in enumerate((("A", "student", "#f4c3ad", "full-schedule teacher"),
                                                  ("A_early", "student", STUDENT, "early-stopped teacher"),
                                                  ("A_early", "oracle", "white", "ceiling (ridge)"))):
            v = st[st.config.eq(cfg) & st.who.eq(who)].groupby("held_out")["value"]
            m = v.mean().reindex([x_.replace("\n", " ") for x_ in settings])
            ax.bar(x + (k - 1) * 0.26, m, 0.24, color=col, edgecolor=DARK, lw=0.4, label=lab,
                   hatch="////" if who == "oracle" else None)
        ax.set_xticks(x, settings)
        ax.set(ylim=(0, 1.32), yticks=np.arange(0, 1.01, 0.2), ylabel="linear $R^2$, student vs its teacher",
               title="Student-teacher agreement")
        ax.legend(loc="upper right", fontsize=6, handlelength=1)
        panel_label(ax, "c", x=-0.3, y=1.04)

        order = list(MSI_MODELS)          # top to bottom; models on a shared vertical axis
        rows, first = [], None
        for j, metric in enumerate(metric_names):
            ax = fig.add_subplot(gs[1, j], sharey=first)
            first = first or ax
            for i, m in enumerate(order):
                if m == "A_early_teacher" and metric.startswith("ARI vs teacher"):
                    continue                     # the teacher's own clusters
                for held, marker, dy in (("held-out animals", "o", -0.16), ("held-out half", "^", 0.16)):
                    v = ca[(ca.model == m) & (ca.metric_name == metric) & (ca.held_out == held)].value
                    ax.scatter(v, np.full(len(v), i + dy), s=7, color=MSI_MODELS[m][1], alpha=0.55, lw=0, marker=marker)
                    ax.scatter(v.mean(), i + dy, s=34, color=MSI_MODELS[m][1], edgecolor="black", lw=0.6, marker=marker,
                               zorder=3)
                    rows.append(dict(metric=metric, model=m, held_out=held, mean=v.mean(), n_units=len(v)))
            ax.set_yticks(range(len(order)), [MSI_MODELS[m][0].replace("-\n", "-").replace("\n", " ") for m in order], fontsize=6.3)
            ax.set_ylim(len(order) - 0.5, -0.5)
            if j:
                ax.tick_params(labelleft=False)
            ax.grid(axis="y", color="#eeeeee", lw=0.5)
            ax.set_title(metric)
            panel_label(ax, "def"[j], x=-0.62 if j == 0 else -0.08, y=1.04)
        handles = [Line2D([], [], marker="o", ls="", color=DARK, label="held-out animals (6 units)"),
                   Line2D([], [], marker="^", ls="", color=DARK, label="held-out half of the training section (6 units)")]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.02), fontsize=7)
        save(fig, out, "S_heldout_rnamsi_agreement", pd.DataFrame(rows))


def draw_fig5def(out):
    L = pd.concat([pd.read_csv(d / "dopamine_structure_long.csv") for d in BENCH])
    models = ["A_early_teacher", "A_early", "G", "E", "input_pca10"]
    L = L[L.model.isin(models) & L.metric.isin(["structural_gain", "moran_intact_region", "specificity_frac"])
          & (L.latent.eq("joint") | L.model.eq("input_pca10"))]
    L = L[L.setting.isin(["in_sample_full", "transfer_paired"]) & L.design.isin(["pairwise", "within_section"])]
    setting = {("pairwise", "in_sample"): "training\nsection", ("within_section", "same_section"): "held-out\nhalf",
               ("pairwise", "other_animal"): "held-out\nanimals"}
    L = L[[(d, r) in setting for d, r in zip(L.design, L.relation)]].copy()
    L["where"] = [setting[(d, r)] for d, r in zip(L.design, L.relation)]
    U = unit_means(L, ["where", "metric", "model"])
    titles = {"structural_gain": "Dopamine structural gain\n(Fig. 5d)", "moran_intact_region":
              "Moran's I, intact striatum\n(Fig. 5e)", "specificity_frac": "Intact-striatal share of\ndopamine-high spots (Fig. 5f)"}
    xs = list(setting.values())
    rows = []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.45))
        for ax, (metric, title) in zip(axes, titles.items()):
            for k, m in enumerate(models):
                label, col = MSI_MODELS[m]
                v = U[(U.metric == metric) & (U.model == m)].groupby("where")["value"]
                mean, sem = v.mean().reindex(xs), v.sem().reindex(xs)
                xx = np.arange(len(xs)) + (k - 2) * 0.07
                ax.errorbar(xx, mean, yerr=sem, color=col, marker="o", ms=4, lw=1.3, capsize=1.5,
                            ls="--" if m == "input_pca10" else "-", label=label.replace("\n", " "))
                rows += [dict(metric=metric, model=m, where=w.replace("\n", " "), mean=mean[w], sem=sem[w],
                              n_units=int(v.size().get(w, 0))) for w in xs]
            if metric == "structural_gain":
                ax.axhline(0, color=DARK, lw=0.6, ls=":")
            ax.set_xticks(range(len(xs)), xs)
            ax.set_title(title)
            panel_label(ax, "abc"[list(titles).index(metric)], x=-0.2)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=7)
        save(fig, out, "S_heldout_rnamsi_fig5def", pd.DataFrame(rows))


# ------------------------------------------------------------------------------- RNA-ATAC
ATAC_MODELS = {"student": ("student", STUDENT), "student_in_teacher": ("student vs\nteacher P22", "#f4c3ad"),
               "teacher_nograph": ("teacher,\nno graph", TEACHER), "nonspatial": ("non-spatial\nMultiGATE", NONSPATIAL),
               "multivi": ("MultiVI", GREY), "pca": ("RNA PCA /\nATAC LSI", DARK)}
COUPLINGS = {"knn": ("15-NN", "o"), "ot": ("balanced OT", "s"), "uot": ("unbalanced OT", "D")}


def agreement_map(own, teacher):
    """Rename back-projected clusters to the teacher clusters they overlap most (one-to-one)."""
    a, ua = pd.factorize(own)
    b, ub = pd.factorize(teacher)
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (a, b), 1)
    r, c = linear_sum_assignment(-C)
    mapping = dict(zip(r, c))
    return np.array([mapping.get(i, -1) for i in a]) == b


def draw_multiome(out):
    t = pd.read_csv(ALIGN / "alignment_summary.csv", index_col=[0, 1])
    p = pd.read_csv(ALIGN / "placement_w1_summary.csv", index_col=[0, 1])
    null = pd.read_csv(ALIGN / "celltypist_permutation_null_knn.csv")
    maps = {}
    for f in ALIGN.glob("backprojected_maps*.npz"):
        z = np.load(f, allow_pickle=True)
        maps.update({k: z[k] for k in z.files})
    xy = maps["coords"][:, :2]
    models = list(ATAC_MODELS)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.2, 6.3))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0], hspace=0.95, wspace=0.5)

        ax = fig.add_subplot(gs[0, 0])
        for i, m in enumerate(models):
            ax.plot([i - 0.3, i + 0.3], [t.loc[(m, "knn"), "in_sample_ari"]] * 2, color=DARK, lw=1.2)
            for k, (c, (lab, mk)) in enumerate(COUPLINGS.items()):
                ax.scatter(i + (k - 1) * 0.18, t.loc[(m, c), "backproject_ari"], marker=mk, s=26,
                           color=ATAC_MODELS[m][1], edgecolor="black", lw=0.5, zorder=3)
        ax.set_xticks(range(len(models)), [ATAC_MODELS[m][0].replace("\n", " ") for m in models], rotation=50, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set(ylabel="ARI with teacher clusters on P22", ylim=(0, 1.05), title="Multiome clusters on P22")
        panel_label(ax, "a", x=-0.3, y=1.04)

        ax = fig.add_subplot(gs[0, 1])
        nm = null.assign(kind=np.where(null.rep < 0, "real", "permuted")).groupby(["model", "kind"])[["nmi", "kappa"]].mean()
        for i, m in enumerate(models):
            ax.bar(i, t.loc[(m, "knn"), "celltypist_nmi"], 0.6, color=ATAC_MODELS[m][1], edgecolor=DARK, lw=0.4)
            ax.plot([i - 0.3, i + 0.3], [nm.loc[(m, "permuted"), "nmi"]] * 2, color="black", lw=1)
        ax.set_xticks(range(len(models)), [ATAC_MODELS[m][0].replace("\n", " ") for m in models], rotation=50, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set(ylabel="NMI with teacher clusters on P22", title="CellTypist labels on P22")
        panel_label(ax, "b", x=-0.3, y=1.04)

        ax = fig.add_subplot(gs[0, 2])
        for i, m in enumerate(models):
            for k, (c, (lab, mk)) in enumerate(COUPLINGS.items()):
                row = p.loc[(m, c)]
                ax.errorbar(i + (k - 1) * 0.18, row.median_gain / 23, yerr=[[(row.median_gain - row.gain_lo) / 23],
                            [(row.gain_hi - row.median_gain) / 23]], fmt=mk, ms=4.5, color=ATAC_MODELS[m][1],
                            mec="black", mew=0.5, ecolor=DARK, elinewidth=0.6)
        ax.axhline(0, color=DARK, lw=0.6, ls=":")
        ax.set_xticks(range(len(models)), [ATAC_MODELS[m][0].replace("\n", " ") for m in models], rotation=50, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set(ylabel="W1 gain over a same-cell-type\npartner (spot pitches)", title="Known RNA-ATAC pairing")
        handles = [Line2D([], [], marker=mk, ls="", color=DARK, label=lab) for lab, mk in COUPLINGS.values()]
        ax.legend(handles=handles, loc="upper right", fontsize=6.3)
        panel_label(ax, "c", x=-0.3, y=1.04)

        tk = maps["teacher_K10"]
        rows = []
        for j, m in enumerate(("student", "nonspatial", "multivi")):
            ax = fig.add_subplot(gs[1, j])
            ok = agreement_map(maps[f"{m}__knn__own_K10"], tk)
            ax.scatter(xy[:, 0], xy[:, 1], c=np.where(ok, "#1b7837", "#e0e0e0"), s=2.2, lw=0, rasterized=True)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{ATAC_MODELS[m][0].replace(chr(10), ' ')}: {ok.mean():.0%} match")
            rows.append(dict(model=m, frac_agree=ok.mean()))
            panel_label(ax, "def"[j], x=-0.05, y=1.12)
        handles = [Line2D([], [], marker="s", ls="", color="#1b7837", label="placed Multiome cluster = teacher cluster"),
                   Line2D([], [], marker="s", ls="", color="#e0e0e0", label="different cluster")]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=7)
        save(fig, out, "S_heldout_multiome_p22", pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    draw_agreement(args.out_dir)
    draw_fig5def(args.out_dir)
    draw_multiome(args.out_dir)


if __name__ == "__main__":
    main()
