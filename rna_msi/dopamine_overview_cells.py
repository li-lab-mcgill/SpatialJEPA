try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Regenerate the Fig. 5 dopamine overview from one SpatialJEPA run, headless.
#
# dopamine_nmf_signature.py is an interactive, cell-based script whose later
# cells (cCRE GSEA, motif hand-off, regulons, CATlas) feed other figures. This driver
# executes only the cells Fig. 5 needs -- NMF, target MOFA+, dopamine-factor selection
# and the figure functions -- in one namespace, then draws FIG_dopamine_overview.
#
# Run selection follows script 5's env overrides:
#   SPATIALJEPA_RUN_TAG        model/projection dir suffix (e.g. "__fix2000")
#   SPATIALJEPA_NMF_CACHE_DIR  reuse another run's NMF fit (source-data only)
#   SPATIALJEPA_FIG_DIR        where figures go (never defaults to the manuscript here)
#   MOFA_GPU_DEVICE            GPU index within CUDA_VISIBLE_DEVICES for MOFA+
# Panels d-f read <SPATIALJEPA_OUT>/multigrate_mouse_sma/<RUN_ID>/dopamine_metrics.csv from
# dopamine_embedding_benchmark.py (run with the same SPATIALJEPA_RUN_TAG); --benchmark-csv
# overrides it, e.g. to pair g-j from one run with d-f from another.
#
# MOFA+ runs single-threaded and is the slow step, and the two targets are independent:
# `--mofa-only spatial|multiome` fits just that target's model (same arguments as the
# script-5 cell) so both can run in parallel; a final plain run then finds both model
# files, skips training and draws the figure.
#
# Env: $PY_ANALYSIS

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

SCRIPT = Path(__file__).resolve().parent / "dopamine_nmf_signature.py"

# cells needed to fit one target's MOFA+ model; the MOFA cell is then filtered to the
# statement that fits that target
MOFA_CELLS = (
    "Load data",
    "Load joint data",
    "MOFA+ analysis of spatial and multiome targets",
    "Load trimodal targets",
    "Run MOFA+ on targets",
)
MOFA_TARGET_VARS = {"spatial": "spatial_mofa_outfile", "multiome": "multiome_mofa_outfile"}

# headers (prefix match) of the script-5 cells Fig. 5 depends on, in file order
FIG5_CELLS = (
    "Load data",
    "Load joint data",
    "Run NMF on spatial Visium data",
    "MOFA+ analysis of spatial and multiome targets",
    "extract cCREs related to MXD",
    "Load trimodal targets",
    "Run MOFA+ on targets",
    "load model for downstream analysis",
    "explore dopamine MOFA model",
    "",                      # unnamed cell: dopamine top hits + factor selection
    "figure functions",
)

# cells behind Fig. 6 (regulatory chromatin of the student dopamine factor), run after
# the factor selection so they use the chosen (or overridden) student factor
FIG6_CELLS = (
    "ATAC cCRE enrichment of the dopamine-associated factor",
    "find top features for the dopamine / C12 factor",
    "plot cCRE enrichment of the top C12 features",
    "TF motif enrichment hand-off",
    "RNA-side regulon enrichment",
)
# Fig. 6a is a screenshot adapted from Zhang et al. (2023); it is reused from the
# manuscript PNG (pixel box: left, top, right, bottom)
FIG6_PANEL_A_SOURCE = Path(
    f"{os.environ['SPATIALJEPA_MANUSCRIPT']}/figures/SMA_mouse_mofa_enrichment.png"
)
FIG6_PANEL_A_BOX = (70, 0, 1150, 575)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate the Fig. 5 dopamine overview from one SpatialJEPA run.",
        allow_abbrev=False,
    )
    parser.add_argument("--mofa-only", choices=tuple(MOFA_TARGET_VARS), default=None,
                        help="only fit this target's MOFA+ model, then exit")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="figure directory (also exported as SPATIALJEPA_FIG_DIR)")
    parser.add_argument("--name", default="SMA_mouse_dopamine_nmf_mofa")
    parser.add_argument("--teacher-factor", default=None,
                        help="override the automatic teacher dopamine factor, e.g. Factor8")
    parser.add_argument("--student-factor", default=None,
                        help="override the automatic student dopamine factor, e.g. Factor6")
    parser.add_argument("--benchmark-csv", type=Path, default=None,
                        help="dopamine_metrics.csv for panels d-f (default: this run's)")
    parser.add_argument("--fig6", action="store_true",
                        help="also run the Fig. 6 cells (cCRE GSEA, motifs, regulons) on the "
                             "student factor and draw the Fig. 6 board")
    parser.add_argument("--fig6-name", default="SMA_mouse_mofa_enrichment")
    parser.add_argument("--gc-motifs", type=Path, default=None,
                        help="<dir>/<run>: use gc_matched_motif_enrichment.py's GC-adjusted (logistic) "
                             "motif statistics for panels c-d, e.g. .../gc_matched_motifs/fixed")
    return parser.parse_args()


def split_cells(path: Path):
    lines = path.read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.match(r"^#\s?%%", line)] + [len(lines)]
    for a, b in zip(starts, starts[1:]):
        header = re.sub(r"^#\s?%%\s*", "", lines[a]).strip()
        # pad with blank lines so tracebacks report script-5 line numbers
        yield header, "\n" * a + "".join(lines[a:b])


def only_target_statement(code: str, target: str) -> str:
    """Keep the top-level statement of the MOFA cell that assigns this target's outfile."""
    tree = ast.parse(code)
    keep = [
        node for node in tree.body
        if MOFA_TARGET_VARS[target] in {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    ]
    if len(keep) != 1:
        raise RuntimeError(f"expected one MOFA statement for {target!r}, found {len(keep)}")
    tree.body = keep
    return ast.unparse(tree)


def orient_factor(model, factor, view, anchor="msi:Dopamine"):
    """Present `factor` of a mofax model with `anchor` loading positively.

    MOFA factors are defined up to a joint sign flip of Z and W, so this is a relabelling.
    The Fig. 5 drawing functions orient factors themselves, but the Fig. 6 cells rank
    peaks by the raw signed ATAC loadings; without this a factor whose dopamine weight
    happens to be negative shows the cCRE enrichment at the wrong end. Applied in
    memory by wrapping the model's accessors; the model file is untouched.
    """
    import pandas as pd

    weight = model.get_weights(views=[view], factors=[factor], df=True)
    if float(weight.loc[anchor].iloc[0]) >= 0:
        return False
    get_weights, get_factors = model.get_weights, model.get_factors

    def flipped(accessor):
        def wrapper(*args, df=False, absolute_values=False, **kwargs):
            if absolute_values:
                return accessor(*args, df=df, absolute_values=True, **kwargs)
            out = accessor(*args, df=True, **kwargs)
            if isinstance(out, pd.DataFrame) and factor in out.columns:
                out = out.copy()
                out[factor] = -out[factor]
            return out if df else out.to_numpy()
        return wrapper

    model.get_weights = flipped(get_weights)
    model.get_factors = flipped(get_factors)
    print(f"[fig5] {factor}: flipped so {anchor} loads positively ({view})", flush=True)
    return True


def run_cells(cells, namespace, *, mofa_only=None, after=None):
    """Execute the listed script-5 cells in file order; ``after`` maps a cell header
    to a callback run right after that cell."""
    wanted = list(cells)
    for header, code in split_cells(SCRIPT):
        if not wanted or not header.startswith(wanted[0]) or (wanted[0] == "" and header):
            continue
        key = wanted.pop(0)
        if mofa_only and header.startswith("Run MOFA+ on targets"):
            code = only_target_statement(code, mofa_only)
        print(f"[fig5] running cell: {header or '(unnamed)'}", flush=True)
        exec(compile(code, str(SCRIPT), "exec"), namespace)
        if after and key in after:
            after[key](namespace)
    if wanted:
        raise RuntimeError(f"script-5 cells not found: {wanted}")


def repair_degenerate_gsea(pre, term, n_perm=1000, seed=0):
    """Permutation statistics for a prerank result gseapy could not normalise.

    gseapy normalises ES by the mean of the same-sign null ES. When the ranking has a
    heavy tail on one side, every random set can have the opposite sign; gseapy then
    reports NES = 1 and p = 1 although the observed ES exceeds the whole null. In that
    case recompute the same weighted ES statistic over `n_perm` label permutations:
    p = (1 + #null >= ES) / (1 + n_perm) (one-sided, in the direction of ES) and
    NES = ES / mean|null ES|. The result entry is updated in place and both versions
    are printed.
    """
    import numpy as np

    r = pre.results[term]
    scores = np.abs(np.asarray(pre.ranking, dtype=float))
    hit = np.zeros(len(scores), dtype=bool)
    hit[np.asarray(r["hits"], dtype=int)] = True

    def es(h):
        run = np.cumsum(np.where(h, scores, 0.0) / scores[h].sum() - (~h) / (~h).sum())
        return run[np.argmax(np.abs(run))]

    observed = es(hit)
    rng = np.random.default_rng(seed)
    null = np.array([es(rng.permutation(hit)) for _ in range(n_perm)])
    same_sign = null[np.sign(null) == np.sign(observed)]
    print(f"[fig6] {term}: gseapy ES={r['es']:.3f} NES={r['nes']:.3f} p={r['pval']:.3g}; "
          f"recomputed ES={observed:.3f}, same-sign null {len(same_sign)}/{n_perm}", flush=True)
    if len(same_sign) > 0:
        return False
    exceed = (null >= observed).sum() if observed > 0 else (null <= observed).sum()
    r["nes"] = float(observed / np.abs(null).mean())
    r["pval"] = r["fdr"] = float((1 + exceed) / (1 + n_perm))
    print(f"[fig6] {term}: no same-sign null ES -> NES = ES/mean|null| = {r['nes']:.3f}, "
          f"permutation p = {r['pval']:.3g}", flush=True)
    return True


def use_gc_adjusted_motifs(namespace, spec):
    """Swap panel c-d motif tables for gc_matched_motif_enrichment.py's GC-adjusted test.

    `spec` is <gc_matched_motifs dir>/<run label>; the logistic test (hit ~ foreground +
    GC + GC^2, same foreground/background peaks as 5__) supplies pvalue/padj, so the
    significance tiers in panels c and d mean the same thing as before, net of GC.
    """
    import numpy as np
    import pandas as pd

    for ct in ("d2msn", "mxd"):
        table = pd.read_csv(spec.parent / f"{spec.name}_{ct}_gc_controlled_motifs.tsv", sep="\t")
        enr = pd.DataFrame({
            "motif_name": table["motif_name"].str.replace(r" \(MA[0-9.]+\)$", "", regex=True),
            "pvalue": table["logit_pvalue"],
            "padj": table["logit_padj"],
            "odds_ratio": np.exp(table["logit_log_odds_fg_adj_gc"]),
        }).sort_values(["padj", "pvalue"]).reset_index(drop=True)
        namespace[f"{ct}_motif_enrichment"] = enr
        print(f"[fig6] {ct}: GC-adjusted motif table, {(enr.padj < 0.05).sum()} at FDR<0.05, "
              f"{(enr.pvalue < 0.05).sum()} nominal; top: {', '.join(enr.motif_name[:6])}", flush=True)


def fig6_board(namespace, output_file, panel_a=FIG6_PANEL_A_SOURCE, panel_a_box=FIG6_PANEL_A_BOX):
    """Fig. 6 in the manuscript layout: (a) cell-type cCRE maps, (b) cCRE GSEA,
    (c) top TF motifs, (d) RNA-regulon x ATAC-motif concordance.

    Panels b-d are FIG_ccre_tf_overview's drawing helpers on this run's results; panel a
    is the adapted literature screenshot, copied from the manuscript figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    ns = namespace
    cell_types = ("D2MSN", "MXD")
    gsea = {"D2MSN": ns["d2msn_multiome_gsea"], "MXD": ns["mxd_multiome_gsea"]}
    motifs = {"D2MSN": ns["d2msn_motif_enrichment"], "MXD": ns["mxd_motif_enrichment"]}
    with plt.rc_context(ns["FIG_RCPARAMS"]):
        fig = plt.figure(figsize=(7 + 3.2 * len(cell_types), 12))
        sf_left, sf_right = fig.subfigures(1, 2, width_ratios=[1.0, 1.15], wspace=0.02)
        sf_a, sf_gsea, sf_motif = sf_left.subfigures(3, 1, height_ratios=[0.8, 1.0, 1.0], hspace=0.04)
        ax_a = sf_a.add_subplot(111)
        ax_a.imshow(np.asarray(Image.open(panel_a).convert("RGB").crop(panel_a_box)))
        ax_a.set_axis_off()
        for sf_cell, label in zip(np.atleast_1d(sf_gsea.subfigures(1, len(cell_types), wspace=0.05)),
                                  cell_types):
            host = sf_cell.add_subplot(111)
            host.set_axis_off()
            ns["_fig_draw_gsea"](host, gsea[label], label)
        # The gaps must hold the right-hand panel's motif / TF tick labels: at matplotlib's
        # default wspace they ran into the left panel's legend and largest dots.
        for ax, label in zip(np.atleast_1d(sf_motif.subplots(1, len(cell_types),
                                                             gridspec_kw={"wspace": 0.75})), cell_types):
            ns["_fig_draw_top_motifs"](ax, motifs[label], label, n_top=15)
        for ax, label in zip(np.atleast_1d(sf_right.subplots(1, len(cell_types),
                                                             gridspec_kw={"wspace": 0.5})), cell_types):
            ns["_fig_draw_regulon_concordance"](ax, ns["regulon_res"], motifs[label], label,
                                                n_top=20, title=f"motif concordance ({label})")
        for sf, letter in ((sf_a, "a"), (sf_gsea, "b"), (sf_motif, "c"), (sf_right, "d")):
            sf.text(0.0, 1.0, letter, fontsize=20, fontweight="bold", va="top", ha="left")
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[fig6] wrote {output_file}", flush=True)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SPATIALJEPA_FIG_DIR"] = str(args.out_dir)

    namespace = {"__name__": "fig5_driver", "__file__": str(SCRIPT)}
    if args.mofa_only:
        run_cells(MOFA_CELLS, namespace, mofa_only=args.mofa_only)
        print(f"[fig5] MOFA+ {args.mofa_only} model: "
              f"{namespace[MOFA_TARGET_VARS[args.mofa_only]]}", flush=True)
        return
    # The automatic choice (largest |dopamine loading| / bivariate Moran's I vs imputed
    # dopamine) can land on a factor driven by other tissue structure; a factor chosen
    # instead for its dopamine-related MSI loadings is passed here. Applied right after
    # the selection cell, so the Fig. 6 cells use it too.
    def apply_overrides(ns):
        for arg, var, mofa in ((args.teacher_factor, "spatial_dopamine_factor_n", "spatial_mofa"),
                               (args.student_factor, "multiome_dopamine_factor_n", "multiome_mofa")):
            if arg is None:
                continue
            if arg not in ns[mofa].get_factors(df=True).columns:
                raise KeyError(f"{arg} is not a factor of {mofa}")
            print(f"[fig5] {var}: {ns[var]} -> {arg} (override)", flush=True)
            ns[var] = arg
        orient_factor(ns["spatial_mofa"], ns["spatial_dopamine_factor_n"], "msi_teacher")
        orient_factor(ns["multiome_mofa"], ns["multiome_dopamine_factor_n"], "msi_student")

    cells = FIG5_CELLS[:-1] + (FIG6_CELLS if args.fig6 else ()) + FIG5_CELLS[-1:]
    run_cells(cells, namespace, after={"": apply_overrides})

    print(f"[fig5] RUN_ID={namespace['RUN_ID']}; teacher factor "
          f"{namespace['spatial_dopamine_factor_n']}, student factor "
          f"{namespace['multiome_dopamine_factor_n']}", flush=True)
    for suffix in ("png", "svg"):
        namespace["FIG_dopamine_overview"](
            benchmark_metrics=args.benchmark_csv,
            output_file=str(args.out_dir / f"{args.name}.{suffix}"),
        )
    print(f"[fig5] wrote {args.out_dir / args.name}.png/.svg")
    if args.fig6:
        if args.gc_motifs is not None:
            use_gc_adjusted_motifs(namespace, args.gc_motifs)
        for label in ("D2MSN", "MXD"):
            repair_degenerate_gsea(namespace[f"{label.lower()}_multiome_gsea"], f"{label}_cCRE")
        fig6_board(namespace, args.out_dir / f"{args.fig6_name}.png")
        namespace["FIG_ccre_tf_overview"](
            output_file=str(args.out_dir / "sma_dopamine_ccre_tf_overview.svg"))


if __name__ == "__main__":
    sys.exit(main())
