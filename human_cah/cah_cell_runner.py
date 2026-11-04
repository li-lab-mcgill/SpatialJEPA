try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Recreate the SEA-AD CaH transfer figure (manuscript Fig. 7) from one SpatialJEPA bundle.
#
# The three CaH scripts are interactive, cell-based and hard-code their output root
# (outputs/sea_ad_lipid_gps_minimal/cah, which backs the manuscript figure). This driver
# executes only the cells the figure needs, in one namespace per stage, with the output
# root redirected by checked in-memory substitutions; the scripts on disk are untouched.
#
#   impute  snrna_msi_impute.py: per-donor student/teacher embeddings and
#           PLS-imputed MSI for every CaH snRNA donor, the aligned RNA/MSI MuData, and the
#           Xenium hand-off for the figure section (panel e). The RNA UMAP is copied from
#           the manuscript MuData (same nuclei; the UMAP is RNA-only and model-free) so
#           panel b keeps its layout and the 714k-cell UMAP is not recomputed.
#   mofa    snrna_mofa.py's model: 5 factors, seed 42, 100 iterations.
#   figure  downstream_analyses.py figure functions: panels a-f
#           (FIG_cah_transfer_main) plus the bivariate-Moran row (g) of the manuscript.
#
# Factor choice: MOFA orders factors by variance explained, so the dopamine axis need
# not be Factor1 for a new model. `--factor auto` takes the factor whose scores track
# the imputed dopamine best (|Spearman r| over nuclei) and orients it so msi:Dopamine
# loads positively; every factor is reported.
#
# Envs: impute/figure/mofa run in nichecompass_liana (the hand-off worker itself runs in
# the spatialMETA env as a subprocess).

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

LIPID_GP = Path(__file__).resolve().parent
IMPUTE_SCRIPT = LIPID_GP / "snrna_msi_impute.py"
MOFA_SCRIPT = LIPID_GP / "snrna_mofa.py"
DOWNSTREAM_SCRIPT = LIPID_GP / "downstream_analyses.py"
MANUSCRIPT_CAH = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/sea_ad_lipid_gps_minimal/cah")
XENIUM_SECTION = "H21.33.002.CX18.CaH.02.01"
XENIUM_SECTION_KEY = "Specimen_ID"


def parse_args():
    parser = argparse.ArgumentParser(description="Recreate the SEA-AD CaH transfer figure.",
                                     allow_abbrev=False)
    parser.add_argument("--stage", choices=("impute", "mofa", "figure"), required=True)
    parser.add_argument("--model-dir", type=Path, help="SpatialJEPA bundle (impute)")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="output root; results go to <out-root>/cah")
    parser.add_argument("--factor", default="auto", help="MOFA factor for the figure, or 'auto'")
    parser.add_argument("--name", default="sea_ad_cah")
    return parser.parse_args()


def split_cells(path: Path):
    lines = path.read_text().splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.match(r"^#\s?%%", line)] + [len(lines)]
    for a, b in zip(starts, starts[1:]):
        header = re.sub(r"^#\s?%%\s*", "", lines[a]).strip()
        yield header, "\n" * a + "".join(lines[a:b])


def substitute(code: str, old: str, new: str) -> str:
    if code.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}, found {code.count(old)}")
    return code.replace(old, new)


def exec_definitions(code: str, namespace, script: Path) -> None:
    """Run a cell's imports, defs and constants one statement at a time, dropping its
    top-level calls, try- and if-blocks. A def whose default arguments need names from
    the skipped (AD-association) cells is left undefined and logged."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.Expr, ast.Try, ast.If)):
            continue
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), namespace)
        except NameError as error:
            name = getattr(node, "name", type(node).__name__)
            print(f"[fig7]   skipped {name} ({error})", flush=True)


def run_cells(script: Path, cells, namespace, *, patches=None, defs_only=()):
    """Execute the listed cells (header prefixes, file order) of `script`."""
    wanted = list(cells)
    patches = patches or {}
    for header, code in split_cells(script):
        if not wanted or not header.startswith(wanted[0]) or (wanted[0] == "" and header):
            continue
        key = wanted.pop(0)
        for old, new in patches.get(key, ()):
            code = substitute(code, old, new)
        print(f"[fig7] {script.name}: running cell {header or '(unnamed)'}", flush=True)
        if key in defs_only:
            exec_definitions(code, namespace, script)
        else:
            exec(compile(code, str(script), "exec"), namespace)
    if wanted:
        raise RuntimeError(f"{script.name}: cells not found: {wanted}")


# --------------------------------------------------------------------------- impute
def manuscript_umap(obs_names):
    import h5py
    import numpy as np
    import pandas as pd

    with h5py.File(MANUSCRIPT_CAH / "snrna_spatialmeta_aligned.h5mu", "r") as f:
        index = f["mod/rna/obs"].attrs.get("_index", "_index")
        names = f[f"mod/rna/obs/{index}"][:].astype(str)
        umap = f["mod/rna/obsm/X_umap"][:]
    positions = pd.Index(names).get_indexer(pd.Index(obs_names).astype(str))
    if (positions < 0).any():
        raise ValueError(f"{(positions < 0).sum()} nuclei missing from the manuscript UMAP")
    return np.asarray(umap[positions], dtype=np.float32)


def write_aligned_with_manuscript_umap(ns, donor_ids, *, output_path):
    """0__'s write_aligned_snrna_mudata, with the RNA UMAP taken from the manuscript."""
    import anndata as ad
    import mudata as md

    rna_parts, msi_parts = [], []
    for donor_id in donor_ids:
        rna_path, msi_path = ns["spatialmeta_aligned_output_paths"](donor_id)
        rna_parts.append(rna_path)
        msi_parts.append(msi_path)
    concat_dir = output_path.parent / ".snrna_spatialmeta_concat"
    concat_dir.mkdir(parents=True, exist_ok=True)
    rna_path, msi_path = concat_dir / "aligned_rna.h5ad", concat_dir / "imputed_msi.h5ad"
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    for path in (rna_path, msi_path, tmp_path):
        path.unlink(missing_ok=True)
    ad.experimental.concat_on_disk(rna_parts, rna_path, axis=0, join="inner")
    ad.experimental.concat_on_disk(msi_parts, msi_path, axis=0, join="inner")
    rna = ad.read_h5ad(rna_path)
    msi = ad.read_h5ad(msi_path, backed="r")
    try:
        if not rna.obs_names.equals(msi.obs_names):
            raise ValueError("Combined aligned RNA and imputed MSI observation order differs.")
        rna.obsm["X_umap"] = manuscript_umap(rna.obs_names)
        mudata = md.MuData({"rna": rna, "msi": msi})
        mudata.uns["spatialmeta_device"] = ns["SPATIALMETA_DEVICE"]
        mudata.uns["spatialmeta_batch_size"] = ns["SPATIALMETA_BATCH_SIZE"]
        mudata.uns["donor_ids"] = donor_ids
        mudata.uns["spatialmeta_model_dir"] = str(ns["SPATIALMETA_MODEL_DIR"])
        mudata.write_h5mu(tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        msi.file.close()
        tmp_path.unlink(missing_ok=True)
    rna_path.unlink(missing_ok=True)
    msi_path.unlink(missing_ok=True)
    print(f"[fig7] wrote aligned MuData {output_path}", flush=True)
    return output_path


def stage_impute(args):
    if args.model_dir is None:
        raise SystemExit("--model-dir is required for --stage impute")
    os.environ.update(
        SEA_AD_ACTIVE_REGION="cah",
        SPATIALMETA_MODEL_DIR=str(args.model_dir),
        SPATIALMETA_DEVICE=os.environ.get("SPATIALMETA_DEVICE", "cuda:0"),
        SEA_AD_FIG7_OUTPUT_ROOT=str(args.out_root),
    )
    ns = {"__name__": "fig7_impute", "__file__": str(IMPUTE_SCRIPT)}
    patches = {"configuration": [(
        'OUTPUT_ROOT = Path(__import__("os").environ["SPATIALJEPA_OUT"]) / "sea_ad_lipid_gps_minimal"',
        'OUTPUT_ROOT = Path(os.environ["SEA_AD_FIG7_OUTPUT_ROOT"])',
    )]}
    run_cells(IMPUTE_SCRIPT, ("", "configuration", "dataset helpers", "spatialMETA hand-off helpers",
                              "load SEA-AD modalities"), ns, patches=patches)
    assert Path(ns["REGION_OUTPUT_ROOT"]) == args.out_root / "cah", ns["REGION_OUTPUT_ROOT"]
    aligned = Path(ns["SPATIALMETA_ALIGNED_MUDATA_PATH"])
    if aligned.exists():
        print(f"[fig7] aligned MuData exists, skipping snRNA inference: {aligned}")
    else:
        ns["write_aligned_snrna_mudata"] = (
            lambda donor_ids, *, output_path: write_aligned_with_manuscript_umap(
                ns, donor_ids, output_path=output_path))
        run_cells(IMPUTE_SCRIPT, ("Run GPU inference one donor at a time",), ns)

    # Xenium hand-off for the figure section (panel e reads its student_msi)
    spatial = ns["load_spatial_transcriptomics"](ns["fs"])
    section = spatial[spatial.obs[XENIUM_SECTION_KEY].astype(str) == XENIUM_SECTION].to_memory()
    donors = section.obs[ns["DONOR_KEY"]].astype(str).unique()
    if len(donors) != 1:
        raise ValueError(f"{XENIUM_SECTION}: expected one donor, found {list(donors)}")
    print(f"[fig7] Xenium hand-off: {XENIUM_SECTION} ({section.n_obs:,} cells, donor {donors[0]})")
    ns["ensure_spatialmeta_embeddings"]([
        {"kind": "merfish", "adata": section, "donor_id": donors[0], "section_id": XENIUM_SECTION},
    ])


# --------------------------------------------------------------------------- mofa
def stage_mofa(args):
    import muon as mu

    cah = args.out_root / "cah"
    ns = {"__name__": "fig7_mofa", "__file__": str(MOFA_SCRIPT)}
    patches = {"": [(
        'CAH_MOFA_MODEL_PATH = Path(__import__("os").environ["SPATIALJEPA_HOME"], "outputs/sea_ad_lipid_gps_minimal/cah/snrna_mofa.h5mu")',
        f'CAH_MOFA_MODEL_PATH = Path({str(cah / "snrna_mofa.h5mu")!r})',
    )]}
    if os.environ.get("CAH_MOFA_GPU_MODE") == "0":     # CPU fallback when the GPU is too small
        patches[""].append(("CAH_MOFA_GPU_MODE = True", "CAH_MOFA_GPU_MODE = False"))
    run_cells(MOFA_SCRIPT, ("",), ns, patches=patches)
    if Path(ns["CAH_MOFA_MODEL_PATH"]).exists():
        print(f"[fig7] MOFA model exists: {ns['CAH_MOFA_MODEL_PATH']}")
        return
    mudata = mu.read_h5mu(cah / "snrna_spatialmeta_aligned.h5mu")
    ns["train_cah_mofa_model"](mudata)


# --------------------------------------------------------------------------- figure
def orient_factor(model_path, factor_index, anchor_view="msi", anchor="msi:Dopamine"):
    """Sign-flip one factor (Z and W jointly) so `anchor` loads positively (relabelling only)."""
    import h5py

    with h5py.File(model_path, "r") as f:   # only open for writing when a flip is needed
        names = [x.decode() for x in f[f"features/{anchor_view}"][:]]
        if f[f"expectations/W/{anchor_view}"][factor_index, names.index(anchor)] >= 0:
            return False
    with h5py.File(model_path, "r+") as f:
        for view in [v.decode() for v in f["views/views"][:]]:
            w = f[f"expectations/W/{view}"]
            w[factor_index, :] = -w[factor_index, :]
        for group in [g.decode() for g in f["groups/groups"][:]]:
            z = f[f"expectations/Z/{group}"]
            z[factor_index, :] = -z[factor_index, :]
    return True


def choose_factor(cah, requested):
    """Report every factor's link to imputed dopamine; return the one to plot (0-based)."""
    import h5py
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr

    model_path = cah / "snrna_mofa.h5mu"
    with h5py.File(model_path, "r") as f:
        group = list(f["expectations/Z"].keys())[0]
        Z = f[f"expectations/Z/{group}"][:]
        names = [x.decode() for x in f["features/msi"][:]]
        W = f["expectations/W/msi"][:]
        cells = f[f"samples/{group}"][:].astype(str)
    with h5py.File(cah / "snrna_spatialmeta_aligned.h5mu", "r") as f:
        index = f["mod/msi/obs"].attrs.get("_index", "_index")
        msi_cells = f[f"mod/msi/obs/{index}"][:].astype(str)
        msi_vars = f["mod/msi/var/" + f["mod/msi/var"].attrs.get("_index", "_index")][:].astype(str)
        X = f["mod/msi/X"]
        dopamine = X[:, list(msi_vars).index("msi:Dopamine")]
    dopamine = pd.Series(dopamine, index=msi_cells).reindex(cells).to_numpy()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(cells), size=min(100_000, len(cells)), replace=False)
    da = names.index("msi:Dopamine")
    rows = []
    for k in range(Z.shape[0]):
        w = W[k] * (np.sign(W[k, da]) or 1.0)
        rows.append(dict(factor=f"Factor{k + 1}",
                         r_imputed_dopamine=abs(spearmanr(Z[k, pick], dopamine[pick]).correlation),
                         dopamine_weight=float(abs(W[k, da])),
                         dopamine_rank=int((np.abs(W[k]) > abs(W[k, da])).sum() + 1),
                         top_msi=", ".join(n.replace("msi:", "") for n in np.array(names)[np.argsort(-w)[:5]])))
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    if requested == "auto":
        k = int(table["r_imputed_dopamine"].to_numpy().argmax())
    else:
        k = int(requested.replace("Factor", "")) - 1
    if orient_factor(model_path, k):
        print(f"[fig7] flipped the sign of Factor{k + 1} so msi:Dopamine loads positively")
    table.to_csv(cah / "figures" / "factor_dopamine_link.csv", index=False)
    return k


def draw_moran_row(ax, table, x):
    """Panel g: bivariate Moran's I between the three Xenium maps, smoothed (unsmoothed).
    `x` gives each map's horizontal position (its panel centre) in `ax` data units."""
    pairs = [("oligo_density", "imputed_dopamine"), ("oligo_density", "projected_factor1"),
             ("imputed_dopamine", "projected_factor1")]
    value = {(r.variant, r.field_a, r.field_b): r.bivariate_I for r in table.itertuples()}
    text_x = x["projected_factor1"] + 0.6 * (x["projected_factor1"] - x["imputed_dopamine"])
    for row, (a, b) in enumerate(pairs):
        y = -row
        ax.plot([x[a], x[b]], [y, y], color="k", lw=1.5, zorder=1)
        ax.scatter([x[a], x[b]], [y, y], s=160, color="#AAAAAA", edgecolor="#333333", lw=1.5, zorder=2)
        ax.text(text_x, y, f"{value['as plotted', a, b]:+.2f} ({value['unsmoothed', a, b]:+.2f})",
                va="center", ha="center", fontsize=13)
    ax.text(text_x, 0.85, "Moran's $I$", ha="center", va="bottom", fontsize=14)
    ax.text(text_x, 0.55, "smoothed (unsmoothed)", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(-2.5, 1.3)
    ax.set_axis_off()


def fig7_board(ns, section, factor, moran, output_file, smooth=30):
    """FIG_cah_transfer_main's layout (a-f) with the manuscript's Moran row (g) added."""
    import matplotlib.pyplot as plt

    with plt.rc_context(ns["FIG_RC"]):
        fig = plt.figure(figsize=(14, 14.5), constrained_layout=True)
        outer = fig.add_gridspec(3, 1, height_ratios=[1.8, 1.0, 0.45])
        top = outer[0].subgridspec(1, 2, width_ratios=[0.26, 0.74], wspace=0.08)
        upper_right = top[1].subgridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.05, wspace=0.02)
        axes = {"a": fig.add_subplot(top[0]), "b": fig.add_subplot(upper_right[0, 0])}
        ns["plot_factor1_multiomic_weights"](axes["a"], ns["mofa"], factor=factor)
        ns["plot_factor1_umaps"](ns["mofa_adata"], axes["b"], fig.add_subplot(upper_right[0, 1]),
                                 factor=factor)
        axes["c"] = fig.add_subplot(upper_right[1, :])
        ns["plot_factor1_subclass_violin"](axes["c"], ns["mofa_adata"], factor=factor)
        bottom = outer[1].subgridspec(1, 3, wspace=0.08)
        number = factor.replace("Factor", "")
        for key, column, title, kwargs in (
            ("d", "dens_oligo", "local oligodendrocyte density",
             dict(cmap="magma", diverging=False, smooth=0, size=10.0)),
            ("e", "imputed_dopamine", f"imputed dopamine ({smooth}-NN mean)",
             dict(smooth=smooth, size=8.0)),
            ("f", "mofa_factor1", f"projected Factor {number} ({smooth}-NN mean)",
             dict(smooth=smooth, size=8.0)),
        ):
            axes[key] = fig.add_subplot(bottom["def".index(key)])
            ns["plot_spatial_continuous"](axes[key], section, column, title, **kwargs)
        axes["g"] = fig.add_subplot(outer[2])
        # place each dumbbell end under the centre of its map: with the layout fixed,
        # the g axis spans figure fractions [x0, x1], so data x == figure fraction
        fig.canvas.draw()
        fig.set_layout_engine("none")
        g_box = axes["g"].get_position()
        axes["g"].set_xlim(g_box.x0, g_box.x1)
        centre = {k: (axes[k].get_position().x0 + axes[k].get_position().x1) / 2 for k in "def"}
        draw_moran_row(axes["g"], moran, {"oligo_density": centre["d"],
                                          "imputed_dopamine": centre["e"],
                                          "projected_factor1": centre["f"]})
        for key, ax in axes.items():
            ax.text(-0.02, 1.02, key, transform=ax.transAxes, fontsize=22, fontweight="bold",
                    ha="right", va="bottom")
        for suffix in ("png", "svg"):
            fig.savefig(output_file.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[fig7] wrote {output_file.with_suffix('.png')} (+ .svg)", flush=True)


def stage_figure(args):
    cah = args.out_root / "cah"
    (cah / "figures").mkdir(parents=True, exist_ok=True)
    k = choose_factor(cah, args.factor)
    factor = f"Factor{k + 1}"
    print(f"[fig7] plotting {factor}", flush=True)

    ns = {"__name__": "fig7_figure", "__file__": str(DOWNSTREAM_SCRIPT)}
    root_patch = [(
        'CAH_OUTPUT_ROOT = Path(\n    __import__("os").environ["SPATIALJEPA_HOME"], "outputs/sea_ad_lipid_gps_minimal/cah"\n)',
        f'CAH_OUTPUT_ROOT = Path({str(cah)!r})',
    )]
    run_cells(DOWNSTREAM_SCRIPT, ("load data", "load data and MOFA model"), ns,
              patches={"load data and MOFA model": root_patch})
    import scanpy as sc

    # mofa_adata as in the downstream script (its cell also draws an interactive UMAP);
    # needed before the figure cells, whose functions take it as a default argument
    ns["mofa_adata"] = sc.AnnData(
        ns["mofa"].get_factors(),
        obs=ns["scrna_msi_mudata"].mod["msi"].obs,
        obsm=ns["scrna_msi_mudata"].mod["rna"].obsm,
        var=ns["mofa"].get_weights(df=True).columns.to_frame(),
    )
    # default argument of the (unused) AD-association panels, set in a skipped cell
    ns["PRIMARY"] = "Overall AD neuropathological Change"
    figure_cells = ("Reusable Factor 1 multi-panel figure", "Transfer-characterisation figure",
                    "spatial cross-correlation between the three panels")
    run_cells(DOWNSTREAM_SCRIPT, figure_cells, ns, defs_only=figure_cells)
    section = ns["load_xenium_factor_section"]()          # projected onto Factor1 by default
    ns["project_spatial_onto_factor"](section, mofa_model=ns["mofa"], factor_index=k,
                                      key="mofa_factor1")
    section = ns["attach_xenium_imputed_msi"](section)
    section.obs["dens_oligo"] = ns["local_category_density"](section, "Oligodendrocyte_Subclass")
    moran, _ = ns["cah_xenium_spatial_crosscorrelation"](
        section, output_file=cah / "figures" / "cah_xenium_bivariate_moran.csv")
    fig7_board(ns, section, factor, moran, cah / "figures" / args.name)
    ns["FIG_cah_transfer_main"](xenium_section=section, factor=factor,
                                output_file=cah / "figures" / "cah_transfer_main.svg")


def main():
    args = parse_args()
    {"impute": stage_impute, "mofa": stage_mofa, "figure": stage_figure}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
