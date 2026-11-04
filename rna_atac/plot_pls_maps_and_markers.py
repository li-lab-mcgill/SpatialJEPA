"""Supplementary Fig. 4 extensions: marker-gene weights of RNA PLS_0-2 in source and
target with the marker genes' expression maps, and the ATAC PLS_0-2 maps that main
Fig. 4c leaves out.

Reuses plot_transfer_panels.py as a library: its nine-component PLS fits (copied
from that run's cache, so nothing is refitted) and its sign handling (PLS_0 oriented
by myelin markers, other models aligned and refined across the four fits), so every
sign matches Fig. 4c,d. The gene sets are the six modules of Fig. 4d
(plot_pls_biological_annotation in analyze_rna_atac.py). Gene weights are the
models' y_weights_, as in Fig. 4d, scaled to unit SD across genes within each model
and component so source and target share one scale. The expression maps follow
Fig. 4e (plot_mobp_phactr1_expression): mean SCT-scaled RNA at each coordinate,
restricted to coordinates occupied in both source and target, Gaussian-smoothed at
sigma = 1 spot, one colour scale per gene. Mobp is left out because Fig. 4e shows it.

    python plot_pls_maps_and_markers.py [--output DIR]
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
from pathlib import Path
import argparse
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

import plot_transfer_panels as F4

REVISION = F4.DEFAULT_OUT
DEFAULT_OUT = F4.BASE / "outputs/rna_atac_figures/fig4_supp_pls_markers"
MODULES = {     # Fig. 4d modules, in its row order
    "Oligodendrocyte / myelin": ["Mobp", "Mal", "Mag", "Mbp", "Plp1", "Cnp"],
    "Striatal / MSN": ["Pde10a", "Rgs9", "Gng7", "Ppp1r1b", "Drd1", "Adora2a"],
    "Mural / smooth muscle": ["Cald1", "Myh11", "Mylk", "Acta2", "Tagln", "Rgs5", "Pdgfrb"],
    "Endothelial / BBB": ["Flt1", "Abcb1a", "Slco1c1", "Kdr", "Cldn5", "Pecam1"],
    "Astrocyte": ["Aqp4", "Slc1a3", "Aldoc", "Gfap", "S100b", "Glul"],
    "ECM / adhesion": ["Col1a1", "Col1a2", "Col3a1", "Col5a1", "Dcn", "Fn1"],
}
COLORS = {"Oligodendrocyte / myelin": "#d95f02", "Striatal / MSN": "#1b9e77",
          "Mural / smooth muscle": "#7570b3", "Endothelial / BBB": "#e7298a",
          "Astrocyte": "#66a61e", "ECM / adhesion": "#a6761d"}
N_SHOW = 3
MAP_GENES = ["Mal", "Mag", "Pde10a", "Rgs9", "Gng7"]   # myelin and striatal markers of Fig. 4d, minus Mobp (Fig. 4e)


def models(out):
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for f in (REVISION / "cache").iterdir():
        if not (cache / f.name).exists():
            shutil.copy2(f, cache / f.name)
    data = F4.load_metadata()
    funcs = F4.original_functions()
    return data, F4.fit_models(data, out, funcs), funcs


def gene_weights(model, var_names, funcs):
    W = funcs["pls_loadings_df"](model, var_names).iloc[:N_SHOW].T
    W = W.groupby(level=0, sort=False).mean()          # duplicate gene names, as in Fig. 4d
    return W / W.std(axis=0)


def marker_maps(data):
    """Fig. 4e's expression maps for MAP_GENES: coordinate means on the shared lattice, smoothed."""
    import scanpy as sc

    means = {}
    for domain in ("source", "target"):
        d = data[f"{domain}_rna"]
        X = np.asarray(sc.AnnData(X=F4.load_x(f"{domain}_rna"), var=d["var"].copy(),
                                  obs=d["obs"].copy())[:, MAP_GENES].X, dtype=float)
        tables = [F4.spot_means(d["xy"], X[:, k]).rename(columns={"value": g}) for k, g in enumerate(MAP_GENES)]
        merged = tables[0]
        for t in tables[1:]:
            merged = merged.merge(t, on=["x", "y"], how="inner")
        means[domain] = merged
    shared = means["source"].merge(means["target"], on=["x", "y"], how="inner", suffixes=("_source", "_target"))
    xy = shared[["x", "y"]].to_numpy()
    full_xy = np.asarray(data["source_rna"]["xy"], dtype=float)
    for g in MAP_GENES:
        for domain in ("source", "target"):
            shared[f"{g}_{domain}"] = F4.gaussian_smooth(shared[f"{g}_{domain}"].to_numpy(), xy, full_xy, sigma_spots=1)
    return shared


def draw(Ws, Wt, maps, out):
    shared = Ws.index.intersection(Wt.index)
    genes = [(m, g) for m, gs in MODULES.items() for g in gs if g in shared]
    missing = sorted({g for gs in MODULES.values() for g in gs} - set(shared))
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    W, H = 10.4, 3.85                                   # inches; panels placed in inches so the map grid
    fig = plt.figure(figsize=(W, H))                    # spans the heatmap's full height
    inch = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    side, top = 1.42, 3.35                              # map side; top edge shared by heatmap and maps
    # (a) per-gene weights of the Fig. 4d gene sets
    ax = inch(0.75, 0.62, 1.75, top - 0.62)
    M = np.column_stack([v for k in range(N_SHOW) for v in (Ws.loc[[g for _, g in genes], f"PLS_{k}"],
                                                           Wt.loc[[g for _, g in genes], f"PLS_{k}"])])
    lim = float(np.nanpercentile(np.abs(M), 98))
    art = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(genes)), [g for _, g in genes], fontsize=6.5, style="italic")
    for t, (m, _) in zip(ax.get_yticklabels(), genes):
        t.set_color(COLORS[m])
    ax.set_xticks(range(2 * N_SHOW), ["S", "T"] * N_SHOW, fontsize=7)
    for k in range(N_SHOW):
        ax.text(2 * k + 0.5, -1.2, f"PLS_{k}", ha="center", va="bottom", fontsize=8)
        if k:
            ax.axvline(2 * k - 0.5, color="k", lw=0.6)
    edges = np.cumsum([sum(1 for m2, _ in genes if m2 == m) for m in MODULES])[:-1]
    for e in edges:
        ax.axhline(e - 0.5, color="k", lw=0.6)
    ax.tick_params(length=0)
    cb = fig.colorbar(art, cax=inch(0.75, 0.3, 1.75, 0.09), orientation="horizontal")
    cb.set_label("gene weight (SD units)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.text(0.0, (top + 0.12) / H, "a", fontsize=13, fontweight="bold")
    # (b) expression maps of the marker genes, as Fig. 4e
    xy = maps[["x", "y"]].to_numpy()
    fig.text(2.68 / W, (top + 0.12) / H, "b", fontsize=13, fontweight="bold")
    module_of = {g: m for m, gs in MODULES.items() for g in gs}
    for c, g in enumerate(MAP_GENES):
        pooled = np.concatenate([maps[f"{g}_source"], maps[f"{g}_target"]])
        norm = Normalize(float(pooled.min()), float(pooled.max()))
        for r, domain in enumerate(("source", "target")):
            a = inch(3.0 + 1.48 * c, top - side - r * (side + 0.06), side, side)
            sc_art = F4.spatial(a, xy, maps[f"{g}_{domain}"].to_numpy(), norm=norm, size=2.8)
            if r == 0:
                a.set_title(g, fontsize=9, style="italic", color=COLORS[module_of[g]])
            if c == 0:
                a.set_ylabel("Source RNA" if domain == "source" else "Target RNA", fontsize=8.5)
        cb = fig.colorbar(sc_art, cax=inch(3.0 + 1.48 * c, 0.3, side, 0.09), orientation="horizontal")
        cb.ax.tick_params(labelsize=5.5)
    rows = []
    for k in range(N_SHOW):
        c = f"PLS_{k}"
        mg = [g for _, g in genes]
        x, y = Ws.loc[mg, c], Wt.loc[mg, c]
        rows.append(dict(component=c, n_set_genes=len(mg), r_set_genes=np.corrcoef(x, y)[0, 1],
                         r_all_genes=np.corrcoef(Ws.loc[shared, c], Wt.loc[shared, c])[0, 1],
                         sign_agreement=(np.sign(x) == np.sign(y)).mean()))
    tab = []
    for k in range(N_SHOW):
        c = f"PLS_{k}"
        for m, gl in MODULES.items():
            g = [h for h in gl if h in shared]
            tab.append(dict(component=c, module=m, n_genes=len(g), source_mean=Ws.loc[g, c].mean(),
                            target_mean=Wt.loc[g, c].mean()))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"S_Fig4_pls_marker_genes.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "pls_marker_agreement.csv", index=False)
    pd.DataFrame(tab).to_csv(out / "pls_module_mean_weights.csv", index=False)
    pd.concat({"source": Ws.loc[[g for _, g in genes]], "target": Wt.loc[[g for _, g in genes]]},
              axis=1).to_csv(out / "pls_marker_gene_weights.csv")
    maps.to_csv(out / "marker_gene_maps.csv.gz", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    if missing:
        print("module genes absent from the shared gene set:", ", ".join(missing))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    data, fits, funcs = models(a.output)
    Ws = gene_weights(fits["source_rna"], data["source_rna"]["var"].index, funcs)
    Wt = gene_weights(fits["target_rna"], data["target_rna"]["var"].index, funcs)
    draw(Ws, Wt, marker_maps(data), a.output)
    # ATAC PLS_0-2 maps: the separate panel the Fig. 4 revision already drew on the RNA maps' scale
    for ext in ("png", "svg", "pdf"):
        shutil.copy2(REVISION / f"supplementary_atac_pls.{ext}", a.output / f"S_Fig4_atac_pls_maps.{ext}")


if __name__ == "__main__":
    main()
