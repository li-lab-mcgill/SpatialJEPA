"""Regenerate screenshot panels a/c/d/e and separate ATAC/Phactr1 RNA maps.

The screenshot lettering differs from main.tex and older script names.
Only named function definitions are loaded from analyze_rna_atac.py: its
training, inference and manuscript-writing top-level code is never executed.
PLS fitting, sign alignment, enrichment, and Scanpy marker scores follow that
source. Outputs and cached fits are local to --output.
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
import ast
import hashlib
import json
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scanpy as sc
import sklearn
from anndata.io import read_elem
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

BASE = Path(os.environ['SPATIALJEPA_HOME'])
DATA = BASE / "data/aligned_data"
ORIGINAL = Path(os.environ["SPATIALJEPA_RNA_ATAC"]) / "analyze_rna_atac.py"
LABELS = BASE / "outputs/rna_atac_transfer_clusters/shared_params"
DEFAULT_OUT = BASE / "outputs/rna_atac_figures/fig4_revision_20261020"
DOMAINS = ("source_rna", "target_rna", "source_atac", "target_atac")
MYELIN = ["Mbp", "Plp1", "Mobp", "Mal", "Cnp", "Mog", "Mag", "Cldn11", "Ermn", "Sox10"]
# Panel f's original shortlist_peaks_two_row_smooth1_no_mal uses 10.5 pt
# titles, 6 pt colorbar ticks and a 5.1 x 5.25 inch canvas.
RC = {
    "font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
    "axes.linewidth": 0.5, "svg.fonttype": "none", "pdf.fonttype": 42,
    "legend.frameon": False,
}


def original_functions():
    """Load the existing numerical routines without running the pipeline."""
    names = {
        "_apply_pls_component_signs", "orient_pls_components_by_markers",
        "align_pls_component_signs", "refine_pls_signs_across_models",
        "pls_loadings_df", "plot_pls_biological_annotation",
    }
    tree = ast.parse(ORIGINAL.read_text())
    selected = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in names:
            continue
        if node.name == "plot_pls_biological_annotation":
            # Keep all original numerical work (including the full five-pair
            # multiple-testing family), and replace only the plotting tail.
            stop = next(i for i, stmt in enumerate(node.body)
                        if isinstance(stmt, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "module_ids"
                                for t in stmt.targets))
            node.body = node.body[:stop] + ast.parse(
                "return dict(scores=heat_scores, qvalues=heat_qvalues, "
                "labels=module_display, matches=match_records)"
            ).body
        selected.append(node)
    assert len(selected) == len(names)
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(ORIGINAL), "exec"), namespace)
    return namespace


def load_metadata():
    result = {}
    for name in DOMAINS:
        with h5py.File(DATA / f"{name}_aligned_with_latents.h5ad") as f:
            result[name] = {
                "obs": read_elem(f["obs"]), "var": read_elem(f["var"]),
                "xy": f["obsm/spatial"][:], "umap": f["obsm/X_umap"][:],
                "Z": f["obsm/MultiGATE_source_aligned"][:],
            }
    return result


def load_x(name):
    with h5py.File(DATA / f"{name}_aligned_with_latents.h5ad") as f:
        x = read_elem(f["X"])
    return x.toarray() if sp.issparse(x) else x


def provenance():
    paths = [ORIGINAL] + [DATA / f"{n}_aligned_with_latents.h5ad" for n in DOMAINS]
    return {
        "original_script_sha256": hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        "inputs": {str(p): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                   for p in paths},
        "n_components": 9, "display_components": [0, 1, 2],
        "versions": {"scanpy": sc.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "enrichment_pairs_tested": 5, "enrichment_permutations": 2000,
        "enrichment_seed_source": 42, "enrichment_seed_target": 43,
        "panel_e_title_font_pt": 10.5, "panel_e_colorbar_tick_font_pt": 6,
    }


def fit_models(data, out, funcs):
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    signature = provenance()
    signature_path = cache / "inputs.json"
    if signature_path.exists():
        previous = json.loads(signature_path.read_text())
        keys = ["inputs", "original_script_sha256", "n_components"]
        if "versions" in previous:
            keys.append("versions")
        if any(previous[k] != signature[k] for k in keys):
            raise RuntimeError("Inputs changed; use a fresh output directory for new fits.")
    signature_path.write_text(json.dumps(signature, indent=2) + "\n")
    models = {}
    for name in DOMAINS:
        path = cache / f"{name}_pls9.pkl"
        if path.exists():
            with path.open("rb") as f:
                models[name] = pickle.load(f)
            print("Loaded fit", name, flush=True)
        else:
            print("Fitting 9-component PLS:", name, flush=True)
            x = load_x(name)
            with threadpool_limits(limits=4):
                model = PLSRegression(n_components=9).fit(data[name]["Z"], x)
            with path.open("wb") as f:
                pickle.dump(model, f)
            models[name] = model
            del x
            print("Saved fit", name, flush=True)
    source = models["source_rna"]
    funcs["orient_pls_components_by_markers"](
        source, data["source_rna"]["Z"], load_x("source_rna"),
        data["source_rna"]["var"].index, MYELIN, components=(0,))
    common_z = np.concatenate([data[n]["Z"] for n in DOMAINS])
    for name in DOMAINS[1:]:
        funcs["align_pls_component_signs"](models[name], source, common_z, label=name)
    funcs["refine_pls_signs_across_models"](
        [(n, models[n]) for n in ("source_rna", "source_atac", "target_rna", "target_atac")],
        common_z, n_show=5)
    for name in DOMAINS:
        data[name]["scores"] = models[name].transform(data[name]["Z"])
    return models


def save(fig, stem):
    # Fixed canvas dimensions: no nonuniform scaling or automatic tight cropping.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.get_tightbbox(renderer)
    width, height = fig.get_size_inches()
    if box.x0 < -.01 or box.y0 < -.01 or box.x1 > width+.01 or box.y1 > height+.01:
        raise ValueError(f"Artists outside {stem.name} canvas: {box.bounds}")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix("." + ext), dpi=300, facecolor="white")
    plt.close(fig)
    print("Saved", stem, flush=True)


def spatial(ax, xy, values, norm=None, cmap="viridis", size=6):
    order = np.argsort(values, kind="stable")
    art = ax.scatter(xy[order, 0], xy[order, 1], c=values[order], s=size,
                     norm=norm, cmap=cmap, linewidths=0, rasterized=True)
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_color("0.5")
        spine.set_linewidth(0.5)
    return art


def plot_d(data, models, funcs, out):
    frames = {name: funcs["pls_loadings_df"](models[name], data[name]["var"].index)
              for name in ("source_rna", "target_rna")}
    results = funcs["plot_pls_biological_annotation"](
        None, None, models["source_rna"], models["target_rna"],
        frames["source_rna"], frames["target_rna"], n_pairs=5)
    scores, qvalues = results["scores"], results["qvalues"]
    scores.to_csv(out / "panel_d_enrichment_all5.csv")
    qvalues.to_csv(out / "panel_d_qvalues_all5.csv")
    # Keep the original five-component color normalization and FDR correction
    # when displaying only the first three components from the screenshot.
    vmax = max(float(np.nanpercentile(np.abs(scores.to_numpy()), 95)), 1)
    values = scores.iloc[:, :6].to_numpy()
    fig = plt.figure(figsize=(7.9, 2.6))
    ax = fig.add_axes([0.235, 0.25, 0.73, 0.63])
    art = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                    extent=(0, 6, 6, 0), interpolation="nearest")
    ax.set_xticks(np.arange(6) + 0.5, ["Source", "Target"] * 3,
                  rotation=90, fontsize=9)
    ax.set_yticks(np.arange(6) + 0.5,
                  [results["labels"][n] for n in scores.index], fontsize=8)
    ax.tick_params(length=0)
    for j in range(3):
        ax.text(2*j+1, 1.025, f"PLS_{j}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10.5)
    for j in (2, 4):
        ax.axvline(j, color="black", lw=0.6)
    for i, j in np.argwhere(qvalues.iloc[:, :6].to_numpy() < 0.05):
        ax.text(j+0.5, i+0.5, "*", ha="center", va="center", fontsize=14,
                color="white" if abs(values[i, j]) > .65*vmax else "black")
    cb = fig.colorbar(art, cax=fig.add_axes([0.03, 0.13, 0.14, 0.035]),
                      orientation="horizontal")
    cb.ax.set_title("Signed loading\nenrichment", fontsize=8, loc="left", pad=3)
    cb.ax.tick_params(labelsize=7)
    save(fig, out / "panel_d_loading_enrichment_wide")
    return results


def plot_c(data, out):
    # Preserve the original five-component color normalization when displaying
    # three columns, shared by both main and supplementary modalities.
    lim = float(np.percentile(np.abs(np.concatenate(
        [data[n]["scores"][:, :5].ravel() for n in DOMAINS])), 99))
    norm = Normalize(-lim, lim)
    for modality, stem in (("rna", "panel_c_rna_pls"),
                           ("atac", "supplementary_atac_pls")):
        fig, axes = plt.subplots(2, 3, figsize=(7.9, 5.25))
        fig.subplots_adjust(left=.07, right=.89, bottom=.025, top=.94,
                            wspace=.045, hspace=.04)
        for row, domain in enumerate(("source", "target")):
            d = data[f"{domain}_{modality}"]
            for col in range(3):
                art = spatial(axes[row, col], d["xy"], d["scores"][:, col],
                              norm=norm, size=6)
                if row == 0:
                    axes[row, col].set_title(f"PLS_{col}")
                if col == 0:
                    axes[row, col].set_ylabel(domain.title(), labelpad=5)
        cb = fig.colorbar(art, cax=fig.add_axes([.915, .27, .015, .45]),
                          orientation="vertical")
        cb.set_label("PLS score", fontsize=10.5)
        cb.ax.tick_params(labelsize=8)
        save(fig, out / stem)
    return lim


def plot_a(data, out):
    from plot_transfer_clusters import plot_a as draw_a
    from transfer_target_leiden import match_clusters
    s = [data["source_rna"], data["source_atac"]]
    t = [data["target_rna"], data["target_atac"]]
    source_names = np.concatenate([d["obs"].index.to_numpy()+suffix
                                   for d, suffix in zip(s, ("_rna", "_atac"))])
    mapping = np.concatenate([d["obs"].source_obs_names.astype(str).to_numpy() for d in t])
    positions = pd.Index(source_names).get_indexer(mapping)
    assert np.all(positions >= 0)
    labels = [pd.read_csv(LABELS / f"shared_k150_r0p9_{domain}_labels.csv",
                          dtype=str)["leiden"].to_numpy()
              for domain in ("source", "target")]
    _, pairs = match_clusters(labels[0][positions], labels[1])
    labels[1] = np.array([pairs[c] for c in labels[1]])
    domains = [(lab, np.concatenate([d["xy"] for d in ds]),
                np.concatenate([d["umap"] for d in ds]), title, "")
               for lab, ds, title in zip(labels, (s, t), ("Source", "Target"))]
    draw_a(domains, sorted(np.unique(labels[0]), key=int),
           out / "panel_a_cluster_spatial_umap", figsize=(5.1, 5.25),
           rc={**RC, "axes.labelsize": 7, "legend.fontsize": 6})


def plot_e_and_phactr1(data, out):
    genesets = [["Pde10a", "Rgs9", "Gng7"], ["Mobp", "Mal"]]
    scores, expression = {}, {}
    for domain in ("source", "target"):
        name = f"{domain}_rna"
        d = data[name]
        a = sc.AnnData(X=load_x(name), var=d["var"].copy(), obs=d["obs"].copy())
        # Identical Scanpy defaults and gene pool to the original marker panel.
        for genes in genesets:
            sc.tl.score_genes(a, gene_list=genes, score_name=",".join(genes),
                              random_state=0, use_raw=False)
        scores[domain] = a.obs[[",".join(g) for g in genesets]].to_numpy()
        expression[domain] = np.asarray(a[:, ["Phactr1"]].X).ravel()
        pd.DataFrame(scores[domain], index=a.obs_names,
                     columns=[",".join(g) for g in genesets]).to_csv(
                         out / f"panel_e_{domain}_scores.csv.gz")
    fig, axes = plt.subplots(2, 2, figsize=(5.1, 5.25), layout="constrained")
    for row, domain in enumerate(("source", "target")):
        for col, genes in enumerate(genesets):
            art = spatial(axes[row, col], data[f"{domain}_rna"]["xy"],
                          scores[domain][:, col])
            if row == 0:
                axes[row, col].set_title(",".join(genes), fontsize=10.5)
            if col == 0:
                axes[row, col].set_ylabel(domain.title(), fontsize=10.5)
            fig.colorbar(art, ax=axes[row, col], fraction=.035, pad=.02
                         ).ax.tick_params(labelsize=6)
    save(fig, out / "panel_e_marker_scores")
    # Exploratory expression maps: no smoothing, no percentile clipping, and
    # a common scale; target nuclei at identical mapped positions are averaged
    # explicitly instead of letting overplotting select the largest value.
    means = {}
    for domain in ("source", "target"):
        xy = data[f"{domain}_rna"]["xy"]
        table = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                              "Phactr1": expression[domain]})
        table.to_csv(out / f"Phactr1_{domain}_rna_profiles.csv.gz", index=False)
        means[domain] = table.groupby(["x", "y"], sort=True).Phactr1.agg(
            ["mean", "count"]).reset_index()
        means[domain].to_csv(out / f"Phactr1_{domain}_rna_spot_means.csv", index=False)
    pooled = np.concatenate([means[d]["mean"] for d in means])
    norm = Normalize(pooled.min(), pooled.max())
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.65))
    fig.subplots_adjust(left=.025, right=.89, top=.84, bottom=.035, wspace=.05)
    for ax, domain in zip(axes, ("source", "target")):
        table = means[domain]
        art = spatial(ax, table[["x", "y"]].to_numpy(), table["mean"].to_numpy(),
                      norm=norm, size=10)
        ax.set_title("Source" if domain == "source" else "Target (mapped)")
    fig.suptitle("Phactr1 RNA expression", fontsize=12, y=.98)
    cb = fig.colorbar(art, cax=fig.add_axes([.915, .19, .018, .5]))
    cb.set_label("Mean SCT-scaled RNA expression", fontsize=9)
    save(fig, out / "exploratory_Phactr1_rna_source_target")
    expression_panel = plot_mobp_phactr1_expression(data, out)
    return {"scale": [float(norm.vmin), float(norm.vmax)],
            "mobp_phactr1_panel": expression_panel,
            "smoothing": "none", "clipping": "none",
            "expression": "RNA AnnData.X (SCT-scaled expression, may be negative)",
            "aggregation": "arithmetic mean of RNA profiles at each occupied coordinate",
            "counts": {d: {"profiles": len(expression[d]), "occupied_spots": len(means[d])}
                       for d in means}}


def spot_means(xy, values):
    table = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "value": values})
    return table.groupby(["x", "y"], sort=True)["value"].mean().reset_index()


def gaussian_smooth(values, xy, full_xy, sigma_spots=1.0):
    """Same spot-spacing Gaussian used for the linked-peak maps."""
    spacing = float(np.median(cKDTree(full_xy).query(full_xy, k=2)[0][:, 1]))
    sigma = sigma_spots * spacing
    tree = cKDTree(xy)
    distances = tree.sparse_distance_matrix(
        tree, 3 * sigma, output_type="coo_matrix"
    )
    graph = sp.csr_matrix(
        (
            np.exp(-0.5 * (distances.data / sigma) ** 2),
            (distances.row, distances.col),
        ),
        shape=(len(xy), len(xy)),
    )
    weights = sp.diags(1 / np.asarray(graph.sum(axis=1)).ravel()) @ graph
    return np.asarray(weights @ values).ravel()


def plot_mobp_phactr1_expression(data, out):
    """Mobp then Phactr1, with Source/Target RNA as row labels.

    Expression is the mean SCT-scaled RNA at each shared coordinate, then
    Gaussian-smoothed at σ = 1 spot, as for the linked-peak maps. Each gene
    keeps one source–target color scale.
    """
    genes = ["Mobp", "Phactr1"]
    means = {}
    for domain in ("source", "target"):
        name = f"{domain}_rna"
        d = data[name]
        values = np.asarray(sc.AnnData(
            X=load_x(name), var=d["var"].copy(), obs=d["obs"].copy()
        )[:, genes].X, dtype=float)
        domain_means = []
        for gene_column, gene in enumerate(genes):
            domain_means.append(spot_means(d["xy"], values[:, gene_column]).rename(
                columns={"value": gene}))
        merged = domain_means[0]
        for table in domain_means[1:]:
            merged = merged.merge(table, on=["x", "y"], how="inner")
        means[domain] = merged
    shared = means["source"].merge(
        means["target"], on=["x", "y"], how="inner", suffixes=("_source", "_target")
    )
    xy = shared[["x", "y"]].to_numpy()
    full_xy = np.asarray(data["source_rna"]["xy"], dtype=float)
    for gene in genes:
        for domain in ("source", "target"):
            shared[f"{gene}_{domain}"] = gaussian_smooth(
                shared[f"{gene}_{domain}"].to_numpy(), xy, full_xy, sigma_spots=1
            )
    fig, axes = plt.subplots(2, 2, figsize=(5.1, 5.25), layout="constrained")
    scales = {}
    for column, gene in enumerate(genes):
        pooled = np.concatenate([
            shared[f"{gene}_source"].to_numpy(),
            shared[f"{gene}_target"].to_numpy(),
        ])
        norm = Normalize(float(pooled.min()), float(pooled.max()))
        scales[gene] = [float(norm.vmin), float(norm.vmax)]
        for row, domain in enumerate(("source", "target")):
            art = spatial(
                axes[row, column], xy, shared[f"{gene}_{domain}"].to_numpy(),
                norm=norm,
            )
            if row == 0:
                axes[row, column].set_title(gene, fontsize=10.5)
            if column == 0:
                axes[row, column].set_ylabel(
                    "Source RNA" if domain == "source" else "Target RNA",
                    fontsize=10.5,
                )
            fig.colorbar(
                art, ax=axes[row, column], fraction=0.035, pad=0.02
            ).ax.tick_params(labelsize=6)
    save(fig, out / "panel_e_mobp_phactr1_rna")
    return {
        "genes": genes,
        "scale": scales,
        "smoothing": "Gaussian σ=1 spot on the shared source–target lattice",
        "clipping": "none",
        "expression": "RNA AnnData.X (SCT-scaled expression, may be negative)",
        "aggregation": "arithmetic mean at each coordinate, then the source–target intersection",
        "n_shared_spots": int(len(shared)),
        "canvas_inches": [5.1, 5.25],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--panels", nargs="+", choices=["a", "c", "d", "e"],
                        default=["d", "c", "a", "e"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC)
    data = load_metadata()
    meta = provenance()
    if set(args.panels) & {"c", "d"}:
        funcs = original_functions()
        models = fit_models(data, args.output, funcs)
        if "d" in args.panels:
            plot_d(data, models, funcs, args.output)
        if "c" in args.panels:
            meta["pls_color_limit"] = plot_c(data, args.output)
    if "a" in args.panels:
        plot_a(data, args.output)
    if "e" in args.panels:
        meta["phactr1"] = plot_e_and_phactr1(data, args.output)
    manifest = args.output / "provenance.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    manifest.write_text(json.dumps({**previous, **meta}, indent=2) + "\n")


if __name__ == "__main__":
    main()
