#%%
#!/usr/bin/env python3
"""Run Seurat SCTransform on the RNA modality of SMA RNA+MSI MuData files.

This mirrors the spatial RNA-ATAC SCT preprocessing used in
``rna_atac/SCT_compare_feature_representations.py``, adapted for SMA
``.h5mu`` exports:

* input:  ``<sample>.h5mu`` with ``mdata.mod["rna"].layers["counts"]`` or raw
  counts in ``.X``
* output: ``<sample>_SCT.h5mu`` by default
* RNA output: ``.X`` is SCT ``scale.data``; raw counts are preserved in
  ``layers["counts"]``; SCT corrected counts/data are stored in
  ``layers["SCT_counts"]`` and ``layers["SCT_data"]``; variable SCT genes are
  marked in ``var["SCT_gene"]``.

The MSI modality and shared spatial metadata are copied through unchanged.
"""

from __future__ import annotations
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.environ["SPATIALJEPA_CONFIG"])

# Bare /lib entries in LD_LIBRARY_PATH (from .bashrc + CUDA repeats) make R extensions
# such as spam.so resolve the system libstdc++, which lacks CXXABI_1.3.15.
_SYSTEM_LD_PATHS = frozenset({
    "/lib",
    "/lib64",
    "/lib/x86_64-linux-gnu",
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
})


def _running_ipython_kernel() -> bool:
    """Detect notebook execution early, before importing IPython directly."""
    return "ipykernel" in sys.modules or bool(os.environ.get("JPY_PARENT_PID"))


def _configure_conda_runtime() -> None:
    """Pin conda R and libstdc++ before Python loads native extensions."""
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    prefix_path = Path(prefix)
    conda_lib = prefix_path / "lib"

    r_home = prefix_path / "lib" / "R"
    if r_home.is_dir():
        os.environ["R_HOME"] = str(r_home)
        os.environ["R_LIBS"] = str(r_home / "library")
        os.environ.pop("R_LIBS_USER", None)

    if conda_lib.is_dir():
        cleaned = [
            entry
            for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":")
            if entry and entry not in _SYSTEM_LD_PATHS
        ]
        os.environ["LD_LIBRARY_PATH"] = ":".join([str(conda_lib), *cleaned])

    libstdcxx = conda_lib / "libstdc++.so.6"
    needs_reexec = (
        libstdcxx.exists()
        and os.environ.get("SMA_SCT_RUNTIME_REEXEC") != "1"
        and str(libstdcxx) not in os.environ.get("LD_PRELOAD", "")
        and not _running_ipython_kernel()
    )
    if needs_reexec:
        env = os.environ.copy()
        current_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = (
            f"{libstdcxx}:{current_preload}" if current_preload else str(libstdcxx)
        )
        env["SMA_SCT_RUNTIME_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_configure_conda_runtime()

import anndata as ad
import anndata2ri
import mudata as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import issparse

import rpy2.robjects as ro


DEFAULT_RNA_MOD = "rna"
DEFAULT_COUNTS_LAYER = "counts"
DEFAULT_SUFFIX = "_SCT"
DEFAULT_VARIABLE_FEATURES_N = 6000
DEFAULT_N_PCS = 30


def default_export_dir() -> Path:
    datapath = os.environ.get("SPATIALJEPA_DATA")
    if not datapath:
        raise RuntimeError("SPATIALJEPA_DATA is not set; source configs/spatialjepa.env or pass --input-dir.")
    return Path(datapath) / "vicari_2023" / "h5mu_export"


def default_plot_dir() -> Path:
    outpath = os.environ.get("SPATIALJEPA_OUT")
    if outpath:
        return Path(outpath) / "SMA" / "SCT_transform"
    return default_export_dir() / "SCT_transform_plots"


def parse_sample_values(values: Optional[Iterable[str]]) -> Optional[list[str]]:
    if not values:
        return None
    samples: list[str] = []
    for value in values:
        samples.extend(part.strip() for part in value.split(",") if part.strip())
    return samples or None


def sample_to_path(input_dir: Path, sample: str) -> Path:
    path = Path(sample)
    if path.suffix == ".h5mu":
        return path if path.is_absolute() else input_dir / path
    return input_dir / f"{sample}.h5mu"


def discover_input_paths(
    input_dir: Path,
    samples: Optional[list[str]],
    sample_glob: str,
    suffix: str,
) -> list[Path]:
    if samples:
        paths = [sample_to_path(input_dir, sample) for sample in samples]
    else:
        paths = sorted(input_dir.glob(sample_glob))
        paths = [
            path
            for path in paths
            if path.suffix == ".h5mu" and not path.stem.endswith(suffix)
        ]

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input .h5mu files: " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError(f"No input .h5mu files found in {input_dir}")
    return paths


def output_path_for(input_path: Path, output_dir: Path, suffix: str) -> Path:
    stem = input_path.stem
    if not stem.endswith(suffix):
        stem = f"{stem}{suffix}"
    return output_dir / f"{stem}.h5mu"


def sort_sparse_indices(adata: ad.AnnData) -> None:
    if issparse(adata.X) and not adata.X.has_sorted_indices:
        adata.X.sort_indices()
    for key in adata.layers:
        layer = adata.layers[key]
        if issparse(layer) and not layer.has_sorted_indices:
            layer.sort_indices()


def matrix_to_float32(matrix):
    if sp.issparse(matrix):
        return matrix.astype(np.float32).tocsr()
    return np.asarray(matrix, dtype=np.float32)


def align_matrix_to_vars(matrix, matrix_vars: list[str], target_vars: pd.Index, name: str):
    matrix_vars_index = pd.Index(matrix_vars)
    if matrix_vars_index.equals(target_vars):
        return matrix_to_float32(matrix)

    indexer = matrix_vars_index.get_indexer(target_vars)
    if (indexer < 0).any():
        missing = list(target_vars[indexer < 0][:10])
        raise ValueError(
            f"{name} is missing {int((indexer < 0).sum())} RNA features; "
            f"examples: {missing}"
        )
    return matrix_to_float32(matrix[:, indexer])


def mitochondrial_features(var_names: Iterable[str]) -> list[str]:
    """Return mitochondrial features across common human/mouse symbol styles."""
    mito_re = re.compile(r"^mt[-.]", re.IGNORECASE)
    return [str(name) for name in var_names if mito_re.match(str(name))]


def counts_adata(adata: ad.AnnData, counts_layer: str) -> ad.AnnData:
    if counts_layer in adata.layers:
        X = adata.layers[counts_layer].copy()
    else:
        X = adata.X.copy()
    out = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())
    sort_sparse_indices(out)
    return out


def run_sctransform(
    adata: ad.AnnData,
    counts_layer: str = DEFAULT_COUNTS_LAYER,
    cell_type_key: str = "RNA_clusters",
    sample_id: str = "sample",
    variable_features_n: int = DEFAULT_VARIABLE_FEATURES_N,
    n_pcs: int = DEFAULT_N_PCS,
    plot_dir: Optional[Path] = None,
) -> ad.AnnData:
    """Run Seurat SCTransform and attach SCT matrices/metadata to an AnnData copy."""
    if not adata.obs_names.is_unique:
        raise ValueError("RNA obs_names must be unique before SCTransform.")
    if not adata.var_names.is_unique:
        raise ValueError("RNA var_names must be unique before SCTransform.")

    out = adata.copy()
    rna_counts = counts_adata(out, counts_layer=counts_layer)
    mt_features = mitochondrial_features(rna_counts.var_names)

    ro.r(
        """
        suppressPackageStartupMessages({
            library(Seurat)
            library(scater)
        })
        """
    )
    anndata2ri.activate()

    ro.globalenv["adata"] = rna_counts
    ro.globalenv["mt_features"] = ro.StrVector(mt_features)
    ro.globalenv["cell_type_key"] = ro.StrVector([cell_type_key])
    ro.globalenv["sample_id"] = ro.StrVector([sample_id])
    ro.globalenv["variable_features_n"] = int(variable_features_n)
    ro.globalenv["n_pcs_requested"] = int(n_pcs)
    ro.globalenv["make_plots"] = bool(plot_dir is not None)
    ro.globalenv["plot_dir"] = ro.StrVector([str(plot_dir) if plot_dir else ""])

    ro.r(
        """
        seurat_obj <- as.Seurat(adata, counts = "X", data = NULL)
        DefaultAssay(seurat_obj) <- "originalexp"

        vars_to_regress <- NULL
        if (length(mt_features) > 0) {
            seurat_obj <- PercentageFeatureSet(
                seurat_obj,
                features = mt_features,
                col.name = "percent.mt"
            )
            vars_to_regress <- "percent.mt"
        } else {
            seurat_obj$percent.mt <- 0
        }

        res <- SCTransform(
            object = seurat_obj,
            assay = "originalexp",
            vars.to.regress = vars_to_regress,
            return.only.var.genes = FALSE,
            do.correct.umi = TRUE,
            verbose = FALSE,
            variable.features.n = variable_features_n,
            min_cells = 0
        )

        sct_genes <- VariableFeatures(res)
        sct_feature_names <- rownames(res[["SCT"]])

        n_pcs_eff <- min(n_pcs_requested, ncol(res) - 1, length(sct_feature_names) - 1)
        if (n_pcs_eff >= 2) {
            res <- RunPCA(
                res,
                verbose = FALSE,
                features = sct_feature_names,
                npcs = n_pcs_eff
            )
            res <- RunUMAP(res, dims = 1:n_pcs_eff, verbose = FALSE)
            res <- FindNeighbors(res, dims = 1:n_pcs_eff, verbose = FALSE)
            res <- FindClusters(res, verbose = FALSE)

            plot_group <- cell_type_key
            if (!(plot_group %in% colnames(res@meta.data))) {
                plot_group <- "seurat_clusters"
            }
            if (make_plots && plot_group %in% colnames(res@meta.data)) {
                dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
                p <- DimPlot(res, group.by = plot_group, label = TRUE) +
                    ggplot2::ggtitle(paste0(sample_id, " - ", plot_group))
                ggplot2::ggsave(
                    file.path(plot_dir, paste0(sample_id, "_seurat_dimplot_", plot_group, ".png")),
                    plot = p,
                    width = 7,
                    height = 5,
                    dpi = 150
                )
            }
        }

        get_sct_layer <- function(layer_name) {
            tryCatch(
                GetAssayData(res, assay = "SCT", layer = layer_name),
                error = function(e) GetAssayData(res, assay = "SCT", slot = layer_name)
            )
        }

        sct_counts_mat <- get_sct_layer("counts")
        sct_data_mat <- get_sct_layer("data")
        sct_scale_mat <- get_sct_layer("scale.data")

        sct_counts_features <- rownames(sct_counts_mat)
        sct_data_features <- rownames(sct_data_mat)
        sct_scale_features <- rownames(sct_scale_mat)

        sct_counts_t <- t(sct_counts_mat)
        sct_data_t <- t(sct_data_mat)
        sct_scale_t <- t(sct_scale_mat)
        seurat_obs <- as.data.frame(res@meta.data)
        seurat_obs_names <- rownames(seurat_obs)
        """
    )

    seurat_obs = ro.r("seurat_obs")
    seurat_obs.index = list(ro.r("seurat_obs_names"))
    seurat_obs = seurat_obs.reindex(out.obs_names)
    # Only add columns SCTransform introduces (percent.mt, nCount_SCT, nFeature_SCT, ...).
    # Re-assigning pre-existing columns would round-trip them through R and recast dtypes
    # (e.g. int64 cluster labels -> int32), which later breaks a per-modality
    # concat(merge="same") against the untouched MSI side and silently drops shared labels.
    for col in seurat_obs.columns:
        if col not in out.obs.columns:
            out.obs[col] = seurat_obs[col].values

    if "pca" in list(ro.r("names(res@reductions)")):
        out.obsm["X_seurat_pca"] = np.asarray(ro.r("res@reductions$pca@cell.embeddings"))
    if "umap" in list(ro.r("names(res@reductions)")):
        out.obsm["X_seurat_umap"] = np.asarray(ro.r("res@reductions$umap@cell.embeddings"))

    target_vars = pd.Index(out.var_names)
    out.layers["SCT_counts"] = align_matrix_to_vars(
        ro.r("sct_counts_t"),
        list(ro.r("sct_counts_features")),
        target_vars,
        "SCT counts",
    )
    out.layers["SCT_data"] = align_matrix_to_vars(
        ro.r("sct_data_t"),
        list(ro.r("sct_data_features")),
        target_vars,
        "SCT data",
    )
    out.X = align_matrix_to_vars(
        ro.r("sct_scale_t"),
        list(ro.r("sct_scale_features")),
        target_vars,
        "SCT scale.data",
    )

    sct_genes = set(str(gene) for gene in list(ro.r("sct_genes")))
    out.var["SCT_gene"] = out.var_names.isin(sct_genes)
    out.uns["SCT"] = {
        "method": "Seurat::SCTransform",
        "counts_layer": counts_layer if counts_layer in adata.layers else "X",
        "vars_to_regress": ["percent.mt"] if mt_features else [],
        "variable_features_n": int(variable_features_n),
        "n_sct_genes": int(out.var["SCT_gene"].sum()),
        "n_mito_features": int(len(mt_features)),
    }

    if plot_dir is not None:
        save_qc_pca_correlation(out, sample_id=sample_id, plot_dir=plot_dir)

    return out


def save_qc_pca_correlation(adata: ad.AnnData, sample_id: str, plot_dir: Path) -> None:
    qc_cols = [
        col
        for col in [
            "percent.mt",
            "nFeature_SCT",
            "nCount_SCT",
            "nFeature_originalexp",
            "nCount_originalexp",
        ]
        if col in adata.obs
    ]
    if not qc_cols:
        return

    n_comps = min(50, adata.n_obs - 1, adata.n_vars - 1)
    if n_comps < 2:
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    sc.pp.pca(adata, n_comps=n_comps)
    pca_df = pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names)
    df = adata.obs[qc_cols].merge(pca_df, left_index=True, right_index=True)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    sns.heatmap(df.corr().iloc[: len(qc_cols), len(qc_cols) :], ax=ax)
    ax.set_xlabel("PCA components")
    ax.set_title(f"{sample_id}: QC correlation with PCA from SCT features")
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"{sample_id}_sct_qc_pca_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def transform_mudata_file(
    input_path: Path,
    output_path: Path,
    rna_mod: str,
    counts_layer: str,
    cell_type_key: str,
    variable_features_n: int,
    n_pcs: int,
    plot_dir: Optional[Path],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path} (pass --overwrite)")

    print(f"[INFO] reading {input_path}")
    mdata = mu.read_h5mu(input_path)
    if rna_mod not in mdata.mod:
        raise KeyError(f"{input_path} has no modality {rna_mod!r}; available: {list(mdata.mod)}")

    sample_id = input_path.stem
    mdata.mod[rna_mod] = run_sctransform(
        mdata.mod[rna_mod],
        counts_layer=counts_layer,
        cell_type_key=cell_type_key,
        sample_id=sample_id,
        variable_features_n=variable_features_n,
        n_pcs=n_pcs,
        plot_dir=plot_dir,
    )
    mdata.update()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] writing {output_path}")
    mdata.write(output_path)

    n_sct = int(mdata.mod[rna_mod].var["SCT_gene"].sum())
    print(f"[OK] {sample_id}: {n_sct} SCT variable genes -> {output_path}")


def parse_args(notebook: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing SMA .h5mu files (default: $SPATIALJEPA_DATA/vicari_2023/h5mu_export)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for SCT .h5mu files (default: --input-dir)",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        help="Sample IDs or .h5mu paths. Comma-separated and space-separated forms are accepted.",
    )
    parser.add_argument(
        "--sample-glob",
        default="*.h5mu",
        help="Glob used when --samples is omitted (default: *.h5mu)",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help=f"Output suffix before .h5mu (default: {DEFAULT_SUFFIX})",
    )
    parser.add_argument(
        "--rna-mod",
        default=DEFAULT_RNA_MOD,
        help=f"RNA modality key in MuData (default: {DEFAULT_RNA_MOD})",
    )
    parser.add_argument(
        "--counts-layer",
        default=DEFAULT_COUNTS_LAYER,
        help=f"Raw count layer to use if present (default: {DEFAULT_COUNTS_LAYER})",
    )
    parser.add_argument(
        "--cell-type-key",
        default="RNA_clusters",
        help="obs column for Seurat DimPlot labels; falls back to seurat_clusters if absent.",
    )
    parser.add_argument(
        "--variable-features-n",
        type=int,
        default=DEFAULT_VARIABLE_FEATURES_N,
        help=f"SCTransform variable.features.n (default: {DEFAULT_VARIABLE_FEATURES_N})",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=DEFAULT_N_PCS,
        help=f"Number of PCs for Seurat PCA/UMAP/neighbors (default: {DEFAULT_N_PCS})",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for SCT QC plots (default: $SPATIALJEPA_OUT/SMA/SCT_transform)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Seurat DimPlot and QC/PCA heatmap outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    if notebook:
        return parser.parse_known_args()[0]
    return parser.parse_args()

def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False

#%%
def main() -> None:
    args = parse_args(is_notebook())
    input_dir = args.input_dir or default_export_dir()
    output_dir = args.output_dir or input_dir
    plot_dir = None if args.no_plots else (args.plot_dir or default_plot_dir())
    samples = parse_sample_values(args.samples)

    input_paths = discover_input_paths(
        input_dir=input_dir,
        samples=samples,
        sample_glob=args.sample_glob,
        suffix=args.suffix,
    )

    print(f"[INFO] processing {len(input_paths)} file(s)")
    for input_path in input_paths:
        transform_mudata_file(
            input_path=input_path,
            output_path=output_path_for(input_path, output_dir, args.suffix),
            rna_mod=args.rna_mod,
            counts_layer=args.counts_layer,
            cell_type_key=args.cell_type_key,
            variable_features_n=args.variable_features_n,
            n_pcs=args.n_pcs,
            plot_dir=plot_dir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
