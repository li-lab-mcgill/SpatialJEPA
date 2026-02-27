#!/usr/bin/env python
"""
Train a Tangram cell-to-space mapping model for mouse brain snMultiome data.

Tangram maps single-cell RNA expression onto a spatial reference by optimising
a cell-to-spot probability matrix M (n_sc_cells × n_sp_spots).  Two
Spatial_Net graphs (compatible with MultiGATE) are derived from M and saved:

  Option A — pseudo-coordinate kNN:
      Each sc cell is assigned pseudo-coordinates x̃ = M @ coords_sp, then a
      kNN graph is built on those 2-D coordinates.

  Option B — mapping-overlap affinity (recommended):
      Cell-cell similarity is defined as the cosine similarity of their spot
      probability vectors (rows of M).  A kNN graph is built on these vectors.

Both graphs are saved as CSV files whose schema matches adata.uns['Spatial_Net']
(columns: Cell1, Cell2, Distance) so they can be loaded directly into
mouse_brain_spatial_rna_atac.py.

The raw mapping matrix M is also saved as a scipy sparse .npz for downstream use.

Usage:
    python train_tangram_mouse_brain.py \\
        --sp-path  $DATAPATH/aligned_data/source_rna_aligned.h5ad \\
        --sc-path  $DATAPATH/aligned_data/target_rna_aligned.h5ad \\
        --n-epochs 1000 \\
        --k-neighbors 15 \\
        --mode cells \\
        --output-dir $OUTPATH/tangram

Defaults use source_rna_aligned.h5ad for both spatial reference and sc query
(i.e. self-mapping), which is useful for validating the pipeline before a
separate non-spatial dataset is available.
"""
# %%
import argparse
import os
import socket
import sys
from pprint import pprint

from dotenv import dotenv_values, load_dotenv

load_dotenv(dotenv_path="/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env")
print("Loaded environment variables from .env or env:", end="\n\n")
pprint(dotenv_values("/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env"))

if os.getenv("DATAPATH") is None:
    raise EnvironmentError(
        "DATAPATH is not set. Export DATAPATH to the base data directory, e.g. "
        "'/home/mcb/users/dmannk/BAKLAVA_base/data'."
    )

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import tangram as tg
except:
    from scvi.external import Tangram as tg

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        else:
            # Other types
            return False
    except Exception:
        return False

def parse_args(notebook=False):
    base_path = os.path.join(os.getenv("DATAPATH"), "aligned_data")
    default_output_dir = os.path.join(os.getenv("OUTPATH", os.path.join(os.getenv("DATAPATH"), "..", "outputs")), "tangram")

    parser = argparse.ArgumentParser(
        description="Train Tangram and derive pseudo-spatial graphs for MultiGATE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-data",
        type=str,
        choices=["spatial_rna_atac"],
        default="spatial_rna_atac",
        help="Source RNA data to use for Tangram training.",
    )
    parser.add_argument(
        "--target-data",
        type=str,
        choices=["10x_mouse_brain", "10x_mouse_brain_AD"],
        default="10x_mouse_brain",
        help="Target RNA data to use for Tangram mapping.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help=(
            "Number of top highly-variable genes (ranked by highly_variable_rank) "
            "to use as Tangram training genes.  Must match MultiGATE's HVG selection."
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of Tangram optimisation epochs.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=15,
        help="Number of nearest neighbours for graph construction (both options A and B).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cells", "clusters"],
        default="cells",
        help=(
            "Tangram mapping mode. 'cells' maps each sc cell individually "
            "(accurate, slower for large datasets).  'clusters' maps cell "
            "clusters (faster, lower resolution, requires --cluster-label)."
        ),
    )
    parser.add_argument(
        "--cluster-label",
        type=str,
        default=None,
        help="obs key for pre-computed cluster labels. Required when --mode=clusters.",
    )
    parser.add_argument(
        "--density-prior",
        type=str,
        choices=["rna_count_based", "uniform"],
        default="rna_count_based",
        help="Tangram density prior. 'rna_count_based' weights spots by total counts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PyTorch device for Tangram optimisation (e.g. 'cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Directory where output files will be written.",
    )
    if notebook:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_hvgs(adata, n_top):
    """Return HVG names ranked by highly_variable_rank, top-n."""
    if "highly_variable_rank" not in adata.var.columns:
        raise KeyError(
            "adata.var does not contain 'highly_variable_rank'. "
            "Run sc.pp.highly_variable_genes(..., inplace=True) first."
        )
    ranked = adata.var["highly_variable_rank"].dropna()
    hvg_names = ranked[ranked <= (n_top - 1)].index.tolist()
    if len(hvg_names) == 0:
        raise ValueError(
            "No genes found with highly_variable_rank <= {}. "
            "Check n_top_genes parameter.".format(n_top - 1)
        )
    return hvg_names


def mapping_matrix_to_spatial_net_affinity(M, obs_names, k):
    """
    Option B: build a cell-cell kNN graph from cosine similarity of M rows.

    M : (n_cells, n_spots) ndarray — each row is a cell's probability
        distribution over spatial spots.
    obs_names : array-like of length n_cells — cell barcodes.
    k : int — number of nearest neighbours per cell.

    Returns a DataFrame with columns Cell1, Cell2, Distance (1 - cosine_sim),
    compatible with adata.uns['Spatial_Net'].
    """
    # Cosine similarity: normalise rows then dot-product
    M_norm = normalize(M, norm="l2", axis=1)
    # NearestNeighbors with cosine metric; k+1 to exclude the cell itself
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nbrs.fit(M_norm)
    distances, indices = nbrs.kneighbors(M_norm)

    # distances[:,0] == 0 (self); slice it off
    distances = distances[:, 1:]   # cosine distance = 1 - cosine_similarity
    indices = indices[:, 1:]

    n_cells = M.shape[0]
    cell1, cell2, dist_vals = [], [], []
    for i in range(n_cells):
        for rank in range(k):
            j = int(indices[i, rank])
            cell1.append(obs_names[i])
            cell2.append(obs_names[j])
            dist_vals.append(float(distances[i, rank]))

    return pd.DataFrame({"Cell1": cell1, "Cell2": cell2, "Distance": dist_vals})


def mapping_matrix_to_spatial_net_pseudocoords(M, obs_names, sp_coords, k):
    """
    Option A: assign pseudo-2-D coordinates to each sc cell via x̃ = M @ coords,
    then build a kNN graph on those coordinates.

    M          : (n_cells, n_spots) ndarray
    obs_names  : array-like of length n_cells
    sp_coords  : (n_spots, 2) ndarray — spatial coordinates of each spot
    k          : int — number of nearest neighbours

    Returns a DataFrame with columns Cell1, Cell2, Distance.
    """
    pseudo_coords = M @ sp_coords  # (n_cells, 2)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(pseudo_coords)
    distances, indices = nbrs.kneighbors(pseudo_coords)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    n_cells = M.shape[0]
    cell1, cell2, dist_vals = [], [], []
    for i in range(n_cells):
        for rank in range(k):
            j = int(indices[i, rank])
            cell1.append(obs_names[i])
            cell2.append(obs_names[j])
            dist_vals.append(float(distances[i, rank]))

    return pd.DataFrame({"Cell1": cell1, "Cell2": cell2, "Distance": dist_vals})


#%% ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    notebook = is_notebook()
    args = parse_args(notebook=notebook)

    if args.mode == "clusters" and args.cluster_label is None:
        raise ValueError("--cluster-label is required when --mode=clusters.")

    os.makedirs(args.output_dir, exist_ok=True)
    print("Output directory:", args.output_dir)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading data...")
    
    if args.source_data == "spatial_rna_atac":
        adata_sp = sc.read_h5ad(os.path.join(os.getenv("DATAPATH"), "aligned_data", "source_rna_aligned.h5ad"))

    if args.target_data == "10x_mouse_brain":
        adata_sc = sc.read_h5ad(os.path.join(os.getenv("DATAPATH"), "aligned_data", "target_rna_aligned.h5ad"))

    print("  Spatial reference: {} cells, {} genes".format(adata_sp.n_obs, adata_sp.n_vars))
    print("  SC query:          {} cells, {} genes".format(adata_sc.n_obs, adata_sc.n_vars))

    if "spatial" not in adata_sp.obsm:
        raise KeyError(
            "adata_sp.obsm does not contain 'spatial'. "
            "The spatial reference must have 2-D coordinates in obsm['spatial']."
        )

    # Apply the same coordinate flip as mouse_brain_spatial_rna_atac.py
    adata_sp.obsm["spatial"] = adata_sp.obsm["spatial"][:, [1, 0]] * -1

    # ------------------------------------------------------------------
    # 2. HVG selection — use the spatial reference HVGs as training genes
    #    (mirrors the n_top_genes=2000 selection in mouse_brain_spatial_rna_atac.py)
    # ------------------------------------------------------------------
    print("\n[2] Selecting training genes from spatial reference HVGs...")

    if socket.gethostname() != "ri-muhc-gpu":
        # Replicate the HVG override in mouse_brain_spatial_rna_atac.py
        adata_sp.var["highly_variable"] = False
        adata_sp.var.loc[
            adata_sp.var["highly_variable_rank"].le(args.n_top_genes - 1),
            "highly_variable",
        ] = True

    training_genes = adata_sp.var_names[adata_sp.var["highly_variable"]].tolist()

    # Restrict to genes also present in the sc query
    training_genes = [g for g in training_genes if g in adata_sc.var_names]

    if len(training_genes) == 0:
        raise ValueError(
            "No overlap between spatial-reference HVGs and sc-query gene names. "
            "Check that both datasets share the same gene namespace."
        )
    print("  Training genes: {} (after intersecting with sc query)".format(len(training_genes)))

    # ------------------------------------------------------------------
    # 3. Preprocess adatas for Tangram
    #    tg.pp_adatas filters both adatas to shared genes and stores
    #    training gene info in .uns["training_genes"].
    # ------------------------------------------------------------------
    print("\n[3] Preprocessing adatas for Tangram...")

    # Work on copies so originals stay intact
    adata_sp_tg = adata_sp.copy()
    adata_sc_tg = adata_sc.copy()

    tg.pp_adatas(adata_sc_tg, adata_sp_tg, genes=training_genes)

    print(
        "  After tg.pp_adatas: sc={} genes, sp={} genes".format(
            adata_sc_tg.n_vars, adata_sp_tg.n_vars
        )
    )

    # ------------------------------------------------------------------
    # 4. Run Tangram mapping
    # ------------------------------------------------------------------
    print("\n[4] Running Tangram mapping (mode={}, n_epochs={}, device={})...".format(
        args.mode, args.n_epochs, args.device
    ))

    map_kwargs = dict(
        adata_sc=adata_sc_tg,
        adata_sp=adata_sp_tg,
        mode=args.mode,
        density_prior=args.density_prior,
        num_epochs=args.n_epochs,
        device=args.device,
        verbose=False,
    )
    if args.mode == "clusters":
        map_kwargs["cluster_label"] = args.cluster_label

    adata_map = tg.map_cells_to_space(**map_kwargs)
    # adata_map.X : (n_sc_cells, n_sp_spots) mapping matrix

    print("  Mapping matrix shape:", adata_map.X.shape)

    # ------------------------------------------------------------------
    # 5. Extract mapping matrix M
    # ------------------------------------------------------------------
    M = adata_map.X
    if sp.issparse(M):
        M = M.toarray()
    M = np.asarray(M, dtype=np.float32)  # (n_sc_cells, n_sp_spots)

    sc_obs_names = np.array(adata_sc_tg.obs_names)
    sp_obs_names = np.array(adata_sp_tg.obs_names)
    sp_coords = adata_sp_tg.obsm["spatial"]  # (n_sp_spots, 2)

    # ------------------------------------------------------------------
    # 6. Build Spatial_Net graphs
    # ------------------------------------------------------------------
    k = args.k_neighbors

    print("\n[5] Building Option B: mapping-overlap affinity graph (k={})...".format(k))
    spatial_net_affinity = mapping_matrix_to_spatial_net_affinity(M, sc_obs_names, k)
    print("  Edges: {}".format(len(spatial_net_affinity)))

    print("[6] Building Option A: pseudo-coordinate kNN graph (k={})...".format(k))
    spatial_net_pseudocoords = mapping_matrix_to_spatial_net_pseudocoords(
        M, sc_obs_names, sp_coords, k
    )
    print("  Edges: {}".format(len(spatial_net_pseudocoords)))

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    print("\n[7] Saving outputs to {}...".format(args.output_dir))

    # Spatial_Net CSV files (ready to load as adata.uns['Spatial_Net'])
    affinity_path = os.path.join(args.output_dir, "tangram_spatial_net_affinity.csv")
    spatial_net_affinity.to_csv(affinity_path, index=False)
    print("  Saved:", affinity_path)

    pseudocoords_path = os.path.join(args.output_dir, "tangram_spatial_net_pseudocoords.csv")
    spatial_net_pseudocoords.to_csv(pseudocoords_path, index=False)
    print("  Saved:", pseudocoords_path)

    # Raw mapping matrix (sparse, for flexibility)
    M_sparse = sp.csr_matrix(M)
    mapping_path = os.path.join(args.output_dir, "tangram_mapping_matrix.npz")
    sp.save_npz(mapping_path, M_sparse)
    print("  Saved:", mapping_path)

    # Cell / spot name arrays (index into M rows/columns)
    sc_names_path = os.path.join(args.output_dir, "tangram_sc_obs_names.txt")
    np.savetxt(sc_names_path, sc_obs_names, fmt="%s")
    print("  Saved:", sc_names_path)

    sp_names_path = os.path.join(args.output_dir, "tangram_sp_obs_names.txt")
    np.savetxt(sp_names_path, sp_obs_names, fmt="%s")
    print("  Saved:", sp_names_path)

    print("\nDone.")
    print(
        "\nTo use the affinity graph in mouse_brain_spatial_rna_atac.py, add a "
        "'tangram' branch to the graph_type block:\n\n"
        "    elif graph_type == 'tangram':\n"
        "        tangram_net = pd.read_csv('{}')\n"
        "        target_rna.uns['Spatial_Net'] = tangram_net\n"
        "        target_atac.uns['Spatial_Net'] = tangram_net.copy()".format(affinity_path)
    )


if __name__ == "__main__":
    main()
