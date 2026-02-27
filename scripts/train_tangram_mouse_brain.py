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


try:
    import faiss
except ImportError:
    import torch
    print("faiss is not installed. Using torch.nn.functional.pdist instead.")

def _faiss_gpu_exact_knn_remove_self(
    X,
    k,
    metric="ip",  # "ip" (inner product) or "l2"
    gpu_id=0,
):
    """
    Exact FAISS GPU kNN with robust self-neighbor removal.

    Parameters
    ----------
    X : ndarray, shape (n_samples, d)
        Input vectors (float32 will be enforced).
    k : int
        Number of non-self neighbors to return.
    metric : {"ip", "l2"}
        "ip" for inner product search (use with L2-normalized vectors for cosine),
        "l2" for Euclidean search.
    gpu_id : int
        GPU device id.

    Returns
    -------
    Dk : ndarray, shape (n_samples, k)
        Distances (for metric="ip", this is similarity; caller may convert).
        For metric="l2", FAISS returns squared L2 distances.
    Ik : ndarray, shape (n_samples, k)
        Neighbor indices (self removed).
    """
    X = np.ascontiguousarray(X.astype(np.float32))
    n, d = X.shape

    if k >= n:
        raise ValueError(f"k must be < n_samples (got k={k}, n_samples={n})")

    # Build exact GPU index
    res = faiss.StandardGpuResources()

    if metric == "ip":
        cpu_index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        cpu_index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")

    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    gpu_index.add(X)

    # Search k+1 to allow self-hit
    D, I = gpu_index.search(X, k + 1)

    # Robust self-removal (do not assume self is always the first neighbor)
    rows = np.arange(n)[:, None]
    is_self = (I == rows)

    Dk = np.empty((n, k), dtype=D.dtype)
    Ik = np.empty((n, k), dtype=I.dtype)

    for i in range(n):
        keep = ~is_self[i]
        Ii = I[i][keep]
        Di = D[i][keep]

        if Ii.shape[0] < k:
            # Rare edge case fallback: request more neighbors from CPU exact index
            # (e.g., pathological duplicate/self behavior)
            # This keeps behavior safe and deterministic.
            extra_cpu = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
            extra_cpu.add(X)
            D2, I2 = extra_cpu.search(X[i:i+1], min(n, k + 5))
            I2 = I2[0]
            D2 = D2[0]
            mask2 = I2 != i
            Ii = I2[mask2][:k]
            Di = D2[mask2][:k]

        Ik[i] = Ii[:k]
        Dk[i] = Di[:k]

    return Dk, Ik

def _torch_knn_exact_remove_self(
    X,
    k,
    metric="cosine",   # "cosine" or "euclidean"
    device="cuda",
    chunk_size=None,   # optional: set for memory-safe chunked search
):
    """
    Exact kNN with PyTorch on GPU (or CPU), with self-loop removal.

    Parameters
    ----------
    X : ndarray, shape (n_samples, d)
    k : int
        Number of non-self neighbors to return.
    metric : {"cosine", "euclidean"}
    device : str
        "cuda" or "cpu"
    chunk_size : int or None
        If None, compute full pairwise matrix. If set, do query-chunked search.

    Returns
    -------
    distances : ndarray, shape (n_samples, k)
        cosine distance (1 - cosine sim) or Euclidean distance
    indices : ndarray, shape (n_samples, k)
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    if k >= n:
        raise ValueError(f"k must be < n_samples (got k={k}, n_samples={n})")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    Xt = torch.from_numpy(X).to(dev)

    # Optional normalization for cosine
    if metric == "cosine":
        Xt = torch.nn.functional.normalize(Xt, p=2, dim=1)

    all_dists = []
    all_inds = []

    # Full-matrix mode
    if chunk_size is None:
        if metric == "cosine":
            # similarity matrix
            S = Xt @ Xt.T
            # remove self by setting similarity to -inf
            S.fill_diagonal_(-float("inf"))
            vals, inds = torch.topk(S, k=k, dim=1, largest=True, sorted=True)
            dists = 1.0 - vals  # cosine distance
        elif metric == "euclidean":
            D = torch.cdist(Xt, Xt, p=2)  # exact Euclidean
            D.fill_diagonal_(float("inf"))
            dists, inds = torch.topk(D, k=k, dim=1, largest=False, sorted=True)
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        return dists.detach().cpu().numpy(), inds.detach().cpu().numpy()

    # Chunked mode (memory safer)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Q = Xt[start:end]

        if metric == "cosine":
            S = Q @ Xt.T  # (chunk, n)
            row_idx = torch.arange(start, end, device=dev)
            S[torch.arange(end - start, device=dev), row_idx] = -float("inf")
            vals, inds = torch.topk(S, k=k, dim=1, largest=True, sorted=True)
            dists = 1.0 - vals
        elif metric == "euclidean":
            D = torch.cdist(Q, Xt, p=2)
            row_idx = torch.arange(start, end, device=dev)
            D[torch.arange(end - start, device=dev), row_idx] = float("inf")
            dists, inds = torch.topk(D, k=k, dim=1, largest=False, sorted=True)
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        all_dists.append(dists.detach().cpu())
        all_inds.append(inds.detach().cpu())

    distances = torch.cat(all_dists, dim=0).numpy()
    indices = torch.cat(all_inds, dim=0).numpy()
    return distances, indices


def mapping_matrix_to_spatial_net_affinity(M, obs_names, k, device="cuda", chunk_size=None):
    """
    Option B: build a cell-cell kNN graph from cosine similarity of M rows
    using PyTorch exact search (GPU if available), with self-loops removed.

    M : (n_cells, n_spots) ndarray
    obs_names : array-like length n_cells
    k : int
    device : "cuda" or "cpu"
    chunk_size : int or None
        Use chunking if n_cells is large to avoid OOM.
    """
    M = np.asarray(M, dtype=np.float32)

    # Exact cosine kNN (internally L2-normalizes and removes self)
    distances, indices = _torch_knn_exact_remove_self(
        M, k=k, metric="cosine", device=device, chunk_size=chunk_size
    )

    obs_names = np.asarray(obs_names)
    n_cells = M.shape[0]

    src_idx = np.repeat(np.arange(n_cells), k)
    dst_idx = indices.reshape(-1)
    dist_vals = distances.reshape(-1).astype(np.float32)

    return pd.DataFrame({
        "Cell1": obs_names[src_idx],
        "Cell2": obs_names[dst_idx],
        "Distance": dist_vals,
    })


def mapping_matrix_to_spatial_net_pseudocoords(M, obs_names, sp_coords, k, device="cuda", chunk_size=None):
    """
    Option A: assign pseudo-2-D coordinates via x̃ = M @ coords,
    then build a kNN graph on those coordinates (Euclidean)
    using PyTorch exact search (GPU if available), with self-loops removed.

    M          : (n_cells, n_spots) ndarray
    obs_names  : array-like of length n_cells
    sp_coords  : (n_spots, 2) ndarray
    k          : int
    device     : "cuda" or "cpu"
    chunk_size : int or None
    """
    M = np.asarray(M, dtype=np.float32)
    sp_coords = np.asarray(sp_coords, dtype=np.float32)

    pseudo_coords = M @ sp_coords  # (n_cells, 2)

    distances, indices = _torch_knn_exact_remove_self(
        pseudo_coords, k=k, metric="euclidean", device=device, chunk_size=chunk_size
    )

    obs_names = np.asarray(obs_names)
    n_cells = pseudo_coords.shape[0]

    src_idx = np.repeat(np.arange(n_cells), k)
    dst_idx = indices.reshape(-1)
    dist_vals = distances.reshape(-1).astype(np.float32)

    return pd.DataFrame({
        "Cell1": obs_names[src_idx],
        "Cell2": obs_names[dst_idx],
        "Distance": dist_vals,
    })


#%% ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
#%%
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
        # Rank 0 = highest dispersion, 1 = second highest, ... (correct per-gene rank)
        order = (-adata_sp.var["dispersions_norm"]).argsort()
        rank = np.empty(len(order), dtype=order.dtype)
        rank[order] = np.arange(len(order))
        adata_sp.var["highly_variable_rank"] = rank
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
        verbose=True,
    )
    if args.mode == "clusters":
        map_kwargs["cluster_label"] = args.cluster_label

    adata_map = tg.map_cells_to_space(**map_kwargs)
    # adata_map.X : (n_sc_cells, n_sp_spots) mapping matrix

    print("  Mapping matrix shape:", adata_map.X.shape)

    # Plot Tangram results
    print("\nPlotting Tangram results...")
    tg.plot_training_scores(adata_map, bins=10, alpha=.5)

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
    #M_sparse = sp.csr_matrix(M)
    #mapping_path = os.path.join(args.output_dir, "tangram_mapping_matrix.npz")
    #sp.save_npz(mapping_path, M_sparse)
    #print("  Saved:", mapping_path)

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

#%%
if __name__ == "__main__":
    main()

# %%
