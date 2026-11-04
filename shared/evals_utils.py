import os

from anndata import AnnData
from typing import List

# Keep JAX/XLA memory usage conservative for long HPO runs.
# These need to be set before importing `jax`.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault(
    "JAX_DISABLE_JIT",
    os.environ.get("BAKLAVA_JAX_DISABLE_JIT", "1"),
)

import numpy as np
import jax.numpy as jnp
from jax import jit
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

## cdist and foscttm_moscot from ECLARE/evals_utils.py
@jit
def cdist(
    x: np.ndarray,
    y: np.ndarray,
    metric: str='cosine'
) -> np.ndarray:
    """Compute the pairwise distance matrix between each row of x and y."""
    x2 = jnp.sum(x**2, axis=1, keepdims=True)
    y2 = jnp.sum(y**2, axis=1, keepdims=True)
    xy = jnp.dot(x, y.T)

    if metric == 'euclidean':
        return jnp.sqrt(x2 - 2*xy + y2.T)
    elif metric == 'cosine':
        return 1 - (xy / (jnp.sqrt(x2) * jnp.sqrt(y2.T)))
    elif metric == 'inner':
        return -xy

@jit
def foscttm_moscot( # from https://moscot.readthedocs.io/en/latest/notebooks/tutorials/600_tutorial_translation.html#define-utility-functions
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    c = cdist(x, y)
    foscttm_x = (c < jnp.expand_dims(jnp.diag(c), axis=1)).mean(axis=1)
    foscttm_y = (c < jnp.expand_dims(jnp.diag(c), axis=0)).mean(axis=0)
    foscttm_full = (foscttm_x + foscttm_y) / 2 #jnp.mean(foscttm_x + foscttm_y) / 2
    return foscttm_full


def benchmark_embeddings(
    adata: AnnData,
    batch_key: str,
    label_key: str,
    embedding_obsm_keys: List[str],
    n_jobs: int=1,
) -> None:

    adata.obsm["nichecompass_latent"] = adata.X

    if label_key is None:

        adata.obs["dummy_label"] = "all"

        batch_metrics = BatchCorrection(
            ilisi_knn=True,
            pcr_comparison=True,
            bras=False,
            graph_connectivity=False,
            kbet_per_label=False,
        )

        bm = Benchmarker(
            adata=adata,
            batch_key=batch_key,
            label_key="dummy_label",          # must be a string key per API
            embedding_obsm_keys=embedding_obsm_keys,
            bio_conservation_metrics=None,    # ignore all BioConservation
            batch_correction_metrics=batch_metrics,
            n_jobs=n_jobs,
        )
    else:
        bm = Benchmarker(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            bio_conservation_metrics=BioConservation(
                isolated_labels=False,
                nmi_ari_cluster_labels_leiden=False,
                nmi_ari_cluster_labels_kmeans=True,
                silhouette_label=True,
                clisi_knn=False
            ),
            batch_correction_metrics=BatchCorrection(
                bras=False,
                ilisi_knn=True,
                kbet_per_label=False,
                graph_connectivity=False,
                pcr_comparison=False,
            ),
            embedding_obsm_keys=embedding_obsm_keys,
            n_jobs=n_jobs,
        )

    bm.benchmark()

    return bm.get_results().loc['nichecompass_latent']
