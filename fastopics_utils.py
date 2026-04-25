from fastopic import FASTopic

import scipy.sparse as sp
import numpy as np

__all__ = ["NumericPreprocess", "fit_fastopic"]

class NumericPreprocess:
    def __init__(self, X_csr, gene_names):
        # X_csr: shape (n_cells, n_genes), CSR or convertible to CSR
        # gene_names: list/array of length n_genes (strings)
        self.X_csr = sp.csr_matrix(X_csr)
        self.vocab = list(map(str, gene_names))

    def preprocess(self, docs):
        # docs is ignored; FASTopic just expects this interface
        train_bow = self.X_csr
        vocab = self.vocab
        return {
            "train_bow": train_bow,   # scipy.sparse CSR
            "vocab": vocab            # List[str]
        }

class PresetOnlyDocEmbedder:
    def encode(self, docs):
        raise RuntimeError("preset_doc_embeddings must be provided for FASTopic.")

def fit_fastopic(X, genes, fixed_embeddings, num_topics=50):
    # 1) Wrap numeric matrix and gene names.
    preprocess = NumericPreprocess(X, genes)
    fixed_embeddings = np.asarray(fixed_embeddings, dtype=np.float32)

    if preprocess.X_csr.shape[0] != fixed_embeddings.shape[0]:
        raise ValueError(
            "X and fixed_embeddings must have the same number of cells: "
            f"{preprocess.X_csr.shape[0]} != {fixed_embeddings.shape[0]}"
        )
    if preprocess.X_csr.shape[1] != len(preprocess.vocab):
        raise ValueError(
            "X and genes must have the same number of genes: "
            f"{preprocess.X_csr.shape[1]} != {len(preprocess.vocab)}"
        )

    # 2) Build dummy docs (ignored by our preprocess).
    n_cells = preprocess.X_csr.shape[0]
    dummy_docs = ["x"] * n_cells   # any constant string

    # 3) Instantiate FASTopic with a no-op embedder to avoid text model downloads.
    model = FASTopic(num_topics, preprocess=preprocess,
                    doc_embed_model=PresetOnlyDocEmbedder(),
                    low_memory=True, low_memory_batch_size=2000)

    # 4) Fit using counts as bag-of-genes and fixed embeddings as document embeddings.
    top_words, doc_topic_dist = model.fit_transform(
        dummy_docs,
        preset_doc_embeddings=fixed_embeddings,
    )
    return top_words, doc_topic_dist

if __name__ == "__main__":
    import scanpy as sc
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    source_rna = sc.read_h5ad("/Users/dmannk/BAKLAVA_base/data/aligned_data/source_rna_aligned_with_latents.h5ad")
    
    fixed_embeddings = source_rna.obsm["MultiGATE"].copy()
    rna_counts = source_rna.layers["counts"].copy()

    top_genes, cell_topic_dist = fit_fastopic(rna_counts, source_rna.var_names, fixed_embeddings)
    print(top_genes)
    print(cell_topic_dist)

    sns.clustermap(cell_topic_dist, vmax=0.1)
    plt.show()
