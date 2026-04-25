from fastopic import FASTopic

import scipy.sparse as sp
import numpy as np

__all__ = ["NumericPreprocess"]

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

def fit_fastopic(X, genes, num_topics=50):
    # 1) Get numeric matrix and gene names
    X = rna_counts              # cells × genes
    genes = source_rna.var_names  # iterable of gene names

    # 2) Wrap in NumericPreprocess
    preprocess = NumericPreprocess(X, genes)

    # 3) Build dummy docs (ignored by our preprocess)
    n_cells = X.shape[0]
    dummy_docs = ["x"] * n_cells   # any constant string

    # 4) Instantiate FASTopic
    num_topics = 50
    model = FASTopic(num_topics, preprocess=preprocess,
                    low_memory=True, low_memory_batch_size=2000)

    # 5) Fit model; this will call NumericPreprocess.preprocess(dummy_docs),
    # use your X as train_bow, and ignore the text content of dummy_docs.
    top_words, doc_topic_dist = model.fit_transform(dummy_docs)
    return top_words, doc_topic_dist

if __name__ == "__main__":
    import scanpy as sc
    source_rna = sc.read_h5ad("/home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain.h5ad")
    rna_counts = source_rna.layers["counts"]
    top_words, doc_topic_dist = fit_fastopic(rna_counts, source_rna.var_names)
    print(top_words)
    print(doc_topic_dist)