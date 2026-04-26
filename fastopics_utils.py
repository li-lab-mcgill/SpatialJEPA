#%%
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
                    low_memory=True, low_memory_batch_size=2000,
                    verbose=True)

    # 4) Fit using counts as bag-of-genes and fixed embeddings as document embeddings.
    top_words, doc_topic_dist = model.fit_transform(
        dummy_docs,
        preset_doc_embeddings=fixed_embeddings,
    )
    return model, top_words, doc_topic_dist

def export_adata_with_fastopic(fastopic_model, cell_topic_dist, adata, filename):
    topic_by_genes = fastopic_model.get_beta()
    topic_weights = fastopic_model.get_topic_weights()
    adata.obsm['fastopic_cell_topic_dist'] = cell_topic_dist
    adata.varm['fastopic_genes_topic_weights'] = topic_by_genes.T
    adata.uns['fastopic_global_weights'] = topic_weights
    adata.write_h5ad(os.path.join(os.getenv('DATAPATH'), "aligned_data", filename))
    return

def run_fastopic(adata):
    fixed_embeddings = adata.obsm["MultiGATE"].copy()
    rna_counts = adata.layers["counts"].copy()
    fastopic_model, top_genes, cell_topic_dist = fit_fastopic(rna_counts, adata.var_names, fixed_embeddings)
    export_adata_with_fastopic(fastopic_model, cell_topic_dist, adata)
    return fastopic_model, cell_topic_dist

def visualize_fastopic_model(fastopic_model):
    fig = fastopic_model.visualize_topic(top_n=4)
    fig.show()
    fig = fastopic_model.visualize_topic_hierarchy()
    fig.show()
    fig = fastopic_model.visualize_topic_weights(top_n=20, height=500)
    fig.show()
    return

#%%
if __name__ == "__main__":
#%%
    import anndata as ad
    import os
    from dotenv import load_dotenv
    load_dotenv('/home/mcb/users/dmannk/BAKLAVA_base/BAKLAVA/.env')
    
    source_rna = ad.read_h5ad(os.path.join(os.getenv('DATAPATH'), "aligned_data", "source_rna_aligned_with_latents.h5ad"))
    source_atac = ad.read_h5ad(os.path.join(os.getenv('DATAPATH'), "aligned_data", "source_atac_aligned_with_latents.h5ad"))
    
    #%% fit FASTopic model
    source_rna_fastopic_model, source_rna_cell_topic_dist = run_fastopic(source_rna)
    source_atac_fastopic_model, source_atac_cell_topic_dist = run_fastopic(source_atac)

    #%% visualize FASTopic model
    visualize_fastopic_model(source_rna_fastopic_model)
    visualize_fastopic_model(source_atac_fastopic_model)

    #%% Extract export variables from FASTopic model
    export_adata_with_fastopic(source_rna_fastopic_model, source_rna_cell_topic_dist, source_rna, "source_rna_aligned_with_fastopic.h5ad")
    export_adata_with_fastopic(source_atac_fastopic_model, source_atac_cell_topic_dist, source_atac, "source_atac_aligned_with_fastopic.h5ad")
    