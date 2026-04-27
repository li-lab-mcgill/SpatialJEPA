#%%
from fastopic import FASTopic

import torch
import gc
import scipy.sparse as sp
import numpy as np
import os

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

def send_to_cpu(model):
    model.model = model.model.to("cpu")
    model.device = "cpu"
    if hasattr(model, "train_doc_embeddings"):
        model.train_doc_embeddings = model.train_doc_embeddings.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    return model

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
                    low_memory=False, low_memory_batch_size=2000,
                    device="cuda",
                    verbose=True)

    # 4) Fit using counts as bag-of-genes and fixed embeddings as document embeddings.
    top_words, doc_topic_dist = model.fit_transform(
        dummy_docs,
        preset_doc_embeddings=fixed_embeddings,
        epochs=1000,
    )

    model = send_to_cpu(model)

    return model, top_words, doc_topic_dist

def run_fastopic(adata):
    fixed_embeddings = adata.obsm["MultiGATE_source_aligned"].copy()
    rna_counts = adata.layers["counts"].copy()
    fastopic_model, top_genes, cell_topic_dist = fit_fastopic(rna_counts, adata.var_names, fixed_embeddings)
    return fastopic_model, cell_topic_dist

def export_adata_with_fastopic(fastopic_model, cell_topic_dist, adata, filename):
    topic_by_genes = fastopic_model.get_beta()
    topic_weights = fastopic_model.get_topic_weights()
    topic_embeddings = fastopic_model.topic_embeddings
    gene_embeddings = fastopic_model.word_embeddings
    beta = fastopic_model.get_beta()

    uns = {
        'global_weights': topic_weights,
        'topic_embeddings': topic_embeddings,
        'beta': beta
    }

    adata.obsm['fastopic_cell_topic_dist'] = cell_topic_dist
    adata.varm['fastopic_genes_topic_weights'] = topic_by_genes.T
    adata.varm['fastopic_gene_embeddings'] = gene_embeddings

    adata.uns['fastopic'] = uns

    adata.write_h5ad(os.path.join(os.getenv('DATAPATH'), "aligned_data", filename))
    return

def visualize_fastopic_model(fastopic_model, domain_name):
    outpath = os.path.join(os.getenv('OUTPATH'), "fastopic_visualizations", domain_name)
    os.makedirs(outpath, exist_ok=True)

    def _save_plotly_figure(fig, out_png_path):
        try:
            fig.write_image(out_png_path)
        except Exception as exc:
            out_html_path = os.path.splitext(out_png_path)[0] + ".html"
            print(f"[WARN] Could not export PNG ({out_png_path}): {exc}")
            print(f"[WARN] Falling back to HTML export: {out_html_path}")
            fig.write_html(out_html_path)

    fig = fastopic_model.visualize_topic(top_n=4)
    _save_plotly_figure(fig, os.path.join(outpath, "fastopic_topic_visualization.png"))

    fig = fastopic_model.visualize_topic_hierarchy()
    _save_plotly_figure(fig, os.path.join(outpath, "fastopic_topic_hierarchy_visualization.png"))

    fig = fastopic_model.visualize_topic_weights(top_n=20, height=500)
    _save_plotly_figure(fig, os.path.join(outpath, "fastopic_topic_weights_visualization.png"))

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

    target_rna = ad.read_h5ad(os.path.join(os.getenv('DATAPATH'), "aligned_data", "target_rna_aligned_with_latents.h5ad"))
    target_atac = ad.read_h5ad(os.path.join(os.getenv('DATAPATH'), "aligned_data", "target_atac_aligned_with_latents.h5ad"))
    
    #%% fit FASTopic model
    source_rna_fastopic_model, source_rna_cell_topic_dist = run_fastopic(source_rna)
    source_atac_fastopic_model, source_atac_cell_topic_dist = run_fastopic(source_atac)

    target_rna_fastopic_model, target_rna_cell_topic_dist = run_fastopic(target_rna)
    target_atac_fastopic_model, target_atac_cell_topic_dist = run_fastopic(target_atac)

    #%% visualize FASTopic model
    visualize_fastopic_model(source_rna_fastopic_model, "source_rna")
    visualize_fastopic_model(source_atac_fastopic_model, "source_atac")

    visualize_fastopic_model(target_rna_fastopic_model, "target_rna")
    visualize_fastopic_model(target_atac_fastopic_model, "target_atac")
    
    #%% Extract export variables from FASTopic model
    export_adata_with_fastopic(source_rna_fastopic_model, source_rna_cell_topic_dist, source_rna, "source_rna_aligned_with_fastopic.h5ad")
    export_adata_with_fastopic(source_atac_fastopic_model, source_atac_cell_topic_dist, source_atac, "source_atac_aligned_with_fastopic.h5ad")

    export_adata_with_fastopic(target_rna_fastopic_model, target_rna_cell_topic_dist, target_rna, "target_rna_aligned_with_fastopic.h5ad")
    export_adata_with_fastopic(target_atac_fastopic_model, target_atac_cell_topic_dist, target_atac, "target_atac_aligned_with_fastopic.h5ad")