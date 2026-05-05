#%%
from fastopic import FASTopic
from fastopic._utils import Dataset

import torch
import gc
import scipy.sparse as sp
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm

__all__ = [
    "NumericPreprocess",
    "fit_fastopic",
    "fit_fastopic_tied",
    "run_fastopic",
    "run_fastopic_tied",
]

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
        epochs=10000,
    )

    model = send_to_cpu(model)

    return model, top_words, doc_topic_dist

def run_fastopic(adata, counts_key):
    fixed_embeddings = adata.obsm["MultiGATE_source_aligned"].copy()
    rna_counts = adata.layers[counts_key].copy()
    fastopic_model, top_genes, cell_topic_dist = fit_fastopic(rna_counts, adata.var_names, fixed_embeddings, num_topics=20)
    return fastopic_model, cell_topic_dist

def fit_fastopic_tied(
    X1, genes1, fixed_embeddings1,
    X2, genes2, fixed_embeddings2,
    num_topics=20,
    epochs=10000,
    learning_rate=0.002,
    DT_alpha=3.0,
    TW_alpha=2.0,
    theta_temp=1.0,
    device="cuda",
    verbose=True,
    log_interval=10,
):
    """Train two FASTopic models simultaneously with shared (tied) topic embeddings.

    The two modalities share `topic_embeddings` (and the topic-weights parameter used inside
    `DT_ETP`) but keep modality-specific `word_embeddings` / `word_weights`. Both modalities
    must share the same document-embedding dimension, which is the case here because both
    use MultiGATE latents.

    Returns:
        (model1, top_words1, train_theta1), (model2, top_words2, train_theta2)
    """
    fixed_embeddings1 = np.asarray(fixed_embeddings1, dtype=np.float32)
    fixed_embeddings2 = np.asarray(fixed_embeddings2, dtype=np.float32)

    if fixed_embeddings1.shape[1] != fixed_embeddings2.shape[1]:
        raise ValueError(
            "Both modalities must share the document-embedding dimension to tie topic embeddings: "
            f"{fixed_embeddings1.shape[1]} != {fixed_embeddings2.shape[1]}"
        )

    pp1 = NumericPreprocess(X1, genes1)
    pp2 = NumericPreprocess(X2, genes2)

    if pp1.X_csr.shape[0] != fixed_embeddings1.shape[0]:
        raise ValueError(
            f"X1 cell count does not match fixed_embeddings1: {pp1.X_csr.shape[0]} != {fixed_embeddings1.shape[0]}"
        )
    if pp2.X_csr.shape[0] != fixed_embeddings2.shape[0]:
        raise ValueError(
            f"X2 cell count does not match fixed_embeddings2: {pp2.X_csr.shape[0]} != {fixed_embeddings2.shape[0]}"
        )
    if pp1.X_csr.shape[1] != len(pp1.vocab):
        raise ValueError("X1 column count does not match genes1 length.")
    if pp2.X_csr.shape[1] != len(pp2.vocab):
        raise ValueError("X2 column count does not match genes2 length.")

    n1 = pp1.X_csr.shape[0]
    n2 = pp2.X_csr.shape[0]
    embed_size = fixed_embeddings1.shape[1]

    model1 = FASTopic(
        num_topics, preprocess=pp1,
        doc_embed_model=PresetOnlyDocEmbedder(),
        DT_alpha=DT_alpha, TW_alpha=TW_alpha, theta_temp=theta_temp,
        device=device, low_memory=False, low_memory_batch_size=2000,
        verbose=verbose, log_interval=log_interval,
    )
    model2 = FASTopic(
        num_topics, preprocess=pp2,
        doc_embed_model=PresetOnlyDocEmbedder(),
        DT_alpha=DT_alpha, TW_alpha=TW_alpha, theta_temp=theta_temp,
        device=device, low_memory=False, low_memory_batch_size=2000,
        verbose=verbose, log_interval=log_interval,
    )

    # Replicate FASTopic.fit_transform's setup so we can take over the training loop.
    dataset1 = Dataset(
        ["x"] * n1, doc_embedder=None, preprocess=pp1,
        batch_size=n1, device=device, low_memory=False,
        preset_doc_embeddings=fixed_embeddings1,
    )
    dataset2 = Dataset(
        ["x"] * n2, doc_embedder=None, preprocess=pp2,
        batch_size=n2, device=device, low_memory=False,
        preset_doc_embeddings=fixed_embeddings2,
    )

    model1.batch_size = n1
    model2.batch_size = n2

    model1.train_doc_embeddings = torch.as_tensor(dataset1.doc_embeddings).to(device)
    model2.train_doc_embeddings = torch.as_tensor(dataset2.doc_embeddings).to(device)

    model1.model.init(dataset1.vocab_size, embed_size)
    model2.model.init(dataset2.vocab_size, embed_size)

    model1.vocab = dataset1.vocab
    model2.vocab = dataset2.vocab

    model1.model = model1.model.to(device)
    model2.model = model2.model.to(device)

    # Tie shared parameters AFTER `.to(device)` because Module.to may produce new
    # Parameter objects, breaking earlier references. The "live" topic-weights parameter
    # used in the forward pass lives at `DT_ETP.b_dist` (the parent's `topic_weights`
    # attribute is, by quirk of FASTopic's init, an orphan after .to(device) and gets no
    # gradient). Rebind that one explicitly.
    model2.model.topic_embeddings = model1.model.topic_embeddings
    model2.model.DT_ETP.b_dist = model1.model.DT_ETP.b_dist
    model2.model.topic_weights = model1.model.topic_weights

    seen_ids = set()
    shared_params = []
    for p in list(model1.model.parameters()) + list(model2.model.parameters()):
        if not p.requires_grad or id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        shared_params.append(p)

    optimizer = torch.optim.Adam(shared_params, lr=learning_rate)

    model1.model.train()
    model2.model.train()

    for epoch in tqdm(range(1, epochs + 1), desc="Training tied FASTopic"):
        loss_rst = defaultdict(float)
        for (bow1, emb1), (bow2, emb2) in zip(dataset1.dataloader, dataset2.dataloader):
            rst1 = model1.model(bow1, emb1)
            rst2 = model2.model(bow2, emb2)
            loss = rst1["loss"] + rst2["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_rst["loss1"] += rst1["loss"].item() * bow1.shape[0]
            loss_rst["loss2"] += rst2["loss"].item() * bow2.shape[0]

        if verbose and (epoch % log_interval == 0):
            print(
                f"[tied FASTopic] epoch {epoch:03d} "
                f"loss1: {loss_rst['loss1']/n1:.3f} "
                f"loss2: {loss_rst['loss2']/n2:.3f}"
            )

    results = []
    for m in (model1, model2):
        m.beta = m.get_beta()
        m.top_words = m.get_top_words(m.num_top_words)
        m.train_theta = m.transform(doc_embeddings=m.train_doc_embeddings)
        results.append((m, m.top_words, m.train_theta))

    return results[0], results[1]


def run_fastopic_tied(rna_adata, atac_adata, counts_key, num_topics=20):
    """Fit FASTopic on RNA and ATAC adata jointly with shared topic embeddings."""
    rna_emb = rna_adata.obsm["MultiGATE_source_aligned"].copy()
    atac_emb = atac_adata.obsm["MultiGATE_source_aligned"].copy()
    rna_counts = rna_adata.layers[counts_key].copy()
    atac_counts = atac_adata.layers[counts_key].copy()

    (rna_model, _, rna_theta), (atac_model, _, atac_theta) = fit_fastopic_tied(
        rna_counts, rna_adata.var_names, rna_emb,
        atac_counts, atac_adata.var_names, atac_emb,
        num_topics=num_topics,
    )

    rna_model = send_to_cpu(rna_model)
    atac_model = send_to_cpu(atac_model)

    return rna_model, rna_theta, atac_model, atac_theta


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

    ## subsample genes & peaks
    n_genes = source_rna.var['highly_variable_rank'].max().item()  # 1000
    source_rna = source_rna[:, source_rna.var['highly_variable_rank'].le(n_genes)].copy()
    target_rna = target_rna[:, target_rna.var['highly_variable_rank'].le(n_genes)].copy()

    gp_net = source_rna.uns['gene_peak_Net']
    keep_peaks = gp_net[gp_net['Gene'].isin(source_rna.var_names)]['Peak'].unique()

    source_atac = source_atac[:, source_atac.var_names.isin(keep_peaks)].copy()
    target_atac = target_atac[:, target_atac.var_names.isin(keep_peaks)].copy()

    source_atac = source_atac[:, source_atac.var['highly_variable_rank'].nsmallest(source_rna.n_vars).index].copy()
    target_atac = target_atac[:, target_atac.var['highly_variable_rank'].nsmallest(target_rna.n_vars).index].copy()

    print(f"Subsampled to {source_rna.shape[1]} genes and {source_atac.shape[1]} peaks.")
    
    #%% fit FASTopic model

    counts_key = "SCT_counts"
    tied_topics = False

    if tied_topics:
        # with tied (shared) topic embeddings within each modality
        (
            source_rna_fastopic_model,
            source_rna_cell_topic_dist,
            source_atac_fastopic_model,
            source_atac_cell_topic_dist,
        ) = run_fastopic_tied(source_rna, source_atac, counts_key)

        (
            target_rna_fastopic_model,
            target_rna_cell_topic_dist,
            target_atac_fastopic_model,
            target_atac_cell_topic_dist,
        ) = run_fastopic_tied(target_rna, target_atac, counts_key)
    else:
        # with separate topic embeddings for each modality
        source_rna_fastopic_model, source_rna_cell_topic_dist = run_fastopic(source_rna, counts_key)
        source_atac_fastopic_model, source_atac_cell_topic_dist = run_fastopic(source_atac, "counts")

        target_rna_fastopic_model, target_rna_cell_topic_dist = run_fastopic(target_rna, counts_key)
        target_atac_fastopic_model, target_atac_cell_topic_dist = run_fastopic(target_atac, "counts")

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
# %%
