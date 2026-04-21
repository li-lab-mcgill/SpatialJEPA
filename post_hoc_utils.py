import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gene_peak_attention_utils import (
    add_gene_and_peak_columns,
    assign_regulatory_region,
    compute_gene_peak_distance,
    extract_peak_gene_connections,
    filter_by_attention_threshold,
    get_gmm_attention_threshold,
    merge_with_gene_annotations,
    parse_gtf_file,
    plot_attention_distribution,
    plot_distance_distribution,
    save_attention_outputs,
)

def run_gene_peak_attention_tutorial(
    peak_gene_attention=None,
    adata_rna=None,
    adata_atac=None,
    gtf_file=None,
    output_dir=None,
    index_layout="gene_first",
    promoter_window_bp=2000,
    min_attention=0.0,
    save_results=True,
    show_plots=True,
    random_state=0,
):

    """Run tutorial-style gene-peak attention analysis using reusable utilities."""
    if peak_gene_attention is None:
        peak_gene_attention = globals().get("source_peak_gene_attention")
    if peak_gene_attention is None:
        peak_gene_attention = globals().get("target_peak_gene_attention")
    if peak_gene_attention is None:
        raise ValueError("No attention matrix found. Pass peak_gene_attention explicitly.")

    if adata_rna is None:
        adata_rna = globals().get("source_rna")
    if adata_rna is None:
        adata_rna = globals().get("target_rna")
    if adata_rna is None:
        raise ValueError("No RNA AnnData found. Pass adata_rna explicitly.")

    if adata_atac is None:
        adata_atac = globals().get("source_atac")
    if adata_atac is None:
        adata_atac = globals().get("target_atac")
    if adata_atac is None:
        raise ValueError("No ATAC AnnData found. Pass adata_atac explicitly.")

    if gtf_file is None:
        gtf_file = globals().get("gtf_path")
    if gtf_file is None and os.getenv("DATAPATH"):
        gtf_file = os.path.join(
            os.getenv("DATAPATH"),
            "gene_annotations",
            "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
        )
    if gtf_file is None or not os.path.exists(gtf_file):
        raise FileNotFoundError(f"GTF annotation file not found: {gtf_file}")

    if output_dir is None:
        OUTPATH = os.getenv("OUTPATH")
        output_dir = os.path.join(OUTPATH, "MultiGATE", "attention_analysis")
        os.makedirs(output_dir, exist_ok=True)

    peak_gene_df = extract_peak_gene_connections(
        peak_gene_attention=peak_gene_attention,
        peaks=adata_atac.var_names,
        genes=adata_rna.var_names,
        index_layout=index_layout,
        min_attention=min_attention,
    )
    peak_gene_df = add_gene_and_peak_columns(peak_gene_df)

    gtf_df = parse_gtf_file(gtf_file)
    merged_df = merge_with_gene_annotations(peak_gene_df, gtf_df)
    merged_df = compute_gene_peak_distance(merged_df)
    merged_df = assign_regulatory_region(merged_df, promoter_window_bp=promoter_window_bp)

    gmm_result = get_gmm_attention_threshold(
        merged_df["Attention"].to_numpy(),
        random_state=random_state,
    )
    threshold = gmm_result.threshold
    merged_df_threshold = filter_by_attention_threshold(merged_df, threshold)
    merged_df_threshold = assign_regulatory_region(
        merged_df_threshold,
        promoter_window_bp=promoter_window_bp,
    )

    if show_plots:
        plot_attention_distribution(
            merged_df["Attention"].to_numpy(),
            threshold=threshold,
            gmm=gmm_result.model,
        )
        plot_distance_distribution(merged_df_threshold["distance"].to_numpy())
        plt.show()

    if save_results:
        save_attention_outputs(
            merged_df=merged_df,
            merged_df_threshold=merged_df_threshold,
            threshold=threshold,
            output_dir=output_dir,
        )

    summary = {
        "output_dir": output_dir,
        "pairs_total": int(merged_df.shape[0]),
        "pairs_above_threshold": int(merged_df_threshold.shape[0]),
        "attention_threshold": float(threshold),
    }

    globals()["peak_gene_df"] = peak_gene_df
    globals()["merged_df"] = merged_df
    globals()["merged_df_threshold"] = merged_df_threshold
    globals()["attention_threshold"] = threshold

    print("Gene-peak attention analysis complete:", summary)
    return summary


def topic_betas_hallmark_gsea_mouse(
    beta_rna: np.ndarray,
    gene_names: Sequence[str],
    topic_indices: Sequence[int] | np.ndarray,
    *,
    datapath: str | None = None,
    ortholog_filename: str = "human_mouse_gene_orthologs.csv",
    times: int = 1000,
    seed: int = 42,
    tmin: int = 3,
    row_id_prefix: str = "topic",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    For each topic row in ``beta_rna[topic_indices]``, rank genes by beta (high to low),
    pair (gene, beta), translate MSigDB Hallmark (human) to mouse via a local ortholog
    table, build a decoupler GSEA input matrix, and run ``dc.mt.gsea``.

    Parameters
    ----------
    beta_rna
        Topic-by-gene matrix (all topics), same column order as ``gene_names``.
    gene_names
        RNA feature names aligned to columns of ``beta_rna``.
    topic_indices
        Row indices into ``beta_rna`` to analyze (e.g. ``top_active_topics``).
    datapath
        Base data directory (defaults to ``os.environ["DATAPATH"]``).
    ortholog_filename
        CSV under ``<datapath>/gene_annotations/`` with columns
        ``Gene name`` and ``Mouse gene name`` (as in multigate_co_embed).

    Returns
    -------
    dict with keys: ``top_active_genes``, ``ranked_gene_indices``, ``gsea_input``,
    ``net``, ``gsea_scores``, ``gsea_padj``, ``gsea_long``, ``top_terms_per_row``.
    """
    import decoupler as dc

    if datapath is None:
        datapath = os.environ.get("DATAPATH")
    if not datapath:
        raise ValueError("datapath is None and DATAPATH is not set in the environment.")

    gene_names_arr = np.asarray(gene_names, dtype=str)
    if gene_names_arr.ndim != 1 or gene_names_arr.shape[0] != beta_rna.shape[1]:
        raise ValueError(
            f"gene_names length {gene_names_arr.shape[0]} must match beta_rna.shape[1]={beta_rna.shape[1]}."
        )

    topic_indices = np.asarray(topic_indices)
    if topic_indices.ndim != 1:
        raise ValueError("topic_indices must be a 1-D array or sequence.")
    beta_sub = np.asarray(beta_rna)[topic_indices]

    ranked_idx = np.argsort(beta_sub, axis=1)[:, ::-1]
    ranked_genes = gene_names_arr[ranked_idx]
    ranked_betas = np.take_along_axis(beta_sub, ranked_idx, axis=1)
    top_active_genes = [
        list(zip(genes_row, betas_row))
        for genes_row, betas_row in zip(ranked_genes, ranked_betas)
    ]

    hallmark_human = dc.op.hallmark()
    map_path = os.path.join(datapath, "gene_annotations", ortholog_filename)
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f"Ortholog mapping file not found: {map_path}")

    map_df = pd.read_csv(map_path)
    map_df = (
        map_df.rename(columns={"Gene name": "target_human", "Mouse gene name": "target"})[
            ["target_human", "target"]
        ]
        .dropna()
        .drop_duplicates()
    )
    net = (
        hallmark_human.rename(columns={"target": "target_human"})
        .merge(map_df, on="target_human", how="inner")[["source", "target"]]
        .drop_duplicates()
    )

    all_genes = pd.Index(gene_names_arr)
    net = net[net["target"].isin(all_genes)].copy()

    row_ids = [f"{row_id_prefix}_{t}" for t in topic_indices]
    gsea_input = pd.DataFrame(0.0, index=row_ids, columns=all_genes)

    for row_id, gene_beta_pairs in zip(row_ids, top_active_genes):
        pairs = [(str(g), float(b)) for g, b in gene_beta_pairs if pd.notna(g) and pd.notna(b)]
        if not pairs:
            continue
        genes = pd.Index([g for g, _ in pairs])
        betas = np.array([b for _, b in pairs], dtype=float)
        keep = genes.isin(all_genes)
        genes = genes[keep]
        betas = betas[keep]
        dedup = ~genes.duplicated()
        genes = genes[dedup]
        betas = betas[dedup]
        gsea_input.loc[row_id, genes] = betas

    gsea_input = gsea_input.loc[(gsea_input != 0).any(axis=1)]
    if gsea_input.shape[0] == 0:
        raise ValueError("GSEA input has no non-empty rows after filtering.")

    gsea_scores, gsea_padj = dc.mt.gsea(
        gsea_input,
        net,
        tmin=tmin,
        times=times,
        seed=seed,
        verbose=verbose,
    )

    gsea_long = (
        gsea_scores.stack()
        .rename("nes")
        .to_frame()
        .join(gsea_padj.stack().rename("padj"))
        .reset_index()
        .rename(columns={"level_0": "row_id", "level_1": "pathway"})
        .sort_values(["row_id", "padj", "nes"], ascending=[True, True, False])
    )
    top_terms_per_row = gsea_long.groupby("row_id", group_keys=False).head(10)

    return {
        "top_active_genes": top_active_genes,
        "ranked_gene_indices": ranked_idx,
        "gsea_input": gsea_input,
        "net": net,
        "gsea_scores": gsea_scores,
        "gsea_padj": gsea_padj,
        "gsea_long": gsea_long,
        "top_terms_per_row": top_terms_per_row,
    }

