import os
import matplotlib.pyplot as plt

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