# Data

All scripts read and write under one folder, `SPATIALJEPA_HOME` (set in `configs/spatialjepa.env`), which
holds `data/` and `outputs/`. It is assembled from two sources:

1. **The Zenodo deposit** (DOI: [10.5281/zenodo.23933515](https://doi.org/10.5281/zenodo.23933515)): trained models, processed inputs and the intermediate results
   that the figures are drawn from. Unpack its archives into `SPATIALJEPA_HOME`. `MANIFEST.tsv` lists every
   file with its size, SHA-256 checksum and the step that reads it.
2. **Public data**, downloaded from the original sources (below), for the files the deposit does not
   redistribute.

No new sequencing, imaging or mass-spectrometry data were generated.

## The deposit (about 47 GB)

| Folder | Content |
|---|---|
| `data/aligned_data/` | P22 spatial RNA–ATAC (source) and 10x Multiome (target) on the shared feature index, with the model embeddings (`*_with_latents.h5ad`); MultiVI input (`multivi_mdata.h5mu`) |
| `data/vicari_2023/h5mu_export/` | SCTransform-normalised RNA–MSI sections used for training (mouse V11L12-109_B1; human V11T17-102 A1, C1, D1), and the Mendeley sample sheet |
| `outputs/rna_atac_models/20260505_190037/` | RNA–ATAC models of MLflow run 20260505_190037: teacher, student, non-spatial (`artifacts/models/`), domain splits, gene–peak attention; the run record exported from MLflow (`run.json`: parameters and tags; `metrics/`: metric histories) |
| `outputs/MultiGATE/rna_atac_analysis/20260505_190037/` | RNA–ATAC analysis results: integration-metric inputs, UMAPs, held-out Multiome placement |
| `outputs/MultiGATE/attention_analysis/` | Gene–peak links retained from the attention analysis |
| `outputs/rna_atac_transfer_clusters/`, `outputs/rna_atac_figures/` | Leiden labels of the transferred clusters; bivariate Moran's *I* of linked genes |
| `outputs/spatialjepa_models/` | RNA–MSI models: `V11L12-109_B1_seed3` (mouse, Fig. 4), `V11T17-102_seed0/1/2` (human, Fig. 5), `V11L12-109_B1` (reference run). Each holds `teacher.pt`, `student.pt`, `nonspatial.pt`, `student_pls_decoder.joblib`, `early_stopping.json` and, where needed, the training matrix `joint_adata.h5ad` |
| `outputs/spatialjepa_projection/` | Encoded targets (P22 spatial and Multiome) of the mouse model and the reference run, with their MOFA+ models (`*_mofa_model.hdf5`, opened read-write by `mofax`) and the NMF dopamine signature. The figure code loads the MOFA+ models only for the gene and peak sets it plots; no MOFA+ result appears in the manuscript |
| `outputs/rna_msi_runs/shared_components/` | Shared PLS components, chromVAR deviations and motif caches of the mouse model |
| `outputs/human_cah_transfer/` | SEA-AD caudate-head transfer: aligned nuclei, per-donor embeddings, PLS components (`seed0`; `seed1-2` and `fix_s*` results for comparisons and indices) |
| `outputs/spatialjepa_benchmark/` | Held-out-animal benchmark results (S7–S8) |
| `outputs/multigrate_mouse_sma/`, `outputs/multivi_mouse_brain_spatial_rna_atac*/` | Baseline models and embeddings |
| `outputs/target_gene_rankings/` | Target-derived gene rankings that restrict the RNA–MSI training panel |
| `outputs/manuscript_templates/figures/` | Manuscript layout that the S2 figure script draws onto (`reproduce.sh` writes the figures to `outputs/manuscript/figures/`) |

## Public data

| Data | Source | Used for | Path in `SPATIALJEPA_HOME` |
|---|---|---|---|
| P22 mouse brain spatial RNA–ATAC (Zhang et al. 2023) | GEO [GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055) | rebuilding `data/aligned_data` (`rna_atac/align_features.py`) | `data/Spatial_ATAC_RNA/` |
| 10x Genomics mouse brain Multiome | [10x Genomics datasets](https://www.10xgenomics.com/datasets/mouse-brain-nuclei-isolated-with-chromium-nuclei-isolation-kit-saltyez-protocol-and-10x-complex-tissue-dp-ct-sorted-and-ct-unsorted-1-standard) | rebuilding `data/aligned_data` | `data/10x_mouse_brain/` |
| Spatial RNA–MSI (Vicari et al. 2024) | Mendeley Data [doi:10.17632/w7nw4km7xd.1](https://doi.org/10.17632/w7nw4km7xd.1); GEO GSE232910; SciLifeLab Figshare doi:10.17044/scilifelab.22770161 | rebuilding `data/vicari_2023/h5mu_export` (`rna_msi/export_seurat_to_mtx.R`, `build_mudata_from_mtx.py`, `sctransform_rna.py`) | `data/vicari_2023/` |
| SEA-AD caudate head snRNA-seq and Xenium | AWS Open Data: `s3://sea-ad-single-cell-profiling/Caudate_Nucleus/`, `s3://sea-ad-spatial-transcriptomics/caudate_nucleus/` | Fig. 5, S15–S17 (`human_cah/prepare_snrna.py` keeps the cells "Used in analysis") | `data/SEA_AD/sea_ad_cah/CaH_final_nuclei.2025-12-29.h5ad`, `CaH_Xenium_final.2026-01-07.h5ad` |
| SEA-AD neuropathology and CPS tables | [SEA-AD](https://portal.brain-map.org/explore/seattle-alzheimers-disease) donor metadata (quantitative neuropathology; global and local CPS) | Fig. 5d donor table (`human_cah/build_donor_table.py`) | `data/SEA_AD/*.xlsx` |
| Mouse cerebrum cCRE atlas (Li et al. 2021, CATlas) | Supplementary Table 8 ("Cell type assignment of cCREs") | Fig. 4/S13 cCRE enrichment | `data/CATlas/supplementary_tables/` |
| JASPAR motifs | JASPAR (via `pyjaspar`) | motif scans | installed with `envs/spatialjepa-motifs.yml` |

The SEA-AD snRNA-seq file is about 100 GB after filtering; the astrocyte scripts read only the astrocyte rows.

## Rebuilding the processed inputs

The deposit's processed inputs can be rebuilt from the public data:
- `rna_atac/align_features.py` builds the shared RNA–ATAC feature index;
- the `rna_msi/` export scripts and `sctransform_rna.py` build the RNA–MSI sections. They start from the
  co-registered Seurat objects of Vicari et al. (their `MSI_SRT_*.Rmd` workflows).

The `*_with_latents.h5ad` files hold the RNA–ATAC embeddings. They were written by the
"write to disk" block of `rna_atac/analyze_rna_atac.py`, which is commented out so that the analysis does not
overwrite them; uncomment it to rebuild them.
