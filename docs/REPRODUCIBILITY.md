# Reproducing the manuscript

This page maps every figure of the manuscript to the step of `reproduce.sh` and the scripts that produce it.
It also records how each result was checked. The data needed are described in [DATA.md](DATA.md).

## How the code was verified

The scripts in this repository are the ones that produced the manuscript results. They were copied from the
development repositories with mechanical edits only:
- paths moved to settings;
- descriptive file, module and folder names;
- an offline reader for the MLflow run.

Every copied file is listed in [PROVENANCE.tsv](PROVENANCE.tsv) with its source, commit, checksums and the
edits applied.

Two checks were run on a Linux workstation (RTX 2080 Ti GPUs), in the environments that `envs/` describes,
inside a copy-on-write sandbox that recorded every file each step read:
1. The original scripts and this release were run on the original data. Their outputs were compared with
   the files placed in the manuscript.
2. `./reproduce.sh figures` was run from this release on the Zenodo deposit plus the public data listed in
   DATA.md. Every step ran from these files alone. The Check column below compares its outputs with the
   manuscript.
3. The five environments were rebuilt from scratch with `envs/install.sh` (conda 25.1, pip 26), and check 2 was
   repeated with them. Every figure was byte-identical to the one from the original environments except S10,
   where a few pixels differ by one intensity level. Of the 44 tables written, 42 were identical or agreed to
   within 5e-12 relative. Two differed slightly:
   - S10's RNA-PCA null (`direction_null_rnaPCA_*.csv`) agrees to 7.5e-7;
   - the logistic-regression sensitivity table of the Fig. 5 checks (`checks_B_donor.csv`) agrees to 3.7e-6
     in its p-values (for example 0.0289078 vs 0.0289079).

   The rebuilt `msi` environment also encodes the deposited mouse model (teacher and student, CPU) to
   bit-identical latents.

"Identical" below means byte-identical PNG files (and tables equal to within 1e-12, relative).

Retraining the mouse RNA–MSI model from this code, with the settings of `./reproduce.sh train_rna_msi_mouse`,
gave a model that is bit-identical to the deposited one on the same hardware and software: all weights of the
teacher, the student and the non-spatial model, and all embeddings. Use `RETRAIN_SUFFIX=_retrained` to keep
the deposited models. Other GPUs, drivers or library builds may differ in the last
digits.

## Figures

| Figure | Content | Step (`reproduce.sh`) | Script(s) | Check |
|---|---|---|---|---|
| Fig. 1 | Method overview | — | drawn by hand (`docs/figures/methodology.png`) | — |
| Fig. 2a–e | RNA–ATAC source, embeddings, spatial metrics | `analyses` | `rna_atac/analyze_rna_atac.py` | panels; assembled by hand |
| Fig. 2f–g | Held-out Multiome placement | `fig_rna_atac_heldout` | `rna_atac/plot_heldout_multiome_placement.py` | identical |
| Fig. 3a, c–d | Source–target clusters, PLS maps, loading enrichment | `fig_rna_atac_transfer` | `rna_atac/plot_transfer_panels.py` | identical |
| Fig. 3b | DE concordance | `fig_rna_atac_transfer` | `rna_atac/plot_transfer_clusters.py` | identical |
| Fig. 3e–f | *Mobp* and *Phactr1* expression and their linked peaks | `fig_rna_atac_transfer` | `rna_atac/plot_transfer_panels.py` (expression), `rna_atac/plot_linked_peak_maps.py` (peaks) | identical |
| Fig. 3g | Bivariate Moran's *I* of linked genes | `fig_rna_atac_transfer` | `rna_atac/plot_linked_peak_maps.py` | identical |
| Fig. 3 | Assembly | `fig_rna_atac_transfer` | `rna_atac/assemble_transfer_figure.py` | draft layout (with a marker-score panel that the final figure omits); final layout arranged by hand in Inkscape |
| Fig. 4 | Dopamine program | `fig_rna_msi_dopamine` | `rna_msi/plot_dopamine_program.py` | identical |
| Fig. 5 | Human caudate-head transfer | `fig_human_cah` | `human_cah/plot_cah_summary.py` (+ the `human_cah/astrocyte_*` steps) | panels and values identical; final layout arranged by hand |
| S1 | Integration metrics | `fig_rna_atac_supplement` | `rna_atac/plot_integration_metrics.py` | identical |
| S2 | Spatial-structure metrics | `fig_rna_atac_supplement` | `rna_atac/plot_spatial_structure.py` | values identical; manuscript PNG re-exported |
| S3 | Position recovery by latent | `fig_rna_atac_supplement` | `rna_atac/plot_position_recovery_by_latent.py` | identical |
| S4 | Placement vs published annotations | `fig_rna_atac_heldout` | `rna_atac/plot_heldout_vs_published_annotations.py` | identical |
| S5–S6 | ATAC PLS maps; PLS marker genes | `fig_rna_atac_transfer` | `rna_atac/plot_pls_maps_and_markers.py` | identical |
| S7–S8 | Held-out RNA–MSI animals | `fig_rna_msi_supplement` (tables: `ana_heldout_animals`) | `rna_msi/plot_heldout_animals.py` (tables: `student_teacher_diagnostics.py`, `heldout_animals_dopamine_structure.py`, `teacher_consistency.py`, `cluster_agreement.py`) | identical |
| S9 | Teacher/student-imputed dopamine | `fig_rna_msi_supplement` | `rna_msi/supplement/plot_imputed_dopamine_teacher_student.py` | identical |
| S10 | Dopamine dose-response control | `fig_rna_msi_supplement` | `rna_msi/supplement/null_random_latent_directions.py`, `null_rna_pca_directions.py`, `plot_dopamine_dose_response.py` | identical |
| S11 | Metabolite-weight null | `fig_rna_msi_supplement` | `rna_msi/supplement/null_ot_metabolite_weights.py` | identical |
| S12 | RNA-weight mixture | `fig_rna_msi_dopamine` | `rna_msi/plot_dopamine_program.py` | identical |
| S13 | Regulatory analysis | `fig_rna_msi_dopamine` | `rna_msi/plot_dopamine_program.py` | identical to the script output; manuscript PNG cropped |
| S14 | Motif maps | `fig_rna_msi_dopamine` | `rna_msi/supplement/plot_spatial_motif_maps.py` | identical |
| S15–S17 | Metabolite transfer and C2 in the caudate head | `fig_human_cah` | `human_cah/plot_metabolite_transfer_c1.py`, `plot_component_c2.py`, `plot_metabolite_transfer_c2.py` | identical |
| S18–S19 | Schematics | — | drawn in the supplement's LaTeX (TikZ) | — |

`reproduce.sh collect` copies each figure under the manuscript's file name into
`$SPATIALJEPA_MANUSCRIPT/figures/`.

Reported numbers sit in the CSV files written next to the figures:
- `*_values.csv` (held-out placement, Fig. 2f–g, S4, S7–S8);
- `cah_xenium_bivariate_moran_astro.csv` (Fig. 5i);
- `astro_c1_adnc/astro_c1_cache.pkl` (Fig. 5d odds ratio);
- `ecdf_area_by_latent.csv` (S2–S3);
- `pls_marker_*.csv` (S5–S6);
- `direction_null*.csv` (S10);
- `ot_direction_null.csv` (S11);
- `spatial_motif_map_stats.csv` (S14).

## Models

| Name (this release) | Earlier name | Data | Settings |
|---|---|---|---|
| `V11L12-109_B1_seed3` | `V11L12-109_B1__early2000_s3` | mouse RNA–MSI section V11L12-109_B1 | seed 3, 2,000-gene target panel, exact mini-batch neighbourhood, early-stopped teacher (best epoch 29), 1,000 epochs |
| `V11T17-102_seed0/1/2` | `V11T17-102__early_s0/1/2` | human RNA–MSI sections A1, C1, D1 | seeds 0–2, same settings; seed 0 for Fig. 5, seeds 1–2 for replicates |
| `V11L12-109_B1` | (unchanged) | mouse section | reference run: NMF dopamine signature and target containers |
| RNA–ATAC `20260505_190037` | MLflow run `7c2399d1…` | P22 spatial RNA–ATAC → 10x Multiome | MultiGATE backbone; 3,000 epochs, spatial teacher, identity-graph student and non-spatial model, knowledge distillation 0.9 OT / 0.1 KL; all 46 parameters in the deposited `run.json`. Trained with MultiGATE commit `a17e77a`; later commits changed only the scib evaluation metrics and removed an unused TensorFlow model |

Earlier configurations (`fix…`, `induced…`) appear in the code as alternative runs but are not part of the
manuscript. Two files of earlier runs are read for indices only:
- `human_cah_transfer/fix_s0/cah/snrna_mofa.h5mu`;
- `human_cah_transfer/fix_s0/cah/snrna_spatialmeta_aligned.h5mu`.

## Renamed folders

| This release | Earlier |
|---|---|
| `outputs/rna_msi_runs` | `outputs/spatialjepa_fig5_rerun` |
| `outputs/human_cah_transfer` | `outputs/sea_ad_cah_fixed` |
| `outputs/rna_atac_figures` | `outputs/manuscript_fig34_revisions_2026-10` |
| `outputs/rna_atac_transfer_clusters` | `outputs/fig4_target_leiden_20261014` |
| `outputs/MultiGATE/rna_atac_analysis` | `outputs/MultiGATE/co_embed` |
| `outputs/rna_atac_models/20260505_190037` | `mlflow_tracking/MultiGATE/multigate_mouse_brain_live_zeroshot/7c2399d1…` |

## Manual and recovered steps

- **Hand-assembled figures.** Figs. 1–3 and the final layout of Fig. 5 were assembled in Inkscape from the
  script outputs. S2 and S13 in the manuscript are re-exports or crops of the script outputs.
- **Recovered human-caudate helpers.** Five helper scripts had not been kept with the analysis code. They were
  recovered from an archived working folder:
  `human_cah/build_cell_table.py`, `build_donor_table.py`, `add_early_cell_columns.py`,
  `astrocyte_c1_checks.py`, `astrocyte_c1_count_adjusted.py`. With them, the release reproduces Fig. 5 and
  S15–S17 identically.
- **Two format and loading edits beyond renaming.**
  - `rna_atac/plot_linked_peak_maps.py` now also writes the SVG that the Fig. 3 assembly reads.
  - `rna_atac/analyze_rna_atac.py` reads an exported MLflow run when `SPATIALJEPA_RNA_ATAC_RUN_DIR` is set
    (`spatialjepa/mlflow_offline.py`).
