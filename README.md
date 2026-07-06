# SpatialJEPA

`SpatialJEPA` is a JEPA-inspired framework for transferring spatial context from spatial multiomics data to dissociated single-cell multiome data. It uses a MultiGATE-based spatial teacher trained with the full neighborhood graph and distills its latent representation into a student trained with a self-only graph, allowing the student to operate when spatial coordinates are unavailable.

In mouse brain RNA-ATAC experiments, SpatialJEPA supports source-to-target alignment, recovers spatially organized transcriptomic and chromatin-accessibility programs, and produces representations concordant with ligand--receptor pathway structure.

<img src="fig/methodology.png" alt="SpatialJEPA methodology" style="background-color: white; margin: 1cm;">

<details>
<summary><strong>Installation</strong></summary>

Follow these steps to install `SpatialJEPA` in a dedicated Conda environment named `spatialjepa_env`.

```shell
conda create -n spatialjepa_env python=3.7 -y
conda activate spatialjepa_env

# Required system tool for Cal_gene_peak_Net_new
conda install -c bioconda bedtools -y

# Core Python dependencies
conda install scikit-learn pandas scanpy jupyterlab tqdm matplotlib -y
conda install -c conda-forge networkx louvain -y
conda install -c bioconda pybedtools -y

# PyTorch (Python 3.7-compatible range)
pip install "torch>=1.10.0,<1.14.0"

export CFLAGS="-std=c99"
pip install rpy2
pip install -e .
```

#### Detailed instructions

1. **Create a Conda environment**

   ```bash
   conda create -n spatialjepa_env python=3.7 -y
   ```

2. **Activate the environment**

   ```bash
   conda activate spatialjepa_env
   ```

3. **Install `bedtools` (required)**

   `Cal_gene_peak_Net_new` requires `bedtools` and `sortBed` to be available on `PATH`.

   ```bash
   conda install -c bioconda bedtools -y
   ```

4. **Install Python dependencies**

   ```bash
   conda install scikit-learn pandas scanpy jupyterlab tqdm matplotlib -y
   conda install -c conda-forge networkx louvain -y
   conda install -c bioconda pybedtools -y
   pip install "torch>=1.10.0,<1.14.0"
   ```

5. **Install the optional R bridge and `SpatialJEPA` (editable install)**

   ```bash
   export CFLAGS="-std=c99"
   pip install rpy2
   pip install -e .
   ```

6. **Verify installation**

   ```bash
   python -c "import MultiGATE; print('SpatialJEPA installed successfully')"
   ```

#### Troubleshooting

- If `Cal_gene_peak_Net_new` fails with a `sortBed`/`bedtools` message, install `bedtools` and reopen the environment.
- If PyTorch installation fails, verify your Python version is 3.7 and reinstall with the pinned range.
- Ensure your Conda environment is activated before running installation commands.

</details>

## Citation

If you use `SpatialJEPA`, please cite the accepted CIBB 2026 short paper:

> Mann-Krzisnik, Dylan and Li, Yue. “SpatialJEPA: JEPA-inspired graph-context distillation for spatially aware multiomics integration.” Accepted short paper and oral presentation, *21st International Conference on Computational Intelligence Methods for Bioinformatics and Biostatistics (CIBB 2026)*, Rome, Italy, September 2–4, 2026.

`SpatialJEPA` builds on the repo for MultiGATE spatial multiomics framework. Please cite the original MultiGATE work where appropriate:

> Miao, Jishuai, Jinzhao Li, Jingxue Xin, Jiajuan Tu, Muyang Ge, Ji Qi, Xiaocheng Zhou, Ying Zhu, Can Yang, and Zhixiang Lin. "MultiGATE: integrative analysis and regulatory inference in spatial multi-omics data via graph representation learning." *Nature Communications* 16, no. 1 (2025): 9403. https://www.nature.com/articles/s41467-025-63418-x