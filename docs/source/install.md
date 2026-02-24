# Installation

Follow these steps to install MultiGATE in a dedicated Conda environment.

```shell
conda create -n MultiGATEenv python=3.7 -y
conda activate MultiGATEenv

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
pip install MultiGATE
```

> MultiGATE now uses a PyTorch backend by default. The original TensorFlow model class is still available as a legacy class.

## Detailed instructions

1. **Create a Conda environment**

   ```bash
   conda create -n MultiGATEenv python=3.7 -y
   ```

2. **Activate the environment**

   ```bash
   conda activate MultiGATEenv
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

5. **Install optional R bridge and MultiGATE**

   ```bash
   export CFLAGS="-std=c99"
   pip install rpy2
   pip install MultiGATE
   ```

6. **Verify installation**

   ```bash
   python -c "import MultiGATE; print('MultiGATE installed successfully')"
   ```

## Troubleshooting

- If `Cal_gene_peak_Net_new` fails with a `sortBed`/`bedtools` message, install `bedtools` and reopen the environment.
- If PyTorch installation fails, verify your Python version is 3.7 and reinstall with the pinned range.
- Ensure your Conda environment is activated before running installation commands.
