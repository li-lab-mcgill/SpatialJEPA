"""cah_ad/cells.parquet plus the seed0 Fig. 7 readouts (same recipe as cah_ad/build_cells.py)."""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import pickle, sys
import h5py, numpy as np, pandas as pd
sys.path.insert(0, os.environ["SPATIALJEPA_RNA_MSI"])
import msi_components_common as P

ROOT = f"{os.environ['SPATIALJEPA_HOME']}/outputs/human_cah_transfer"
RES = "pls_msi_ot_reg0.001_sec/results.pkl"
df = pd.read_parquet(sys.argv[1])
with h5py.File(f"{ROOT}/seed0/cah/snrna_spatialmeta_aligned.h5mu", "r") as f:
    cells = pd.Index(P.read_index(f["mod/rna/obs"]))
    genes = pd.Index(P.read_index(f["mod/rna/var"])).str.replace(r"^rna:", "", regex=True).str.upper()
    X = P.read_csr(f["mod/rna/X"])
assert cells.equals(df.index)
run = "seed0"
R = pickle.load(open(f"{ROOT}/{run}/cah/{RES}", "rb"))
assert R["scores"].index.equals(cells)
df[f"C1_{run}"] = R["scores"][R["c_da"]].to_numpy()
df[f"ot_da_{run}"] = R["ot_dopamine"].reindex(cells).to_numpy()
beta = R["xenium_dopamine_map"]
cols = np.flatnonzero(genes.isin(beta.index) & ~genes.duplicated())
G = X[:, cols].toarray().astype(np.float64)
G = (G - G.mean(0)) / G.std(0).clip(1e-6)
df[f"da_map_{run}"] = G @ beta.reindex(genes[cols]).to_numpy()
sub = df[f"ot_da_{run}"].notna()
print(run, R["c_da"], "ridge-map vs OT dopamine on the 40k subsample: r =",
      round(np.corrcoef(df.loc[sub, f"da_map_{run}"], df.loc[sub, f"ot_da_{run}"])[0, 1], 3))
print("C1 fix_s0 vs seed0 (all nuclei) r =", round(np.corrcoef(df["C1_fix_s0"], df["C1_seed0"])[0, 1], 3),
      "| dopamine map r =", round(np.corrcoef(df["da_map_fix_s0"], df["da_map_seed0"])[0, 1], 3))
df.to_parquet("cells.parquet")
