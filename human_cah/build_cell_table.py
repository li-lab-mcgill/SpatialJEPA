"""Per-nucleus table for the CaH disease re-test: donor metadata, subclass, depth, and the
current Fig. 7 readouts (PLS C1 per seed; OT-transferred dopamine and its 241-gene ridge map)."""
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
COLS = ["Donor ID", "Subclass", "Supertype", "Age at Death", "Sex", "PMI", "Number of UMIs", "Genes detected",
        "Overall AD neuropathological Change", "Braak", "Thal", "CERAD score", "Cognitive Status",
        "Highest Lewy Body Disease", "LATE", "APOE Genotype", "Neurotypical reference", "Neighborhood",
        "Arteriolosclerosis", "Overall CAA Score", "section"]

with h5py.File(f"{ROOT}/fix_s0/cah/snrna_spatialmeta_aligned.h5mu", "r") as f:
    obs = f["mod/rna/obs"]
    cells = pd.Index(P.read_index(obs))
    df = pd.DataFrame({c: np.asarray(P.read_obs_column(obs, c)) for c in COLS}, index=cells)
    genes = pd.Index(P.read_index(f["mod/rna/var"])).str.replace(r"^rna:", "", regex=True).str.upper()
    X = P.read_csr(f["mod/rna/X"])
df["Subclass"] = df["Subclass"].astype(str).str.replace("Subclass$", "", regex=True)

for run in ("fix_s0", "fix_s1", "fix_s2"):
    R = pickle.load(open(f"{ROOT}/{run}/cah/{RES}", "rb"))
    c = R["c_da"]
    assert R["scores"].index.equals(cells), run
    df[f"C1_{run}"] = R["scores"][c].to_numpy()
    df[f"ot_da_{run}"] = R["ot_dopamine"].reindex(cells).to_numpy()      # 40k subsample only
    beta = R["xenium_dopamine_map"]                                       # ridge over 241 genes
    cols = np.flatnonzero(genes.isin(beta.index) & ~genes.duplicated())
    G = X[:, cols].toarray().astype(np.float64)
    G = (G - G.mean(0)) / G.std(0).clip(1e-6)
    df[f"da_map_{run}"] = G @ beta.reindex(genes[cols]).to_numpy()
    sub = df[f"ot_da_{run}"].notna()
    print(run, c, "ridge-map vs OT dopamine on the 40k subsample: r =",
          round(np.corrcoef(df.loc[sub, f"da_map_{run}"], df.loc[sub, f"ot_da_{run}"])[0, 1], 3), flush=True)
df.to_parquet("cells.parquet")
print(df.shape)
