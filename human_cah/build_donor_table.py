try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import numpy as np, pandas as pd
from scipy.stats import spearmanr
cells = pd.read_parquet("cells.parquet")
for _c in ["Number of UMIs", "Genes detected", "Age at Death", "PMI"]:
    cells[_c] = pd.to_numeric(cells[_c].astype(str), errors="coerce")
D = "Donor ID"
meta_cols = ["Age at Death", "Sex", "PMI", "Overall AD neuropathological Change", "Braak", "Thal", "CERAD score",
             "Cognitive Status", "Highest Lewy Body Disease", "LATE", "APOE Genotype", "Neurotypical reference"]
meta = cells.groupby(D)[meta_cols].first()
assert (cells.groupby(D)[meta_cols].nunique() <= 1).all().all()
frac = pd.crosstab(cells[D], cells["Subclass"], normalize="index")
meta["frac_msn"] = frac.filter(like="MSN").sum(1)
meta["frac_oligo"] = frac["Oligodendrocyte"]
meta["n_nuclei"] = cells.groupby(D).size()
meta["log_umi"] = np.log(cells.groupby(D)["Number of UMIs"].median().astype(float))
for c in ["Highest Lewy Body Disease", "Overall AD neuropathological Change", "Cognitive Status", "LATE",
          "Neurotypical reference", "Braak", "Thal"]:
    print(meta[c].value_counts().to_string(), "\n")
print(pd.crosstab(meta["Highest Lewy Body Disease"], meta["Overall AD neuropathological Change"]))
k = pd.read_excel(f"{os.environ['SPATIALJEPA_HOME']}/data/SEA_AD/6a457afa3e9ab7e81cabccb1_sea-ad_global_and_local_cps_050126.xlsx",
                  sheet_name="Kana 2026").assign(**{D: lambda d: d[D].astype(str)}).set_index(D)
q = pd.read_excel(f"{os.environ['SPATIALJEPA_HOME']}/data/SEA_AD/6a4579d1d67150b0e69ad044_sea-ad-quantitative-neuropathology-063026.xlsx",
                  sheet_name="Kana 2026 (CaH)").assign(**{D: lambda d: d[D].astype(str)}).set_index(D)
q2 = pd.read_excel(f"{os.environ['SPATIALJEPA_HOME']}/data/SEA_AD/6a4579d1d67150b0e69ad044_sea-ad-quantitative-neuropathology-063026.xlsx",
                   sheet_name="CaH").assign(**{D: lambda d: d[D].astype(str)}).set_index(D)
meta = meta.join(k).join(q2[["percent NeuN positive area", "number of NeuN positive cells per area",
                             "percent AT8 positive area", "percent 6e10 positive area", "percent GFAP positive area"]])
meta = meta.join(q[["percent pTDP43 positive area", "tangle_per_tissue_area_um^2"]])
meta.to_csv("donors.csv")
print("\nKana CPS donors:", meta["CPS"].notna().sum(), " NeuN donors:", meta["percent NeuN positive area"].notna().sum())
print("\nDonor MSN fraction vs ...")
for c in ["percent NeuN positive area", "number of NeuN positive cells per area", "CPS", "CPS_AT8", "CPS_6E10",
          "Age at Death", "log_umi", "frac_oligo"]:
    m = meta[["frac_msn", c]].apply(pd.to_numeric, errors="coerce").dropna()
    r, p = spearmanr(m.iloc[:, 0], m.iloc[:, 1]); print(f"  {c:42s} rho {r:+.2f} p {p:.3g} n {len(m)}")
order = ["Not AD", "Low", "Intermediate", "High"]
print(meta.groupby("Overall AD neuropathological Change")["frac_msn"].agg(["mean", "count"]).reindex(order).round(3))
print(meta.groupby("Highest Lewy Body Disease")["frac_msn"].agg(["mean", "count"]).round(3))
