"""GPU part of the two supplementary figures (seed0, Fig. 7 recipe; nothing in Fig. 7 changes).

1  Held-out source sections (as cah_components.stage_validate): each section's MSI held out, U from the
   other two; the held-out section's metabolite weights from its measured MSI vs OT-recovered (the
   paper's estimator) and vs copied from the other two sections. Saves all 787 x K weights per section.
2  The Fig. 7 OT pairing (40,000 nuclei, seed 0, per-section OT, reg 0.001) with the paired MSI per
   nucleus, the decoder (B, R) and the nucleus indices, for subclass x metabolite summaries.
Writes supp_metab.pkl next to this script. Env: nichecompass_liana on a GPU.
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ["SPATIALJEPA_HUMAN_CAH"])
import cah_components as F  # noqa: E402

SC, P = F.SC, F.P
HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer"
RUN, REG = "seed0", 0.001
t0 = time.time()
p = F.paths(RUN, "msi", "ot", REG, "section")
S, T = F.load_source(p["bundle"]), F.load_target(p["cah"])
R = pickle.load(open(p["pls"] / "results.pkl", "rb"))
da = S["metabolites"].get_loc("msi:Dopamine")

held_out = {}
for held in np.unique(S["sections"]):                       # same steps as stage_validate
    tr, te = S["sections"] != held, S["sections"] == held
    b_tr = S["sections"][tr]
    Zt, Mt = SC.standardize(S["Z"][te]), S["msi"][te]
    Zs = SC.standardize(F.centre_within(S["Z"][tr], b_tr))
    Ms = F.centre_within(S["msi"][tr], b_tr)
    Cs = SC.corr_matrix(Zs, Ms, P.column_stats(Ms))
    U = np.linalg.svd(Cs / np.linalg.norm(Cs), full_matrices=False)[0][:, :F.K].copy()
    U *= np.where(np.array([SC.direction_weights(Cs, Zs, U[:, k])[da] for k in range(F.K)]) < 0, -1, 1)
    w = lambda C, Z: np.column_stack([SC.direction_weights(C, Z, U[:, k]) for k in range(F.K)])
    groups = np.random.default_rng(0).integers(0, 5, te.sum())
    held_out[held] = dict(
        U=U, measured=w(SC.corr_matrix(Zt, Mt, P.column_stats(Mt)), Zt), copied=w(Cs, Zs),
        ot=SC.ot_msi_weights(S["Z"][tr], S["msi"][tr], Zt, U, groups, reg=REG, batches=b_tr)[0],
        r_U_vs_full=np.abs(U.T @ R["U"]))                  # |cos| of each held-out component with the full-data ones
    print(f"held out {held} ({time.time() - t0:.0f} s)", flush=True)

Zt_all = SC.standardize(T["Z"])
pick = np.random.default_rng(0).choice(len(Zt_all), size=F.N_OT, replace=False)
Zt = SC.standardize(Zt_all[pick])
groups = np.random.default_rng(0).integers(0, 5, len(pick))
W, dec, Y = SC.ot_msi_weights(S["Z"], S["msi"], Zt, R["U"], groups, reg=REG, batches=S["sections"], return_paired=True)
print(f"pairing reproduces stored weights r = {np.corrcoef(W.ravel(), R['W']['target MSI (OT)'].to_numpy().ravel())[0, 1]:.4f} "
      f"({time.time() - t0:.0f} s)", flush=True)
pickle.dump(dict(held_out=held_out, metabolites=list(S["metabolites"]), comps=R["comps"], pick=pick,
                 paired=Y.astype(np.float32), dec_B=dec["B"], dec_R=dec["R"], dec_k=dec["k"],
                 subclass=T["subclass"], cells=list(T["cells"]), z_mu=Zt_all[pick].mean(0), z_sd=Zt_all[pick].std(0, ddof=1)),
            open(HERE / "supp_metab.pkl", "wb"))
print(f"wrote supp_metab.pkl ({time.time() - t0:.0f} s)", flush=True)
