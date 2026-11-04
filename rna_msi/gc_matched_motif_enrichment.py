try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% GC-controlled re-test of the Fig. 6c TF-motif enrichments.
#
# 5__'s motif test (motif_enrichment_worker.py) contrasts the GSEA leading-edge cCRE peaks
# (foreground) against the remaining cCRE peaks of the same cell type among the 2,000
# MOFA-selected ATAC peaks (background), scanning with a flat 0.25 base background. A
# foreground that is more AT-rich than its background then enriches AT-rich motifs for
# composition alone. Two controls, on the same MOODS/JASPAR2024 scan and Fisher test:
#
#   matched   background redrawn so its GC distribution matches the foreground's:
#             N_PER_FG peaks per foreground peak from the same GC bin, drawn from all
#             multiome ATAC peaks overlapping the cell type's CATlas cCREs (still
#             "other cCREs of the same type", now GC-matched); repeated over SEEDS draws
#   logit     the original foreground vs background, with GC (+ GC^2) as covariates:
#             hit ~ foreground + GC + GC^2, one-sided Wald p for the foreground term
#
# Env: $PY_MOTIF

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motif_enrichment_worker import (  # noqa: E402
    extract_sequences, fetch_jaspar_motifs, motif_to_log_odds, read_bed4,
)

OUT = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs")
PROJ = OUT / "spatialjepa_projection"
DATA = Path(f"{os.environ['SPATIALJEPA_HOME']}/data")
FASTA = Path.home() / ".local/share/genomes/mm10/mm10.fa"
BEDTOOLS_BIN = os.environ['SPATIALJEPA_BEDTOOLS_BIN']
RUNS = {"manuscript": "V11L12-109_B1", "fixed": "V11L12-109_B1__fix2000_s3"}
CELL_TYPES = {"d2msn": lambda ct: ct.str.contains("D2MSN"), "mxd": lambda ct: ct.eq("MXD")}
PVALUE_THRESHOLD, PSEUDOCOUNT = 5e-4, 0.8          # as motif_enrichment_worker.py
GC_BIN, N_PER_FG, SEEDS = 0.02, 3, (0, 1, 2)


def parse_args():
    p = argparse.ArgumentParser(description="GC-controlled Fig. 6c motif enrichment.")
    p.add_argument("--out-dir", type=Path, default=OUT / "rna_msi_runs" / "gc_matched_motifs")
    p.add_argument("--processes", type=int, default=16)
    p.add_argument("--inputs", nargs="+", default=None, metavar="LABEL=DIR",
                   help="directories holding <cell type>_{foreground,background}.bed "
                        "(default: each RUNS projection's motif_enrichment/)")
    return p.parse_args()


def peaks_bed(names):
    df = pd.Series(names).str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    df["name"] = list(names)
    return df.dropna().astype({"start": int, "end": int})


def ccre_pool(cell_type):
    """All multiome ATAC peaks overlapping this cell type's CATlas cCREs."""
    import h5py
    import pybedtools

    os.environ["PATH"] = BEDTOOLS_BIN + os.pathsep + os.environ["PATH"]
    with h5py.File(PROJ / RUNS["manuscript"] / "multiome_target_rna_atac_msi.h5mu", "r") as f:
        var = f["mod/atac/var"]
        peaks = var[var.attrs.get("_index", "_index")][:].astype(str)
    (table,) = (DATA / "CATlas" / "supplementary_tables").glob("Supplementary Table 8*.txt")
    cc = pd.read_csv(table, sep=r"\t|\|", engine="python", names=["region", "id", "ct"], skiprows=1)
    regions = peaks_bed(cc.loc[CELL_TYPES[cell_type](cc["ct"]), "region"])
    hits = pybedtools.BedTool.from_dataframe(peaks_bed(peaks)).sort().intersect(
        pybedtools.BedTool.from_dataframe(regions[["chrom", "start", "end"]]).sort(), u=True)
    names = sorted({iv.name for iv in hits})
    print(f"[gc] {cell_type}: {len(names):,} of {len(peaks):,} multiome peaks overlap its cCREs", flush=True)
    return peaks_bed(names)


def gc_fraction(seq):
    acgt = sum(seq.count(b) for b in "ACGT")
    return (seq.count("G") + seq.count("C")) / acgt if acgt else np.nan


def matched_background(fg_gc, pool_gc, seed):
    """N_PER_FG pool peaks per foreground peak, from the same GC bin (widening if short)."""
    rng = np.random.default_rng(seed)
    pool_bins = np.floor(pool_gc.to_numpy() / GC_BIN).astype(int)
    available = {b: list(rng.permutation(pool_gc.index[pool_bins == b])) for b in np.unique(pool_bins)}
    chosen = []
    for b in np.floor(fg_gc.to_numpy() / GC_BIN).astype(int):
        for width in range(0, 6):          # same bin, then neighbours
            candidates = [k for k in (b - width, b + width) if available.get(k)]
            if sum(len(available[k]) for k in candidates) >= N_PER_FG or width == 5:
                break
        for _ in range(N_PER_FG):
            candidates = [k for k in (b - width, b + width) if available.get(k)]
            if not candidates:
                break
            k = max(candidates, key=lambda c: len(available[c]))
            chosen.append(available[k].pop())
    return pd.Index(chosen)


# --- scanning (one MOODS scanner per worker) ---------------------------------
_SCANNER = None


def _init_worker(matrices, thresholds):
    global _SCANNER
    import MOODS.scan as ms
    import MOODS.tools as mt

    _SCANNER = ms.Scanner(7)
    _SCANNER.set_motifs(matrices, mt.flat_bg(4), thresholds)


def _scan_chunk(seqs):
    out = []
    for seq in seqs:
        res = _SCANNER.scan(seq) if seq else None
        if res is None:
            out.append(None)
            continue
        out.append(np.array([bool(res[2 * j]) or bool(res[2 * j + 1]) for j in range(len(res) // 2)]))
    return out


def scan(seqs, matrices, thresholds, processes):
    names = list(seqs)
    chunks = [names[i::processes * 4] for i in range(processes * 4)]
    with Pool(processes, initializer=_init_worker, initargs=(matrices, thresholds)) as pool:
        results = pool.map(_scan_chunk, [[seqs[n] for n in chunk] for chunk in chunks])
    n_motifs = len(matrices) // 2
    hits = {}
    for chunk, res in zip(chunks, results):
        for name, row in zip(chunk, res):
            hits[name] = row if row is not None else np.zeros(n_motifs, dtype=bool)
    return pd.DataFrame.from_dict(hits, orient="index")


# --- tests -------------------------------------------------------------------
def fisher_table(hits, fg, bg, motif_names):
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    F, B = hits.loc[fg].to_numpy(), hits.loc[bg].to_numpy()
    rows = []
    for j, name in enumerate(motif_names):
        a, c = int(F[:, j].sum()), int(B[:, j].sum())
        odds, p = fisher_exact([[a, len(fg) - a], [c, len(bg) - c]], alternative="greater")
        rows.append((name, a / len(fg), c / len(bg), odds, p))
    df = pd.DataFrame(rows, columns=["motif_name", "fg_frac", "bg_frac", "odds_ratio", "pvalue"])
    df["padj"] = multipletests(df["pvalue"], method="fdr_bh")[1]
    return df


def logit_table(hits, fg, bg, gc, motif_names):
    import statsmodels.api as sm
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    idx = fg.append(bg)
    g = gc.loc[idx].to_numpy()
    g = (g - g.mean()) / g.std()
    X = sm.add_constant(np.column_stack([np.r_[np.ones(len(fg)), np.zeros(len(bg))], g, g ** 2]))
    H = hits.loc[idx].to_numpy()
    rows = []
    for j, name in enumerate(motif_names):
        y = H[:, j].astype(float)
        if y.sum() < 5 or y.sum() > len(y) - 5:
            rows.append((name, np.nan, 1.0))
            continue
        try:
            fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            beta, se = fit.params[1], fit.bse[1]
            rows.append((name, beta, float(norm.sf(beta / se))))
        except Exception:  # noqa: BLE001 - separation etc.
            rows.append((name, np.nan, 1.0))
    df = pd.DataFrame(rows, columns=["motif_name", "log_odds_fg_adj_gc", "pvalue"])
    df["padj"] = multipletests(df["pvalue"], method="fdr_bh")[1]
    return df


def main():
    args = parse_args()
    if args.inputs:
        inputs = dict(item.split("=", 1) for item in args.inputs)
        inputs = {label: Path(d) for label, d in inputs.items()}
    else:
        inputs = {run: PROJ / run_dir / "motif_enrichment" for run, run_dir in RUNS.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    motifs = fetch_jaspar_motifs("CORE", "vertebrates")
    import MOODS.tools as mt

    bg_flat = mt.flat_bg(4)
    matrices, thresholds, motif_names = [], [], []
    for motif in motifs:
        fwd, rc = motif_to_log_odds(motif, bg_flat, PSEUDOCOUNT)
        thr = mt.threshold_from_p(fwd, bg_flat, PVALUE_THRESHOLD)
        matrices += [fwd, rc]
        thresholds += [thr, thr]
        motif_names.append((motif.name, motif.matrix_id))
    counts = pd.Series([n for n, _ in motif_names]).value_counts()
    motif_names = [n if counts[n] == 1 else f"{n} ({i})" for n, i in motif_names]
    print(f"[gc] {len(motif_names)} JASPAR2024 motifs", flush=True)

    # assemble every peak set, then scan the union once
    sets, needed = {}, []
    for ct in CELL_TYPES:
        pool = ccre_pool(ct)
        pool_seqs = extract_sequences(pool, str(FASTA))
        pool_gc = pd.Series({k: gc_fraction(s) for k, s in pool_seqs.items()})
        for run, d in inputs.items():
            fg_bed, bg_bed = read_bed4(d / f"{ct}_foreground.bed"), read_bed4(d / f"{ct}_background.bed")
            fg, bg = pd.Index(fg_bed["name"]), pd.Index(bg_bed["name"])
            seqs = extract_sequences(pd.concat([fg_bed, bg_bed]), str(FASTA))
            gc = pd.Series({k: gc_fraction(s) for k, s in seqs.items()})
            fg, bg = fg.intersection(gc.index), bg.intersection(gc.index)
            candidates = pool_gc.drop(fg, errors="ignore")
            matched = {s: matched_background(gc.loc[fg], candidates, s) for s in SEEDS}
            sets[run, ct] = dict(fg=fg, bg=bg, matched=matched)
            print(f"[gc] {run} {ct}: fg {len(fg)} (GC {gc.loc[fg].mean():.3f}), original bg {len(bg)} "
                  f"(GC {gc.loc[bg].mean():.3f}), matched bg "
                  + ", ".join(f"{len(m)} (GC {pool_gc.loc[m].mean():.3f})" for m in matched.values()),
                  flush=True)
            needed.append(seqs)
            needed.append({k: pool_seqs[k] for m in matched.values() for k in m})
    all_seqs = {}
    for part in needed:
        all_seqs.update(part)
    all_gc = pd.Series({k: gc_fraction(s) for k, s in all_seqs.items()})
    print(f"[gc] scanning {len(all_seqs):,} peaks with {args.processes} processes", flush=True)
    cache = args.out_dir / "peak_motif_hits.pkl"
    hits = pd.read_pickle(cache) if cache.exists() else None
    if hits is None or not set(all_seqs) <= set(hits.index) or hits.shape[1] != len(motif_names):
        hits = scan(all_seqs, matrices, thresholds, args.processes)
        hits.to_pickle(cache)
    else:
        print(f"[gc] reusing cached scan {cache}", flush=True)

    summary, tables = [], {}
    for (run, ct), s in sets.items():
        orig = fisher_table(hits, s["fg"], s["bg"], motif_names)
        logit = logit_table(hits, s["fg"], s["bg"], all_gc, motif_names)
        matched = [fisher_table(hits, s["fg"], m, motif_names) for m in s["matched"].values()]
        merged = orig.set_index("motif_name")[["fg_frac", "bg_frac", "odds_ratio", "padj"]].add_prefix("orig_")
        merged = merged.join(logit.set_index("motif_name")[["log_odds_fg_adj_gc", "pvalue", "padj"]].add_prefix("logit_"))
        for seed, m in zip(SEEDS, matched):
            merged = merged.join(m.set_index("motif_name")[["bg_frac", "odds_ratio", "padj"]]
                                 .add_prefix(f"matched{seed}_"))
        padj_m = merged[[f"matched{s}_padj" for s in SEEDS]]
        merged["matched_median_padj"] = padj_m.median(axis=1)
        merged["matched_n_fdr"] = (padj_m < 0.05).sum(axis=1)
        merged = merged.sort_values(["matched_median_padj", "logit_padj"])
        merged.to_csv(args.out_dir / f"{run}_{ct}_gc_controlled_motifs.tsv", sep="\t")
        tables[run, ct] = merged
        robust = merged.index[(merged["matched_n_fdr"] >= 2) & (merged["logit_padj"] < 0.05)]
        summary.append(dict(
            run=run, cell_type=ct,
            fg_gc=all_gc.loc[s["fg"]].mean(), orig_bg_gc=all_gc.loc[s["bg"]].mean(),
            matched_bg_gc=np.mean([all_gc.loc[m].mean() for m in s["matched"].values()]),
            n_fdr_original=int((merged["orig_padj"] < 0.05).sum()),
            n_fdr_logit_gc=int((merged["logit_padj"] < 0.05).sum()),
            **{f"n_fdr_matched_seed{sd}": int((merged[f"matched{sd}_padj"] < 0.05).sum()) for sd in SEEDS},
            n_robust=len(robust),
            robust_motifs=", ".join(robust[:15]),
            top_original=", ".join(merged.sort_values("orig_padj").index[:8]),
        ))
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(summary.drop(columns=["robust_motifs", "top_original"]).round(3).to_string(index=False))
    for row in summary.itertuples():
        print(f"\n{row.run} {row.cell_type}\n  original top: {row.top_original}\n  robust (>=2/3 matched draws "
              f"AND GC-adjusted logit, FDR<0.05): {row.robust_motifs or '(none)'}")
        t = tables[row.run, row.cell_type]
        named = ["RXRB", "RXRG", "NR2F1", "MEIS2", "Hmx3", "DMRT3", "POU2F2", "Hmga1", "JDP2", "POU3F4",
                 "Pou5f1::Sox2", "MEF2D", "JUND", "FOSL1", "JUN::JUNB"]
        cols = ["orig_padj", "logit_padj", "matched_median_padj", "matched_n_fdr"]
        print(t.loc[t.index.intersection(named), cols].round(4).to_string())


if __name__ == "__main__":
    main()
