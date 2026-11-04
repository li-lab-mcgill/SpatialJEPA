"""TF motif enrichment worker.

conda env: scenicplus (needs MOODS, pyjaspar, pyfaidx, genomepy, scipy, statsmodels)

Runs an explicit foreground-vs-background PWM scan + Fisher's exact test: for
each JASPAR2024 motif, scan foreground and background peak sequences (both
strands) with MOODS, then test whether the motif's per-peak hit rate is
higher in the foreground than the background. This is the "MOODS + pyjaspar"
first-pass design -- lightweight, no pre-built cisTarget/DEM score database,
and the statistics (Fisher's exact, BH-corrected) are explicit.

Invoked via subprocess from environments that lack these dependencies (e.g.
nichecompass_liana), so it takes a single argument: the path to a small JSON
config, rather than a CLI flag per option.

Usage:
    /path/to/scenicplus/bin/python motif_enrichment_worker.py <config.json>

config.json keys:
    foreground_bed   : path to a BED4 file (chrom, start, end, name; no header)
                        of foreground peaks
    background_bed   : path to a BED4 file of background peaks
    output_tsv        : path to write the enrichment results TSV
    fasta_path        : path to a genome FASTA (optional; auto-downloaded via
                         genomepy to its default cache if omitted or missing)
    genome_name       : genomepy genome name for auto-download (default "mm10")
    tax_group         : pyjaspar tax_group filter (default "vertebrates")
    collection        : pyjaspar collection filter (default "CORE")
    pvalue_threshold  : per-motif MOODS match p-value threshold (default 5e-4)
    pseudocount       : pseudocount for PFM -> log-odds (default 0.8)
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_config(config_path):
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_fasta(config):
    """Return a local genome FASTA path, auto-downloading via genomepy if needed."""
    fasta_path = config.get("fasta_path")
    if fasta_path and Path(fasta_path).exists():
        return fasta_path

    import genomepy

    genome_name = config.get("genome_name", "mm10")
    print(f"No usable fasta_path given; fetching {genome_name!r} via genomepy...")
    genome = genomepy.install_genome(genome_name, provider="UCSC")
    return str(genome.genome_file)


def read_bed4(path):
    return pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end", "name"])


def extract_sequences(bed_df, fasta_path):
    """dict: peak name -> uppercase sequence. Peaks on missing contigs are dropped."""
    import pyfaidx

    fasta = pyfaidx.Fasta(fasta_path)
    seqs = {}
    dropped = 0
    for row in bed_df.itertuples(index=False):
        if row.chrom not in fasta:
            dropped += 1
            continue
        seqs[row.name] = fasta[row.chrom][int(row.start):int(row.end)].seq.upper()
    if dropped:
        print(f"  dropped {dropped}/{len(bed_df)} peaks on contigs absent from the FASTA.")
    return seqs


def fetch_jaspar_motifs(collection, tax_group):
    from pyjaspar import jaspardb

    db = jaspardb(release="JASPAR2024")
    return db.fetch_motifs(collection=collection, tax_group=[tax_group])


def motif_to_log_odds(motif, bg, pseudocount):
    """Return (forward, reverse-complement) MOODS log-odds matrices for one motif."""
    import MOODS.tools as mt

    counts = [list(motif.counts[base]) for base in "ACGT"]
    fwd = mt.log_odds(counts, bg, pseudocount)
    rc = mt.reverse_complement(fwd)
    return fwd, rc


def build_scanner(matrices, bg, thresholds):
    import MOODS.scan as ms

    # MOODS lookahead-filter window; must be <= the shortest motif length, not the
    # longest. Setting it to the longest motif silently rejects nearly every motif
    # before scanning (only the widest survive), yielding ~no hits. 7 is canonical.
    window = 7
    scanner = ms.Scanner(window)
    scanner.set_motifs(matrices, bg, thresholds)
    return scanner


def region_hit_mask(scanner, seqs, n_matrices):
    """bool array (n_regions, n_motifs): True if either strand has >=1 match."""
    names = list(seqs)
    n_motifs = n_matrices // 2
    hits = np.zeros((len(names), n_motifs), dtype=bool)
    for i, name in enumerate(names):
        seq = seqs[name]
        if not seq:
            continue
        results = scanner.scan(seq)
        for j in range(n_motifs):
            hits[i, j] = bool(results[2 * j]) or bool(results[2 * j + 1])
    return names, hits


def run_motif_enrichment(config):
    fasta_path = resolve_fasta(config)

    fg_bed = read_bed4(config["foreground_bed"])
    bg_bed = read_bed4(config["background_bed"])
    print(f"Foreground peaks: {len(fg_bed)}; background peaks: {len(bg_bed)}")

    fg_seqs = extract_sequences(fg_bed, fasta_path)
    bg_seqs = extract_sequences(bg_bed, fasta_path)

    import MOODS.tools as mt

    bg_freq = mt.flat_bg(4)  # uniform background; keeps the test's assumptions explicit
    pseudocount = config.get("pseudocount", 0.8)
    pvalue_threshold = config.get("pvalue_threshold", 5e-4)

    motifs = fetch_jaspar_motifs(
        config.get("collection", "CORE"), config.get("tax_group", "vertebrates")
    )
    print(f"Fetched {len(motifs)} JASPAR2024 motifs.")

    matrices, thresholds, motif_ids, motif_names = [], [], [], []
    for motif in motifs:
        fwd, rc = motif_to_log_odds(motif, bg_freq, pseudocount)
        thr = mt.threshold_from_p(fwd, bg_freq, pvalue_threshold)
        matrices += [fwd, rc]
        thresholds += [thr, thr]
        motif_ids.append(motif.matrix_id)
        motif_names.append(motif.name)

    scanner = build_scanner(matrices, bg_freq, thresholds)
    _, fg_hits = region_hit_mask(scanner, fg_seqs, len(matrices))
    _, bg_hits = region_hit_mask(scanner, bg_seqs, len(matrices))

    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    n_fg, n_bg = len(fg_seqs), len(bg_seqs)
    rows = []
    for j, (motif_id, motif_name) in enumerate(zip(motif_ids, motif_names)):
        fg_hit = int(fg_hits[:, j].sum())
        bg_hit = int(bg_hits[:, j].sum())
        table = [[fg_hit, n_fg - fg_hit], [bg_hit, n_bg - bg_hit]]
        odds_ratio, pvalue = fisher_exact(table, alternative="greater")
        rows.append({
            "motif_id": motif_id,
            "motif_name": motif_name,
            "fg_hit": fg_hit,
            "fg_total": n_fg,
            "fg_frac": fg_hit / n_fg if n_fg else np.nan,
            "bg_hit": bg_hit,
            "bg_total": n_bg,
            "bg_frac": bg_hit / n_bg if n_bg else np.nan,
            "odds_ratio": odds_ratio,
            "pvalue": pvalue,
        })

    result = pd.DataFrame(rows)
    _, padj, _, _ = multipletests(result["pvalue"], method="fdr_bh")
    result["padj"] = padj
    result = result.sort_values(["padj", "pvalue"]).reset_index(drop=True)
    return result


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <config.json>")
    config = load_config(sys.argv[1])
    result = run_motif_enrichment(config)

    output_tsv = Path(config["output_tsv"])
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_tsv, sep="\t", index=False)
    print(f"Wrote {len(result)} motif enrichment rows -> {output_tsv}")


if __name__ == "__main__":
    main()
