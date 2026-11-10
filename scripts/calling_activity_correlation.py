"""Calling activity vs performance: for each backbone, correlates each species' calling
activity in the test soundscapes (n_pos, the number of positive detection windows at
native window length) against its soundscape AUPRC and AUROC, each a two-sided Spearman
correlation. AUPRC/AUROC are the mean across the 10 repeats.

Run from this directory: python3 calling_activity_correlation.py
"""
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

FR = "/Volumes/sarahdrive/output/recognisers/final results/soundscape eval"

SOURCES = [
    ("BirdNET", f"{FR}/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv"),
    ("Perch", f"{FR}/bhc01_perch_10repeat_soundscape_eval/per_species_metrics.csv"),
]


def load_regional_means(path):
    """{species: (mean AUPRC, mean AUROC, n_pos)} across the 10 repeats, in-vocabulary only."""
    ap_sums, auroc_sums, counts, npos = {}, {}, {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["in_vocabulary"] != "True":
                continue
            sp = r["species"]
            ap_sums[sp] = ap_sums.get(sp, 0.0) + float(r["ap"])
            auroc_sums[sp] = auroc_sums.get(sp, 0.0) + float(r["auroc"])
            counts[sp] = counts.get(sp, 0) + 1
            npos[sp] = int(r["n_pos"])
    n = max(counts.values())
    ap = {sp: ap_sums[sp] / counts[sp] for sp in ap_sums if counts[sp] == n}
    auroc = {sp: auroc_sums[sp] / counts[sp] for sp in auroc_sums if counts[sp] == n}
    return ap, auroc, npos


def main():
    for label, path in SOURCES:
        ap, auroc, npos = load_regional_means(path)
        print(f"{label}:")

        sp = sorted(s for s in set(ap) & set(npos) if npos[s] > 0)
        x = np.log10([npos[s] for s in sp])
        y = np.array([ap[s] for s in sp])
        rho, p = spearmanr(x, y)
        print(f"  calling activity vs AUPRC: n={len(sp)}  rho={rho:+.3f}  p={p:.4g}")

        sp2 = sorted(s for s in set(auroc) & set(npos) if npos[s] > 0)
        x2 = np.log10([npos[s] for s in sp2])
        y2 = np.array([auroc[s] for s in sp2])
        rho2, p2 = spearmanr(x2, y2)
        print(f"  calling activity vs AUROC: n={len(sp2)}  rho={rho2:+.3f}  p={p2:.4g}")
        print()


if __name__ == "__main__":
    main()
