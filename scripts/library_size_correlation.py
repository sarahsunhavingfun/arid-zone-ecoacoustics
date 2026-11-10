"""Training library size vs soundscape AUPRC: for every species in the usable population
(soundscape_species_population.get_population(), n = 52), correlates log(library size)
against mean soundscape AUPRC across the 10 repeats, for both backbones.

Library size is counted directly from the current class folders in sarahdrive/BHC01,
not from a saved sample_counts.csv -- the library is actively growing, so a live count is
the only way to know what it actually holds right now.

Run from this directory: python3 library_size_correlation.py
"""
import csv
import math
import os
import statistics as st
from collections import defaultdict

from scipy.stats import spearmanr

from soundscape_species_population import get_population

FR = "/Volumes/sarahdrive/output/recognisers/final results"
LIB = "/Volumes/sarahdrive/BHC01"


def count_library_sizes() -> dict:
    """Clip count per class, straight off disk."""
    counts = {}
    for entry in os.scandir(LIB):
        if not entry.is_dir() or entry.name.startswith(".") or entry.name.lower() in ("background", "noise"):
            continue
        common = (entry.name.split("_", 1)[1].strip().lower() if "_" in entry.name
                  else entry.name.strip().lower())
        common = "gray teal" if common == "grey teal" else common
        counts[common] = len([f for f in os.listdir(entry.path) if not f.startswith(".")])
    return counts


def load_ss_ap_mean(path):
    by_sp = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_sp[r["species"].strip().lower()].append(float(r["ap"]))
    return {s: st.mean(v) for s, v in by_sp.items()}


def main():
    population = get_population()
    counts = count_library_sizes()
    print(f"population: n = {len(population)}")
    print(f"library size range: {min(counts[s] for s in population)}-"
          f"{max(counts[s] for s in population)} clips")

    bn_ss = load_ss_ap_mean(f"{FR}/soundscape eval/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv")
    pc_ss = load_ss_ap_mean(f"{FR}/soundscape eval/bhc01_perch_10repeat_soundscape_eval/per_species_metrics.csv")

    for name, ss in [("BirdNET", bn_ss), ("Perch", pc_ss)]:
        xs = [math.log(counts[s]) for s in population]
        ys = [ss[s] for s in population]
        rho, p = spearmanr(xs, ys)
        print(f"{name}: n={len(population)}  rho={rho:.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
