"""Focal-recording vs soundscape AUPRC (Fig. 7): for every species in the usable
population (soundscape_species_population.get_population(), n = 52), compares its
5-fold-CV focal-recording AUPRC against its mean soundscape AUPRC across the 10 repeats,
for both backbones.

Run from this directory: python3 focal_vs_soundscape_auprc.py
"""
import csv
import statistics as st
from collections import defaultdict

from scipy.stats import spearmanr

from soundscape_species_population import get_population

FR = "/Volumes/sarahdrive/output/recognisers/final results"


def _label_to_common(label: str) -> str:
    common = (label.split("_", 1)[1] if "_" in label else label).strip().lower()
    return "gray teal" if common == "grey teal" else common


def load_clip_ap(path):
    """Mean focal-recording AUPRC per class, from the dedicated 5-fold CV (not the 10-repeat
    production models -- the 5-fold CV measures clip performance, the 10 repeats measure
    training stochasticity; they are not interchangeable)."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[_label_to_common(r["label"])] = float(r["mean_auprc"])
    return out


def load_ss_ap_mean(path):
    """Mean soundscape AUPRC per species, averaged across the 10 repeats."""
    by_sp = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_sp[r["species"].strip().lower()].append(float(r["ap"]))
    return {s: st.mean(v) for s, v in by_sp.items()}


def main():
    population = get_population()
    print(f"population: n = {len(population)}")

    bn_clip = load_clip_ap(f"{FR}/clip validation/BHC01_5foldcv/bhc01_5foldcv_per_class_metrics.csv")
    pc_clip = load_clip_ap(f"{FR}/clip validation/BHC01_perch_5foldcv/bhc01_perch_5foldcv_per_class_metrics.csv")
    bn_ss = load_ss_ap_mean(f"{FR}/soundscape eval/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv")
    pc_ss = load_ss_ap_mean(f"{FR}/soundscape eval/bhc01_perch_10repeat_soundscape_eval/per_species_metrics.csv")

    for name, clip, ss in [("BirdNET", bn_clip, bn_ss), ("Perch", pc_clip, pc_ss)]:
        xs = [clip[s] for s in population]
        ys = [ss[s] for s in population]
        rho, p = spearmanr(xs, ys)
        print(f"\n{name} (n={len(population)}):")
        print(f"  focal AUPRC:      mean={st.mean(xs):.3f} sd={st.pstdev(xs):.3f} "
              f"range={min(xs):.3f}-{max(xs):.3f}")
        print(f"  soundscape AUPRC: mean={st.mean(ys):.3f} sd={st.pstdev(ys):.3f} "
              f"range={min(ys):.3f}-{max(ys):.3f}")
        print(f"  spearman rho={rho:.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
