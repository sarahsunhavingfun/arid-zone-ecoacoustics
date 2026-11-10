"""Win-loss margin test: for each species a regional BHC01 model shares with its
foundation counterpart, classifies the species as a regional win, loss, or tie, then asks
whether the SIZE of the margin differs systematically between wins and losses.

Win/loss rule: a species is a regional WIN only if the regional model's mean AUPRC across
its 10 repeats beats the foundation model's AUPRC by more than the regional model's own
standard deviation across those repeats -- i.e. the margin has to clear the model's own
repeat-to-repeat noise floor, not just be positive. Everything that doesn't clear that bar
(close calls and clear losses alike) is a loss. This is the same rule used to decide, per
species, which model's score an ensemble takes.

The test itself is a two-sided Mann-Whitney U test on the absolute margin size, wins vs
losses -- it asks whether wins tend to be bigger or smaller than losses, not just which
group is more numerous.

Run from this directory: python3 win_loss_margin_test.py
"""
import csv
import statistics as st
from collections import defaultdict

from scipy.stats import mannwhitneyu

FR = "/Volumes/sarahdrive/output/recognisers/final results/soundscape eval"

COMPARISONS = [
    ("BHC01_BirdNET vs BirdNET V2.4",
     f"{FR}/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv",
     f"{FR}/birdnet_geo_broken_hill/per_species_metrics.csv", "birdnet-global"),
    ("BHC01_BirdNET vs BirdNET V3.0",
     f"{FR}/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv",
     f"{FR}/birdnet3_global_soundscape_eval/per_species_metrics.csv", "birdnet-global-3.0"),
    ("BHC01_Perch vs Perch V2.0",
     f"{FR}/bhc01_perch_10repeat_soundscape_eval/per_species_metrics.csv",
     f"{FR}/perch_geo_broken_hill/per_species_metrics.csv", "perch"),
]


def load_regional(path):
    """{species: [AUPRC across the 10 repeats]}, in-vocabulary rows only."""
    by_sp = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["in_vocabulary"] == "True":
                by_sp[r["species"]].append(float(r["ap"]))
    return by_sp


def load_foundation(path, model_name):
    """{species: AUPRC} for one named model, in-vocabulary rows only."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == model_name and r["in_vocabulary"] == "True":
                out[r["species"]] = float(r["ap"])
    return out


def main():
    for label, regional_path, foundation_path, foundation_model in COMPARISONS:
        regional = load_regional(regional_path)
        foundation = load_foundation(foundation_path, foundation_model)
        shared = sorted(set(regional) & set(foundation))

        wins, losses = [], []
        for s in shared:
            vals = regional[s]
            mean, sd = st.mean(vals), st.stdev(vals)
            margin = mean - foundation[s]
            (wins if margin > sd else losses).append(abs(margin))

        u, p = mannwhitneyu(wins, losses, alternative="two-sided")
        print(f"{label}:")
        print(f"  shared species: {len(shared)}")
        print(f"  wins: {len(wins)} (mean margin {st.mean(wins):.3f})  "
              f"losses: {len(losses)} (mean margin {st.mean(losses):.3f})")
        print(f"  Mann-Whitney U={u:.1f}  p={p:.4f}\n")


if __name__ == "__main__":
    main()
