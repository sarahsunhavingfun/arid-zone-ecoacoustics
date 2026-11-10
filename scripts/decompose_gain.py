"""Decomposes each regional model's coverage-penalised macro AUPRC gain over its
foundation counterpart into two parts, over the 54 ground-truth species the test
soundscapes are scored against:

  - coverage gain: species the foundation model's vocabulary does not cover at all (it
    scores 0 there) but the regional model does -- pure vocabulary coverage, not a
    per-species accuracy comparison.
  - shared-class improvement: the mean per-species AP difference (regional - foundation)
    over species both models can represent.

`in_vocabulary` here is each per_species_metrics.csv row's own flag from the real
soundscape-eval scoring pipeline (synonym/taxonomy-aware canonicalisation already
applied there), not a literal scientific-name string match.

Run from this directory: python3 decompose_gain.py
"""
import csv

FR = "/Volumes/sarahdrive/output/recognisers/final results/soundscape eval"

REGIONAL = {
    "BirdNET": (f"{FR}/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv", "bhc01_repeat"),
    "Perch": (f"{FR}/bhc01_perch_10repeat_soundscape_eval/per_species_metrics.csv", "bhc01_perch_repeat"),
}

COMPARISONS = [
    ("BHC01-BirdNET vs global BirdNET 3.0", "BirdNET",
     f"{FR}/birdnet3_global_soundscape_eval/per_species_metrics.csv", "birdnet-global-3.0"),
    ("BHC01-BirdNET vs global BirdNET 2.4", "BirdNET",
     f"{FR}/birdnet_geo_broken_hill/per_species_metrics.csv", "birdnet-global"),
    ("BHC01-Perch vs global Perch", "Perch",
     f"{FR}/perch_geo_broken_hill/per_species_metrics.csv", "perch"),
]


# A handful of ground-truth label spellings differ across independent soundscape-eval runs
# (different audio parses of the same soundscape set) -- normalise before joining on name.
NAME_FIX = {
    "australian plague locust": "plague locust",
    "river grass-shaker": "thomson river grass-shaker",
}


def _norm(sp):
    return NAME_FIX.get(sp, sp)


def load_global(path, model):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == model:
                out[_norm(r["species"])] = {"ap": float(r["ap"]), "in_vocab": r["in_vocabulary"] == "True"}
    return out


def load_regional(path, prefix, n=10):
    sums, invocab_any = {}, {}
    for i in range(1, n + 1):
        model = f"{prefix}{i}"
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["model"] != model:
                    continue
                sp = _norm(r["species"])
                sums[sp] = sums.get(sp, 0.0) + float(r["ap"])
                invocab_any[sp] = invocab_any.get(sp, False) or (r["in_vocabulary"] == "True")
    return {sp: {"ap": sums[sp] / n, "in_vocab": invocab_any[sp]} for sp in sums}


def decompose(label, global_d, regional_d):
    all_species = set(global_d) | set(regional_d)
    n = len(all_species)
    shared = [s for s in all_species if global_d.get(s, {"in_vocab": False})["in_vocab"]
                                      and regional_d.get(s, {"in_vocab": False})["in_vocab"]]
    novel = [s for s in all_species if regional_d.get(s, {"in_vocab": False})["in_vocab"]
                                    and not global_d.get(s, {"in_vocab": False})["in_vocab"]]
    global_only = [s for s in all_species if global_d.get(s, {"in_vocab": False})["in_vocab"]
                                          and not regional_d.get(s, {"in_vocab": False})["in_vocab"]]

    global_macro_all = sum(global_d[s]["ap"] for s in all_species) / n
    regional_macro_all = sum(regional_d[s]["ap"] for s in all_species) / n
    total_gain = regional_macro_all - global_macro_all

    improvement = sum(regional_d[s]["ap"] - global_d[s]["ap"] for s in shared) / n
    coverage_gain = sum(regional_d[s]["ap"] for s in novel) / n
    coverage_loss = -sum(global_d[s]["ap"] for s in global_only) / n

    print(f"\n=== {label} ===")
    print(f"  N (total scorable GT-present species) = {n}")
    print(f"  shared={len(shared)}  novel(regional-only)={len(novel)}  global-only={len(global_only)}")
    print(f"  global macro AUPRC_all   = {global_macro_all:.4f}")
    print(f"  regional macro AUPRC_all = {regional_macro_all:.4f}")
    print(f"  TOTAL gain = {total_gain:+.4f}")
    print(f"    from shared classes  = {improvement:+.4f}  (n={len(shared)})")
    print(f"    from novel species   = {coverage_gain:+.4f}  (n={len(novel)})")
    if global_only:
        print(f"    from global-only loss = {coverage_loss:+.4f}  (n={len(global_only)})")
    print(f"    check sum = {improvement + coverage_gain + coverage_loss:+.4f} "
          f"(should equal {total_gain:+.4f})")


def main():
    for label, backbone, global_path, global_model in COMPARISONS:
        reg_path, reg_prefix = REGIONAL[backbone]
        global_d = load_global(global_path, global_model)
        regional_d = load_regional(reg_path, reg_prefix)
        decompose(label, global_d, regional_d)


if __name__ == "__main__":
    main()
