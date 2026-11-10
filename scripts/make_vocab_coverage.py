"""Builds bhc01_backbone_vocab_coverage.csv: for each of BHC01's 203 classes, whether
BirdNET 3.0 and Perch 2.0 can represent that species at all (as a column in their own
output vocabulary), independent of how well either scores it.

Matching is scientific-name-based against each model's real output columns (read from
their soundscape-eval score caches, which is the exact vocabulary the ensemble/comparison
pipeline sees), with a short verified-synonym list for cases where a model names the same
species differently (a spelling variant, a genus move, or BirdNET 3.0's "(Domestic type)"
suffix on domesticated species) -- without it, a species is wrongly counted as missing
when it is really just spelled differently. Subspecies (e.g. Dingo, Canis lupus dingo) are
NOT resolved to their parent species: a model having the parent species is a judgement call
about whether that counts as "covering" the subspecies, not a naming fix, so Dingo is left
genuinely missing here.

Run from this directory: python3 make_vocab_coverage.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.expanduser("~/soundscape-eval"))
from soundscape_eval import cache as C  # noqa: E402

CALL_LIBRARY = os.environ.get(
    "SOUNDSCAPE_EVAL_CALL_LIBRARY",
    os.path.expanduser("~/Library/CloudStorage/OneDrive-UNSW/Will Cornwell's files - call_library"),
)
CACHE_DIR = f"{CALL_LIBRARY}/experiments/_score_cache"
SOUNDSCAPE = "sarah_soundscapes_combined"

FR = "/Volumes/sarahdrive/output/recognisers/final results"
LABELS_PATH = f"{FR}/Hyperparameters and training stochasticity/BHC01_10repeat/bhc01_repeat1_Labels.txt"
OUT_CSV = f"{FR}/bhc01_backbone_vocab_coverage.csv"

BIRDNET_MODEL = "birdnet-global-3.0"
PERCH_MODEL = "perch"

# Verified directly against both models' real output columns -- see build_ensemble.py,
# which uses the same list.
SYNONYMS = {
    "Bos taurus": ["Bos taurus (Domestic type)"],
    "Felis catus": ["Felis catus (Domestic type)"],
    "Litoria caerulea": ["Ranoidea caerulea"],
    "Neobatrachus sudellae": ["Neobatrachus sudelli"],
}


def covered(sci, columns):
    if sci in columns:
        return True
    return any(alt in columns for alt in SYNONYMS.get(sci, []))


def main():
    bhc01 = []
    with open(LABELS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sci, common = line.split("_", 1)
                bhc01.append((sci.strip(), common.strip()))

    birdnet_cols = set(C.load(CACHE_DIR, SOUNDSCAPE, BIRDNET_MODEL).columns_scientific)
    perch_cols = set(C.load(CACHE_DIR, SOUNDSCAPE, PERCH_MODEL).columns_scientific)

    rows = []
    counts = {"present_in_both": 0, "missing_from_birdnet_only": 0,
              "missing_from_perch_only": 0, "missing_from_both": 0}
    for sci, common in bhc01:
        in_bn = covered(sci, birdnet_cols)
        in_pg = covered(sci, perch_cols)
        if in_bn and in_pg:
            category = "present_in_both"
        elif in_pg:
            category = "missing_from_birdnet_only"
        elif in_bn:
            category = "missing_from_perch_only"
        else:
            category = "missing_from_both"
        counts[category] += 1
        rows.append({
            "bhc01_class": f"{sci}_{common}",
            "scientific_name": sci,
            "common_name": common,
            "canonical_key": common.lower(),
            "in_birdnet": in_bn,
            "in_perch": in_pg,
            "category": category,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bhc01_class", "scientific_name", "common_name",
                                          "canonical_key", "in_birdnet", "in_perch", "category"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {OUT_CSV}")
    print(f"birdnet model: {BIRDNET_MODEL}  |  perch model: {PERCH_MODEL}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  novel to birdnet (missing_from_birdnet_only + missing_from_both): "
          f"{counts['missing_from_birdnet_only'] + counts['missing_from_both']}")
    print(f"  novel to perch (missing_from_perch_only + missing_from_both): "
          f"{counts['missing_from_perch_only'] + counts['missing_from_both']}")


if __name__ == "__main__":
    main()
