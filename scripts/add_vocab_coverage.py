"""Adds vocabulary-coverage columns (in_birdnet, in_perch) to the Aridnet species list,
checking each species' scientific name against BirdNET 3.0's and Perch's real output
columns (read from their soundscape-eval score caches), with the same verified-synonym
handling as make_vocab_coverage.py -- a spelling variant, genus move, or BirdNET 3.0's
"(Domestic type)" suffix on domesticated species does not count as missing. Matching is
case-insensitive (the source list has a few inconsistent-case entries).

Rows with no checkable scientific name (unidentified morphospecies, order-level classes,
or the background/noise non-target classes) get "N/A" in both new columns rather than
being marked missing.

Run from this directory: python3 add_vocab_coverage.py
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

IN_CSV = "Aridnet_species_list.csv"
OUT_CSV = "Aridnet_species_list_vocab_coverage.csv"

BIRDNET_MODEL = "birdnet-global-3.0"
PERCH_MODEL = "perch"

# Verified directly against both models' real output columns -- see build_ensemble.py /
# make_vocab_coverage.py, which use the same list.
SYNONYMS = {
    "Bos taurus": ["Bos taurus (Domestic type)"],
    "Felis catus": ["Felis catus (Domestic type)"],
    "Litoria caerulea": ["Ranoidea caerulea"],
    "Neobatrachus sudellae": ["Neobatrachus sudelli"],
    # Genus move: BirdNET 3.0 still carries the old name as its own column (so this is a
    # no-op there); Perch only has the current name. BHC01's own trained class already
    # uses "Thinornis melanops" -- this list's "Elseyornis melanops" is the older name.
    "Elseyornis melanops": ["Thinornis melanops"],
    # Latin grammatical-gender adjective agreement variants -- same species, both models
    # use the ending agreeing with the (masculine) genus rather than this list's.
    "Threskiornis moluccus": ["Threskiornis molucca"],
    "Sugomel niger": ["Sugomel nigrum"],
}

NOT_CHECKABLE = {"n/a", "background", "noise", ""}


def covered(sci, columns_lower):
    if sci.lower() in columns_lower:
        return True
    return any(alt.lower() in columns_lower for alt in SYNONYMS.get(sci, []))


def main():
    birdnet_cols = {c.lower() for c in
                    C.load(CACHE_DIR, SOUNDSCAPE, BIRDNET_MODEL).columns_scientific}
    perch_cols = {c.lower() for c in
                  C.load(CACHE_DIR, SOUNDSCAPE, PERCH_MODEL).columns_scientific}

    with open(IN_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames + ["in_birdnet", "in_perch"]

    counts = {"present_in_both": 0, "missing_from_birdnet_only": 0,
              "missing_from_perch_only": 0, "missing_from_both": 0, "not_checkable": 0}

    for r in rows:
        sci = (r.get("Species Name") or "").strip()
        if sci.lower() in NOT_CHECKABLE:
            r["in_birdnet"] = r["in_perch"] = "N/A"
            counts["not_checkable"] += 1
            continue
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
        r["in_birdnet"] = in_bn
        r["in_perch"] = in_pg

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {OUT_CSV}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
