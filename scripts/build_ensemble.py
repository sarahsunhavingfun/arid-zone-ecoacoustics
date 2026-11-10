"""Builds the regional+global ensemble for each backbone: a class-specific-replacement
combination (Jacuzzi & Olden 2025 style) of the 203-class BHC01 regional model and its
matching global foundation model.

Per class, in this priority:
  1. A species the regional model's 10 repeats robustly beat the foundation model on
     (mean AUPRC across the 10 repeats exceeds the foundation model's AUPRC by more than
     the regional model's own repeat-to-repeat standard deviation) -> use the regional score.
  2. A species in BHC01's vocabulary that the foundation model cannot represent at all
     (no matching output column) -> use the regional score.
  3. Everything else -> use the foundation model's score.

This is column-wise replacement, not score blending: each output column comes entirely
from one source, so AUPRC/AUROC need no cross-model score-space calibration (they are
rank-based within a column, and no column ever mixes two sources). Run once per regional
repeat (10x), so the result is a mean and a spread across repeats, not a single value.

Requires the regional and foundation models' per-window score caches already built by
soundscape-eval (i.e. each model has already been scored once via `soundscape_eval.run`,
so its cache exists under `${call_library}/experiments/_score_cache`).

This script only builds the 20 combined caches (10 repeats x 2 backbones) -- it does not
score them itself. Score them with the real pipeline, using the manifest in this same
folder (ensemble_eval.yaml lists all 20 as `models:`):

    cd ~/soundscape-eval
    export SOUNDSCAPE_EVAL_CALL_LIBRARY=/path/to/call_library
    python3 -m soundscape_eval.run all --manifest ensemble_eval.yaml

Then take the mean/SD of `macro_auprc_all` (and `macro_auroc_all`) across each backbone's
10 rows in the resulting summary.csv -- that is the ensemble's reported AUPRC/AUROC.

Run from this directory: python3 build_ensemble.py
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

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

# BHC01 scientific name -> the name(s) a foundation model's own column list uses for the
# same species, where that differs from BHC01's own name. Verified directly against both
# caches' columns_scientific (spelling variant, genus move, or BirdNET 3.0's "(Domestic
# type)" suffix on domesticated species) -- without this, a species is wrongly counted as
# "novel" (no foundation column at all) when it is really just named differently.
# Canis lupus dingo (Dingo) is deliberately NOT included: neither model has a dingo-specific
# label, only the parent species (Canis lupus / Canis familiaris), and whether that should
# count as "covered" is a judgement call, not a naming fix -- it stays genuinely novel here.
SYNONYMS = {
    "Bos taurus": ["Bos taurus (Domestic type)"],
    "Felis catus": ["Felis catus (Domestic type)"],
    "Litoria caerulea": ["Ranoidea caerulea"],
    "Neobatrachus sudellae": ["Neobatrachus sudelli"],
}

BACKBONES = [
    dict(
        regional_prefix="bhc01_repeat", n_repeats=10,
        foundation_model="birdnet-global-3.0",
        regional_metrics=f"{FR}/soundscape eval/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv",
        foundation_metrics=f"{FR}/soundscape eval/birdnet3_global_soundscape_eval/per_species_metrics.csv",
        out_prefix="birdnet_ensemble_r",
    ),
    dict(
        regional_prefix="bhc01_perch_repeat", n_repeats=10,
        foundation_model="perch",
        regional_metrics=f"{FR}/soundscape eval/bhc01_perch_10repeat_soundscape_eval/per_species_metrics.csv",
        foundation_metrics=f"{FR}/soundscape eval/perch_geo_broken_hill/per_species_metrics.csv",
        out_prefix="perch_ensemble_r",
    ),
]


def load_bhc01_vocabulary(labels_path):
    """(list of scientific names in label-file order, {common name: scientific name})."""
    sci = []
    common_to_sci = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, common = line.split("_", 1)
            s = s.strip()
            sci.append(s)
            common_to_sci[common.strip().lower()] = s
    return sci, common_to_sci


def robust_wins(regional_metrics_csv, foundation_metrics_csv, foundation_model, common_to_sci):
    """Scientific names of species the regional model robustly beats the foundation
    model on: margin (regional mean AUPRC - foundation AUPRC) exceeds the regional
    model's own standard deviation across its 10 repeats.

    per_species_metrics.csv keys species by common name, so each win is translated to
    its scientific name via the BHC01 label file before being returned.
    """
    by_sp = defaultdict(list)
    with open(regional_metrics_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["in_vocabulary"] == "True":
                by_sp[r["species"]].append(float(r["ap"]))

    foundation = {}
    with open(foundation_metrics_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == foundation_model and r["in_vocabulary"] == "True":
                foundation[r["species"]] = float(r["ap"])

    wins = set()
    for sp, vals in by_sp.items():
        if sp not in foundation:
            continue
        mean, sd = st.mean(vals), st.stdev(vals)
        if mean - foundation[sp] > sd:
            if sp not in common_to_sci:
                print(f"  WARNING: win species {sp!r} has no scientific-name match, skipped")
                continue
            wins.add(common_to_sci[sp])
    return wins


def build_repeats(bhc01_sci, regional_prefix, n_repeats, foundation_model, wins, out_prefix):
    foundation_cache = C.load(CACHE_DIR, SOUNDSCAPE, foundation_model)
    foundation_cols = set(foundation_cache.columns_scientific)

    def foundation_name(sci):
        """The name BHC01's `sci` is actually stored under in the foundation model's
        columns, if any (itself, or a known synonym) -- else None (genuinely novel)."""
        if sci in foundation_cols:
            return sci
        for alt in SYNONYMS.get(sci, []):
            if alt in foundation_cols:
                return alt
        return None

    novel = {sci for sci in bhc01_sci if foundation_name(sci) is None}
    regional_sourced = wins | novel
    print(f"{out_prefix}: {len(wins)} robust wins + {len(novel)} novel "
          f"= {len(regional_sourced)} regional-sourced of {len(bhc01_sci)}")

    for i in range(1, n_repeats + 1):
        repeat = C.load(CACHE_DIR, SOUNDSCAPE, f"{regional_prefix}{i}")
        rep_col = {sci: j for j, sci in enumerate(repeat.columns_scientific)}
        gcol = {sci: j for j, sci in enumerate(foundation_cache.columns_scientific)}

        out_scores = np.zeros((repeat.scores.shape[0], len(bhc01_sci)), dtype="float32")
        sources = {"regional": 0, "foundation": 0}
        for j, sci in enumerate(bhc01_sci):
            fname = foundation_name(sci)
            if sci in regional_sourced and sci in rep_col:
                out_scores[:, j] = repeat.scores[:, rep_col[sci]]
                sources["regional"] += 1
            elif fname is not None:
                out_scores[:, j] = foundation_cache.scores[:, gcol[fname]]
                sources["foundation"] += 1
            elif sci in rep_col:
                out_scores[:, j] = repeat.scores[:, rep_col[sci]]
                sources["regional"] += 1

        out = C.ScoreCache(
            model=f"{out_prefix}{i}", soundscape=repeat.soundscape, scores=out_scores,
            starts=repeat.starts, columns_scientific=bhc01_sci, sig_length=repeat.sig_length,
            sample_rate=repeat.sample_rate, labeled_run=repeat.labeled_run,
            extra={"derived_from": [f"{regional_prefix}{i}", foundation_model],
                   "combiner": "class_specific_replacement"},
        )
        path = C.save(CACHE_DIR, out)
        if i == 1:
            print(f"  repeat 1 sources: {sources} -> {path}")


def main():
    bhc01_sci, common_to_sci = load_bhc01_vocabulary(LABELS_PATH)
    print(f"BHC01 vocabulary: {len(bhc01_sci)} species\n")

    for b in BACKBONES:
        wins = robust_wins(b["regional_metrics"], b["foundation_metrics"], b["foundation_model"],
                           common_to_sci)
        build_repeats(bhc01_sci, b["regional_prefix"], b["n_repeats"], b["foundation_model"],
                      wins, b["out_prefix"])
        print()


if __name__ == "__main__":
    main()
