import csv
import os
from collections import defaultdict

FR = "/Volumes/sarahdrive/output/recognisers/final results/soundscape eval"
OUT_DIR = "auprc_by_species_all_10avg"
os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = ["model", "species", "in_vocabulary", "ap", "auroc", "best_f1",
          "best_f1_precision", "best_f1_recall", "recall_at_p50", "recall_at_p80",
          "n_pos", "prevalence"]
NUMERIC = ["ap", "auroc", "best_f1", "best_f1_precision", "best_f1_recall",
           "recall_at_p50", "recall_at_p80", "n_pos", "prevalence"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def average_repeats(rows, model_label):
    by_species = defaultdict(list)
    for r in rows:
        by_species[r["species"]].append(r)
    out = []
    for species, group in by_species.items():
        avg = {"model": model_label, "species": species,
               "in_vocabulary": group[0]["in_vocabulary"]}
        for col in NUMERIC:
            vals = [float(g[col]) for g in group if g[col] not in (None, "")]
            avg[col] = (sum(vals) / len(vals)) if vals else ""
        out.append(avg)
    return out


bn_rows = read_rows(f"{FR}/bhc01_birdnet_10repeat_soundscape_eval/per_species_metrics.csv")
pc_rows = read_rows(f"{FR}/bhc01_perch_10repeat_soundscape_eval/per_species_metrics.csv")
bn_avg = average_repeats(bn_rows, "bhc01_birdnet_10repeat_avg")
pc_avg = average_repeats(pc_rows, "bhc01_perch_10repeat_avg")
print(f"BirdNET regional: {len(bn_rows)} rows across 10 repeats -> {len(bn_avg)} species averaged")
print(f"Perch   regional: {len(pc_rows)} rows across 10 repeats -> {len(pc_avg)} species averaged")

# Each global/foundation model has its own single-run soundscape-eval folder.
GLOBAL_SOURCES = [
    (f"{FR}/birdnet_geo_broken_hill/per_species_metrics.csv", "birdnet-global"),
    (f"{FR}/birdnet3_global_soundscape_eval/per_species_metrics.csv", "birdnet-global-3.0"),
    (f"{FR}/perch_geo_broken_hill/per_species_metrics.csv", "perch"),
]
global_rows = []
for path, model in GLOBAL_SOURCES:
    global_rows += [r for r in read_rows(path) if r["model"] == model]
print(f"Global models kept as-is: {len(global_rows)} rows "
      f"({', '.join(sorted(set(r['model'] for r in global_rows)))})")

combined = bn_avg + pc_avg + [{k: r.get(k, "") for k in FIELDS} for r in global_rows]

out_csv = os.path.join(OUT_DIR, "per_species_metrics.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for row in combined:
        w.writerow({k: row.get(k, "") for k in FIELDS})

print("wrote", out_csv, "total rows:", len(combined))
