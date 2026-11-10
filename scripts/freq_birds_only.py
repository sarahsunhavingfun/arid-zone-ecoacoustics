"""Peak call frequency vs soundscape AUPRC, birds only (n=43): Spearman correlation for
each backbone, using the 43 BHC01 bird species that have both a peak-frequency estimate
and soundscape performance data in species_frequency_vs_performance_bhc01only.csv.
"""
import csv
from scipy.stats import spearmanr

PATH = "/Volumes/sarahdrive/output/recognisers/final results/species_frequency_vs_performance_bhc01only.csv"

rows = []
with open(PATH, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["ss_ap_bn"].strip() != "":
            rows.append(r)

print(f"n = {len(rows)}")

peak = [float(r["peak_hz"]) for r in rows]
bn = [float(r["ss_ap_bn"]) for r in rows]
pc = [float(r["ss_ap_pc"]) for r in rows]

rho_bn, p_bn = spearmanr(peak, bn)
rho_pc, p_pc = spearmanr(peak, pc)
print(f"BirdNET: rho={rho_bn:+.3f}  p={p_bn:.4f}")
print(f"Perch:   rho={rho_pc:+.3f}  p={p_pc:.4f}")
