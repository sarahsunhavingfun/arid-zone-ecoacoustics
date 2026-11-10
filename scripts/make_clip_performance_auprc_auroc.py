import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FR = "/Volumes/sarahdrive/output/recognisers/final results"
# Dedicated 5-fold CV, not the 10-repeat production models -- the 5-fold CV measures
# per-class focal-recording performance, the 10 repeats measure training stochasticity.
BN_CSV = f"{FR}/clip validation/BHC01_5foldcv/bhc01_5foldcv_per_class_metrics.csv"
PC_CSV = f"{FR}/clip validation/BHC01_perch_5foldcv/bhc01_perch_5foldcv_per_class_metrics.csv"
OUT = "clip_performance_auprc_auroc.png"

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"


def load(path):
    auprc, auroc = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            auprc.append(float(r["mean_auprc"]))
            auroc.append(float(r["mean_auroc"]))
    return np.array(auprc), np.array(auroc)


bn_auprc, bn_auroc = load(BN_CSV)
pc_auprc, pc_auroc = load(PC_CSV)

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#e5e5e5", linewidth=0.9, zorder=0)

# y positions: AUPRC band centred at 1.0 (BirdNET above, Perch below),
# AUROC band centred at 0.0 (same offsets).
positions = [1.15, 0.85, 0.15, -0.15]
data = [bn_auprc, pc_auprc, bn_auroc, pc_auroc]
colors = [BLUE, ORANGE, BLUE, ORANGE]

bp = ax.boxplot(
    data,
    positions=positions,
    vert=False,
    widths=0.22,
    patch_artist=True,
    showmeans=True,
    zorder=3,
    meanprops=dict(marker="o", markerfacecolor="dimgray", markeredgecolor="dimgray", markersize=5),
    flierprops=dict(marker="o", markerfacecolor="none", markersize=4, linestyle="none"),
    medianprops=dict(color="black", linewidth=1),
)

for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.45)
    patch.set_edgecolor(c)
for whisk, c in zip(bp["whiskers"], [colors[i // 2] for i in range(len(bp["whiskers"]))]):
    whisk.set_color(c)
for cap, c in zip(bp["caps"], [colors[i // 2] for i in range(len(bp["caps"]))]):
    cap.set_color(c)
for flier, c in zip(bp["fliers"], colors):
    flier.set_markeredgecolor(c)

# macro-mean annotations: directly above the upper (BirdNET) box of each pair,
# directly below the lower (Perch) box, centred on the mean's own x position.
label_dy = {1.15: 0.17, 0.85: -0.19, 0.15: 0.17, -0.15: -0.19}
label_va = {1.15: "bottom", 0.85: "top", 0.15: "bottom", -0.15: "top"}
for pos, vals, c in zip(positions, data, colors):
    mean_x = float(np.mean(vals))
    ax.text(mean_x, pos + label_dy[pos], f"{mean_x:.3f}",
            color=c, ha="center", va=label_va[pos], fontsize=11, fontweight="bold")

ax.set_yticks([1.0, 0.0])
ax.set_yticklabels(["AUPRC", "AUROC"], fontsize=13)
ax.set_ylim(-0.65, 1.65)
ax.set_xlabel("score", fontsize=11)
ax.set_xlim(0.70, 1.00)
ax.set_title("BHC01 custom classifiers: clip-level performance across all classes",
             fontsize=15, loc="left")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", length=0)

legend_handles = [
    mpatches.Patch(facecolor=BLUE, alpha=0.45, edgecolor=BLUE, label="BHC01 BirdNET"),
    mpatches.Patch(facecolor=ORANGE, alpha=0.45, edgecolor=ORANGE, label="BHC01 Perch"),
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="dimgray",
               markeredgecolor="dimgray", markersize=6, label="macro mean"),
]
ax.legend(handles=legend_handles, loc="lower left", frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
print(f"BirdNET AUPRC macro mean: {bn_auprc.mean():.4f} (n={len(bn_auprc)})")
print(f"Perch   AUPRC macro mean: {pc_auprc.mean():.4f} (n={len(pc_auprc)})")
print(f"BirdNET AUROC macro mean: {bn_auroc.mean():.4f}")
print(f"Perch   AUROC macro mean: {pc_auroc.mean():.4f}")
