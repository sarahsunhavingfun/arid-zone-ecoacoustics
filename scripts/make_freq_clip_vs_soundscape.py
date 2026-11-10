import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

CSV = "/Volumes/sarahdrive/output/recognisers/final results/species_frequency_vs_performance_bhc01only.csv"
OUT = "/private/tmp/claude-502/-Users-z5478633-BirdNET-Analyzer-birdnet-analyzer/655d5b95-382c-4c3d-aba1-596da2cc17d1/scratchpad/frequency_clip_vs_soundscape.png"
COLOR = "#ff7f0e"
HIGHLIGHT = {"dromaius novaehollandiae": "emu", "phaps chalcoptera": "common bronzewing"}

rows = []
with open(CSV, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        # Restrict to the 43 birds with soundscape data, so both panels plot the same
        # species -- clip_ap_pc alone is populated for all 169 library birds (every class
        # gets focal-recording CV), which previously let the left panel show far more
        # points than the right.
        if not r["peak_hz"] or not r["ss_ap_pc"]:
            continue
        rows.append(r)
print(f"n species (both panels) = {len(rows)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=150)

for ax, ap_key, title, panel in [(ax1, "clip_ap_pc", "Clip-based AUPRC", "left"), (ax2, "ss_ap_pc", "Soundscape AUPRC", "right")]:
    xs, ys, names = [], [], []
    for r in rows:
        if not r[ap_key]:
            continue
        xs.append(float(r["peak_hz"]))
        ys.append(float(r[ap_key]))
        names.append(r["species"].strip().lower())

    for x, y, name in zip(xs, ys, names):
        is_hl = name in HIGHLIGHT
        ax.scatter([x], [y], color=COLOR, s=60 if is_hl else 28,
                   edgecolors="black" if is_hl else "none", linewidths=1.3 if is_hl else 0,
                   zorder=3 if is_hl else 2, alpha=1.0 if is_hl else 0.85)

    # Highlighted markers are drawn at s=60 (area, pt^2) with a 1.3pt edge stroke, so their
    # rendered radius is sqrt(60/pi) + 1.3/2 =~ 5pt -- shrinkB stops the leader line there
    # instead of running to the marker's exact (x, y) center, which would cut through it.
    MARKER_RADIUS_PT = 5

    for x, y, name in zip(xs, ys, names):
        if name == "dromaius novaehollandiae":
            # No line -- sits just below-right (left panel) / above-right (right panel).
            dy = -0.03 if panel == "left" else 0.03
            ax.annotate("Emu", (x, y), xytext=(x * 1.08, y + dy), fontsize=12, ha="left")
        elif name == "phaps chalcoptera":
            if panel == "left":
                # Near the dense top cluster, so this one keeps a leader line.
                ax.annotate("Common Bronzewing", (x, y), xytext=(x * 1.15, y - 0.14), fontsize=12,
                            ha="left", arrowprops=dict(arrowstyle="-", color="black", lw=1, shrinkB=MARKER_RADIUS_PT))
            else:
                ax.annotate("Common Bronzewing", (x, y), xytext=(x * 1.08, y + 0.03), fontsize=12,
                            ha="left")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(title, fontsize=16, loc="left")
    ax.set_xlabel("peak frequency (Hz)", fontsize=12)
    ax.set_ylabel("AUPRC", fontsize=12)
    ax.set_ylim(-0.03, 1.03)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle("Peak call frequency vs AUPRC: clips vs soundscapes", fontsize=19, fontweight="bold", x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
