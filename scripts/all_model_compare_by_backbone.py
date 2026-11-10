"""Rebuild all_model_compare_by_backbone.png.

Values are cross-checked against the live outputs of the other scripts here.
SD is None for single-run global models.
"""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("all_model_compare_by_backbone.png")

# (label, colour, (AUPRC mean, SD), (AUROC mean, SD))
BIRDNET = [
    ("BirdNET V2.4", "tab:green", (0.151, None), (0.654, None)),
    ("BirdNET V3.0", "tab:red", (0.222, None), (0.724, None)),
    ("BHC01_BirdNET", "tab:blue", (0.237, 0.003), (0.682, 0.008)),
    ("BHC01_BirdNET + BirdNET V3.0", "tab:cyan", (0.292, 0.002), (0.765, 0.003)),
]
PERCH = [
    ("Perch V2.0", "tab:purple", (0.242, None), (0.712, None)),
    ("BHC01_Perch", "tab:orange", (0.301, 0.002), (0.765, 0.003)),
    ("BHC01_Perch + Perch V2.0", "tab:pink", (0.327, 0.001), (0.776, 0.001)),
]
METRICS = ["Macro AUPRC", "Macro AUROC"]

TITLE = "BHC01 soundscape-eval performance, split by backbone family"
SUBTITLE = (
    "Global rows: single stock-model run, no geo constraint. BHC01 rows: mean ± SD "
    "across 10 independently trained models (native window grid). Combined rows: "
    "each global backbone plus BHC01's own regional model, class-specific "
    "replacement per species (checked on these soundscapes, not assumed) — mean ± SD "
    "across the same 10 regional-model repeats, geo-constrained to BHC01's "
    "203-species vocabulary."
)


def fmt(mean, sd):
    return f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}"


def main():
    gap = 1.0  # extra space between the two backbone families
    rows = [(i, r) for i, r in enumerate(BIRDNET)]
    rows += [(len(BIRDNET) + gap + i, r) for i, r in enumerate(PERCH)]
    divider = len(BIRDNET) - 1 + (1 + gap) / 2

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.1), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.215, right=0.99, top=0.68, bottom=0.05, wspace=0.05)
    fig.text(0.015, 0.97, TITLE, fontsize=17, va="top")
    fig.text(0.015, 0.915, "\n".join(textwrap.wrap(SUBTITLE, 110)),
             fontsize=8.5, color="#444", va="top", linespacing=1.3)

    for m, (ax, metric) in enumerate(zip(axes, METRICS)):
        for y, (label, colour, *vals) in rows:
            mean, sd = vals[m]
            if sd is not None:
                ax.errorbar(mean, y, xerr=sd, color=colour, capsize=0, lw=1.2)
            ax.plot(mean, y, "o", color=colour, ms=7, zorder=3)
            ax.annotate(fmt(mean, sd), (mean, y), xytext=(7, 0),
                        textcoords="offset points", va="center",
                        color=colour, fontsize=9.5, fontweight="bold")
        ax.axhline(divider, color="#ccc", lw=0.8)
        ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.grid(axis="x", color="#e3e3e3", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#e3e3e3")
        ax.set_title(metric, loc="left", fontsize=14, pad=14)

    axes[0].set_yticks([y for y, _ in rows], [r[0] for _, r in rows])
    axes[0].set_ylim(rows[-1][0] + 0.7, -0.7)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    main()
