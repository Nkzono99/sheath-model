"""Draw the fixed-reservoir and matching-selection diagrams for the documentation."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


INK = "#202F3C"
MUTED = "#506374"
GRID = "#BECAD2"
BLUE = "#0072B2"
ORANGE = "#B54D00"
OUTER = "#EAF3F7"


def save(fig, directory, stem):
    fig.savefig(directory / f"{stem}.png", dpi=170, facecolor="white")
    plt.close(fig)


def smooth_curve(axis, start, end, color, *, power=1):
    t = np.linspace(0.0, 1.0, 100)
    blend = (t * t * (3 - 2 * t)) ** power
    x0, y0 = start
    x1, y1 = end
    axis.plot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * blend, color=color, lw=2.4)


def fixed_reservoir(output):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.65), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.97, top=0.71, bottom=0.20, wspace=0.29)
    fig.text(0.065, 0.91, "Where does the potential minimum occur?", size=20, weight="bold", color=INK)
    fig.text(0.97, 0.92, "SCHEMATIC", ha="right", size=9, color=MUTED)

    join_x = 0.51
    for axis, title in zip(axes, ("Finite column · A", "Outer model · A", "Outer model · B_out")):
        axis.set(xlim=(-0.035, 1.035), ylim=(-0.69, 1.04), xticks=[], yticks=[])
        axis.spines[:].set_visible(False)
        axis.set_title(title, loc="left", color=INK, fontsize=13, weight="bold", pad=15)
        axis.axvline(0, color=INK, lw=1.0)
        axis.axhline(0, color=GRID, ls="--", lw=1)
        axis.text(1.02, 0.043, "0 V", ha="right", size=9, color=MUTED)
        axis.text(0, -0.083, "Wall", transform=axis.get_xaxis_transform(), ha="left", va="top", color=MUTED)
    axes[0].set_ylabel(r"Potential $\phi$", color=MUTED, labelpad=10)

    finite = axes[0]
    finite.axvline(1, color=INK, lw=1.0)
    finite.text(0.52, 0.93, r"Fixed inflow at $z=H$", ha="center", color=MUTED, size=10)
    finite.annotate("", xy=(0.87, 0.75), xytext=(1, 0.75),
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.2})
    finite.text(1, -0.083, r"Top ($z=H$)", transform=finite.get_xaxis_transform(),
                ha="right", va="top", color=MUTED)
    finite_minimum = (0.79, -0.36)
    smooth_curve(finite, (0, 0.66), finite_minimum, BLUE, power=0.5)
    x = np.linspace(finite_minimum[0], 1, 60)
    # Zero slope at the minimum and a free, nonzero electric field at the top.
    finite.plot(x, finite_minimum[1] * (1 - ((x - x[0]) / (1 - x[0])) ** 2), color=BLUE, lw=2.4)
    finite.plot(1, 0, "o", ms=4.5, color=BLUE)

    for axis in axes[1:]:
        axis.axvspan(join_x, 1.035, color=OUTER, zorder=0)
        axis.axvline(join_x, color=GRID, lw=1)
        axis.text(0.25, 0.93, "Sheath", ha="center", size=9.5, color=MUTED)
        axis.text(0.775, 0.98, "Quasineutral\nouter region", ha="center", va="top", size=9.0, color=MUTED)
        axis.text(join_x, -0.083, r"$\tau=0$", transform=axis.get_xaxis_transform(),
                  ha="center", va="top", color=MUTED)
        axis.text(1, -0.083, "Far reservoir", transform=axis.get_xaxis_transform(),
                  ha="right", va="top", color=MUTED)

    inside, outside = axes[1:]
    inside_minimum = (0.30, -0.47)
    inside_join = (join_x, -0.19)
    smooth_curve(inside, (0, 0.66), inside_minimum, BLUE, power=0.5)
    smooth_curve(inside, inside_minimum, inside_join, BLUE)
    smooth_curve(inside, inside_join, (1, -0.015), BLUE)

    outside_join = (join_x, -0.24)
    outside_minimum = (0.72, -0.43)
    smooth_curve(outside, (0, 0.66), outside_join, ORANGE, power=0.5)
    smooth_curve(outside, outside_join, outside_minimum, ORANGE)
    smooth_curve(outside, outside_minimum, (1, -0.018), ORANGE)

    for axis, (x, phi), color in zip(
        axes, (finite_minimum, inside_minimum, outside_minimum), (BLUE, BLUE, ORANGE)
    ):
        axis.plot(x, phi, "o", ms=6.5, color=color, mec="white", mew=1, zorder=5)
        axis.annotate("Minimum", (x, phi), xytext=(0, -8), textcoords="offset points",
                      va="top", ha="center", fontsize=9.5, color=INK)

    fig.text(0.97, 0.035, "Wall  →  upstream     ·     Distances and potential differences are illustrative",
             ha="right", color=MUTED, size=9)
    save(fig, output, "fixed_reservoir_schematic")


def matching_selection(output):
    fig, (residual, potential) = plt.subplots(1, 2, figsize=(11.8, 4.2),
                                               gridspec_kw={"width_ratios": (1, 1.4)})
    fig.subplots_adjust(left=0.10, right=0.97, top=0.73, bottom=0.23, wspace=0.40)
    fig.text(0.065, 0.91, "Selecting a matching-plane condition", size=20, weight="bold", color=INK)
    fig.text(0.97, 0.92, "SCHEMATIC", ha="right", size=9, color=MUTED)

    residual.set_title("1   Find the target", loc="left", fontsize=13, weight="bold", color=INK, pad=12)
    residual.set(xlim=(-1.45, 1.75), ylim=(-1.2, 1.15), xticks=[], yticks=[])
    residual.set_aspect("equal")
    residual.spines[:].set_visible(False)
    residual.axhline(0, color=GRID, zorder=0)
    residual.axvline(0, color=GRID, zorder=0)
    residual.set_xlabel(r"Field residual  $(E^+-E^-)/\sigma_E$", color=MUTED, labelpad=9)
    residual.set_ylabel(r"Current residual  $J/(e\Gamma_i\sigma_J)$", color=MUTED, labelpad=10)

    # Each tuple gives normalized field and current residuals and its grid potential.
    type_a = [(-1.01, 0.52, 0.48), (-0.88, -0.69, 0.79), (0.88, -0.63, 0.90)]
    type_b = [(1.30, -0.36, 0.27), (0.97, 0.63, 0.57), (0.28, 0.18, 0.85)]
    selected, target = type_b[1:]
    residual.plot(0, 0, "+", ms=8, color=MUTED)
    residual.add_patch(Circle((0, 0), np.hypot(*target[:2]), fill=False, ls="--", lw=1.2, ec=MUTED))
    for states, marker, color in ((type_a, "^", BLUE), (type_b, "o", ORANGE)):
        field, current, _ = zip(*states)
        residual.plot(field, current, marker, ms=8, color=color, mec="white", mew=0.6, ls="")
    residual.annotate("Target", target[:2], xytext=(0.51, -0.26), color=INK, size=10,
                      arrowprops={"arrowstyle": "-", "color": MUTED})

    potential.set_title("2   Relax, limit and select", loc="left", fontsize=13, weight="bold", color=INK, pad=12)
    potential.set(xlim=(0.06, 0.98), ylim=(-1.0, 1.0), xticks=[], yticks=[])
    potential.spines[:].set_visible(False)
    previous = 0.16
    proposal = previous + min(0.75 * (target[2] - previous), 0.35)
    potential.annotate("", xy=(0.98, 0), xytext=(0.06, 0), zorder=0,
                       arrowprops={"arrowstyle": "->", "lw": 1.1, "color": GRID})
    potential.annotate("", xy=(proposal, 0.65), xytext=(previous, 0.65),
                       arrowprops={"arrowstyle": "->", "lw": 1.3, "color": MUTED})
    potential.text((previous + proposal) / 2, 0.72, "Relaxed, capped step",
                   ha="center", va="bottom", size=9.5, color=MUTED)
    potential.vlines([previous, proposal], 0, 0.65, colors=GRID, linestyles=":")

    for states, marker, color in ((type_a, "^", BLUE), (type_b, "o", ORANGE)):
        phi = [state[2] for state in states]
        potential.plot(phi, np.zeros(len(phi)), marker, ms=8, color=color, mec="white", mew=0.6, ls="")
    potential.plot(previous, 0, "D", color=MUTED, ms=6)
    potential.plot(proposal, 0, "o", color=MUTED, mfc="white", ms=6)
    for x, y, label, align in (
        (previous, -0.29, "Previous", "center"),
        (proposal, -0.54, "Proposal", "right"),
        (selected[2], 0.31, "Selected", "center"),
        (target[2], -0.29, "Target", "center"),
    ):
        potential.annotate(label, (x, 0), xytext=(x, y), color=INK, size=10, ha=align,
                           va="bottom" if y > 0 else "top", arrowprops={"arrowstyle": "-", "color": MUTED})
    potential.set_xlabel(r"Candidate plane potential  $\phi_H$", color=MUTED, labelpad=9)

    handles = [Line2D([], [], ls="", marker=marker, color=color, label=label)
               for marker, color, label in (("^", BLUE, "Type A"), ("o", ORANGE, "Type B"))]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.968, 0.035), ncols=2, frameon=False)
    save(fig, output, "matching_selection_schematic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "docs/figures")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    fixed_reservoir(output)
    matching_selection(output)


if __name__ == "__main__":
    main()
