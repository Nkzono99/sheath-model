"""Plot README figures from the Fortran readme_data example's CSV output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

COLORS = {
    0: "#8D969E",  # All attempted candidates rejected by physical checks.
    1: "#0072B2",  # A
    2: "#D55E00",  # B
    3: "#8A65B5",  # A+B
    4: "#009E73",  # C
    5: "#397C90",  # A+C
    6: "#B2993E",  # B+C
    7: "#454560",  # A+B+C
    8: "#E8EDF1",  # No accepted root; at least one search unresolved.
}
LABELS = {
    0: "Candidates rejected",
    1: "A",
    2: "B",
    3: "A + B",
    4: "C",
    5: "A + C",
    6: "B + C",
    7: "A + B + C",
    8: "No root resolved",
}
CONDITIONS = (
    r"$n_i=8.7\,\mathrm{cm}^{-3}$   |   $T_e=12\,\mathrm{eV}$   |   "
    r"$T_{pe}=2.2\,\mathrm{eV}$   |   $v_{sw}=468\,\mathrm{km\,s}^{-1}$   |   $u_e=0$"
)


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def save_figure(fig, directory, stem):
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=190, facecolor="white")
    plt.close(fig)


def plot_profiles(data_dir, output_dir):
    metadata = read_csv(data_dir / "profile_metadata.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.1), sharex="col")
    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.14, top=0.77, hspace=0.12, wspace=0.30
    )
    fig.suptitle(
        "Representative 1D sheath solutions",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(0.08, 0.90, CONDITIONS, fontsize=11)
    fig.text(
        0.08,
        0.85,
        r"Zero current $J_z=0$   |   $n_{pe,ref}=64\,\mathrm{cm}^{-3}$   |   "
        r"$v_i=v_{sw}\sin\alpha$, $n_{pe,0}=n_{pe,ref}\sin\alpha$",
        fontsize=11,
    )
    summaries = []
    for column, (branch, bit) in enumerate(zip("ABC", (1, 2, 4))):
        row = metadata[metadata["type"] == branch][0]
        data = read_csv(data_dir / f"profile_{branch}.csv")
        z, phi, field = (
            data[key] for key in ("z_m", "potential_v", "electric_field_v_m")
        )
        if not np.all(np.isfinite([z, phi, field])) or not np.all(np.diff(z) > 0):
            raise ValueError(f"Invalid profile grid: {branch}")
        if abs(row["current_a_m2"]) > 1e-14 or row["residual"] > 1e-8:
            raise ValueError(
                f"Representative {branch} is not a converged zero-current root"
            )
        # Use the same physical distance on every panel; the full tails remain in CSV.
        for axis in axes[:, column]:
            axis.axhline(0, color="#8F9BA5", linewidth=0.8)
            axis.set_xlim(0, 60)
            axis.grid(True, color="#E7ECF0", linewidth=0.65)
            axis.set_axisbelow(True)
        axes[0, column].plot(z, phi, color=COLORS[bit], linewidth=2.2)
        axes[1, column].plot(z, field, color=COLORS[bit], linewidth=2.2)
        axes[0, column].set_title(
            f"Type {branch}  ·  α = {row['alpha_deg']:g}°",
            loc="left",
            color=COLORS[bit],
            weight="bold",
        )
        axes[0, column].text(
            0.96,
            0.93,
            rf"$\phi_H={phi[0]:.2f}\,\mathrm{{V}}$",
            transform=axes[0, column].transAxes,
            ha="right",
            va="top",
            fontsize=11,
        )
        axes[1, column].set_xlabel(r"Height $z$ [m]")
        axes[0, column].set_ylabel(r"Potential $\phi$ [V]")
        axes[1, column].set_ylabel(r"Electric field $E_z$ [V/m]")
        if branch == "A":
            turning = row["turning_height_m"]
            axes[0, column].scatter(
                [turning], [row["phi_min_v"]], color=COLORS[bit], s=28, zorder=4
            )
            axes[0, column].annotate(
                rf"$\phi_{{min}}={row['phi_min_v']:.2f}$ V",
                xy=(turning, row["phi_min_v"]),
                xytext=(22, 0.9),
                arrowprops={"arrowstyle": "-", "color": COLORS[bit]},
                fontsize=10,
            )
            for axis in axes[:, column]:
                axis.axvline(
                    turning, color=COLORS[bit], alpha=0.35, linestyle=":", linewidth=1
                )
        summaries.append(
            {
                "type": branch,
                "alpha_deg": float(row["alpha_deg"]),
                "surface_potential_v": float(phi[0]),
                "minimum_potential_v": float(row["phi_min_v"]),
                "surface_field_v_m": float(field[0]),
                "current_a_m2": float(row["current_a_m2"]),
                "last_saved_height_m": float(z[-1]),
            }
        )
    fig.text(
        0.08,
        0.04,
        r"$+z$: surface → upstream.  $E_z=-d\phi/dz$.  "
        "All profiles approach zero potential and field upstream; the first 60 m are shown.",
        fontsize=10,
        color="#53616D",
    )
    save_figure(fig, output_dir, "sheath_profiles")
    return summaries


def plot_maps(data_dir, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.3))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.29, top=0.75, wspace=0.18)
    fig.suptitle(
        "Admissible sheath types found",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(0.08, 0.90, CONDITIONS, fontsize=11)
    fig.text(
        0.08,
        0.85,
        r"Source ratio $r=n_{pe,ref}/n_i$   |   "
        r"$n_{pe,0}=r\,n_i\sin\alpha$   |   $v_i=v_{sw}\sin\alpha$",
        fontsize=11,
    )
    cmap = ListedColormap([COLORS[k] for k in range(9)])
    norm = BoundaryNorm(np.arange(-0.5, 9.5), cmap.N)
    observed = set()
    summaries = {}
    for axis, name in zip(axes, ("equilibrium", "field")):
        data = read_csv(data_dir / f"{name}_map.csv")
        x, ratio = np.unique(data["x"]), np.unique(data["source_ratio"])
        if len(data) != len(x) * len(ratio):
            raise ValueError(f"Incomplete {name} map")
        if not np.allclose(data["x"], np.tile(x, len(ratio))) or not np.allclose(
            data["source_ratio"], np.repeat(ratio, len(x))
        ):
            raise ValueError(f"Unexpected {name} grid order")
        found = data["types_found"].reshape(len(ratio), len(x)).astype(int)
        unknown = data["unresolved_types"].reshape(found.shape).astype(int)
        if np.any((found < 0) | (found > 7)):
            raise ValueError(f"Invalid {name} type mask")
        category = np.where((found == 0) & (unknown > 0), 8, found)
        observed.update(np.unique(category))
        axis.set_yscale("log")
        axis.pcolormesh(
            x, ratio, category, shading="nearest", cmap=cmap, norm=norm, rasterized=True
        )
        axis.set_xlim(x[0], x[-1])
        axis.set_ylim(ratio[0], ratio[-1])
        axis.set_yticks([0.5, 1, 2, 4, 8, 16])
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.set_title(
            "Zero-current closure  ·  $J_z=0$"
            if name == "equilibrium"
            else "Prescribed-field closure  ·  α = 20°",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        axis.set_xlabel(
            r"Solar elevation $\alpha$ [deg]"
            if name == "equilibrium"
            else r"Boundary field $E_H$ [V/m]"
        )
        if name == "equilibrium":
            axis.set_ylabel(r"Source ratio $r=n_{pe,ref}/n_i$")
            for branch, alpha in (("A", 60), ("B", 20), ("C", 10)):
                axis.scatter(
                    [alpha], [64 / 8.7], s=35, c="white", edgecolors="#202B34", zorder=4
                )
                axis.annotate(
                    branch,
                    (alpha, 64 / 8.7),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    weight="bold",
                    fontsize=10,
                    color="#17232B",
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                        "pad": 1,
                    },
                )
        else:
            axis.axvline(0, color="#34454F", linewidth=0.8, linestyle="--")
        summaries[name] = {
            "grid": [len(x), len(ratio)],
            "displayed_category_counts": {
                LABELS[int(k)]: int(np.count_nonzero(category == k))
                for k in np.unique(category)
            },
            "cells_with_unresolved_search": int(np.count_nonzero(unknown)),
            "maximum_candidates_reported": int(np.max(data["candidate_count"])),
        }
    handles = [
        Patch(facecolor=COLORS[k], edgecolor="none", label=LABELS[k])
        for k in (1, 2, 4, 3, 5, 6, 7, 0, 8)
        if k in observed
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.11),
        ncol=len(handles),
        frameon=False,
        title="Types found at each grid point",
    )
    fig.text(
        0.08,
        0.055,
        "Coexisting types are shown together; no stability selection is made. "
        "Gray cells do not establish nonexistence.",
        fontsize=10,
        color="#53616D",
    )
    fig.text(
        0.08,
        0.025,
        f"Finite multistart search, {len(x)} × {len(ratio)} points per panel. "
        "The A/B/C markers locate the 1D examples in the left panel.",
        fontsize=10,
        color="#53616D",
    )
    save_figure(fig, output_dir, "sheath_type_maps")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/figures/data"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263743",
            "text.color": "#263743",
            "xtick.color": "#455560",
            "ytick.color": "#455560",
            "pdf.fonttype": 42,
        }
    )
    summary = {
        "fixed_inputs": {
            "ion_density_m3": 8.7e6,
            "electron_temperature_ev": 12.0,
            "photoelectron_temperature_ev": 2.2,
            "solar_wind_speed_mps": 468e3,
            "electron_drift_mode": "zero",
            "ion_drift_mode": "normal",
            "profile_photoelectron_reference_density_m3": 64e6,
            "field_map_sun_elevation_deg": 20.0,
        },
        "profiles": plot_profiles(args.data, args.output),
        "maps": plot_maps(args.data, args.output),
    }
    (args.output / "data").mkdir(exist_ok=True)
    (args.output / "data" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
