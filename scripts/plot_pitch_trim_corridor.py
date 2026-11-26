"""Plot the current steady trim corridor and its pointwise constraint checks."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "figures" / "pitch_trim_authority.csv"
OUTPUT_DIR = ROOT / "figures"


def _read_rows(path=INPUT_CSV):
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"pitch-trim map has no data rows: {path}")
    required = {
        "airspeed_mps",
        "tilt_deg",
        "trim_feasible",
        "point_screen_status",
        "point_screen_reasons",
        "trim_and_rotor_constraints_feasible",
        "corridor_qualified",
        "tail_deflection_deg",
        "tail_authority_margin_deg",
        "front_motor_rpm",
        "front_rpm_limit",
        "front_rpm_margin",
        "rear_motor_rpm",
        "rear_rpm_limit_reserve_adjusted",
        "rear_rpm_margin",
        "max_normalized_residual",
        "cm_alpha_per_rad",
        "cl_beta_per_rad",
        "cl_beta_assessed",
        "cn_beta_per_rad",
        "static_stability_status",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"pitch-trim map is missing columns: {sorted(missing)}")
    return rows


def _number(row, key):
    value = float(row[key])
    return value if np.isfinite(value) else np.nan


def _flag(row, key):
    value = row[key].strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{key} must contain True or False, got {row[key]!r}")
    return value == "true"


def _plot_corridor(rows, path):
    categories = (
        ("No feasible point trim", "no_trim", "x", "0.55"),
        (
            "Point screen failed",
            "point_failed",
            "X",
            "tab:red",
        ),
        (
            "Point screen incomplete",
            "point_incomplete",
            "o",
            "tab:orange",
        ),
        ("Point screen passed; path filters remain", "point_pass", "s", "tab:blue"),
        ("Fully corridor-qualified", "qualified", "*", "tab:green"),
    )
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    for label, category, marker, color in categories:
        selected = []
        for row in rows:
            trim = _flag(row, "trim_feasible")
            qualified = _flag(row, "corridor_qualified")
            point_status = row["point_screen_status"].strip().lower()
            if category == "no_trim":
                match = not trim
            elif category == "point_failed":
                match = trim and point_status == "fail"
            elif category == "point_incomplete":
                match = trim and point_status == "incomplete"
            elif category == "point_pass":
                match = trim and point_status == "pass" and not qualified
            else:
                match = qualified
            if match:
                selected.append(row)
        scatter_options = {
                    "marker": marker,
                    "s": 85 if marker != "*" else 150,
                    "color": color,
                    "linewidths": 0.8,
                    "label": f"{label} ({len(selected)})",
                    "zorder": 3,
        }
        if marker != "x":
                    scatter_options["edgecolors"] = "black"
        axis.scatter(
                    [_number(row, "tilt_deg") for row in selected],
                    [_number(row, "airspeed_mps") for row in selected],
                    **scatter_options,
        )

    tilts = sorted({_number(row, "tilt_deg") for row in rows})
    speeds = sorted({_number(row, "airspeed_mps") for row in rows})
    axis.set_xticks(tilts)
    axis.set_yticks(speeds)
    axis.set_xlim(min(tilts) - 5, max(tilts) + 5)
    axis.set_ylim(min(speeds) - 1, max(speeds) + 1)
    axis.set_xlabel("Nacelle tilt (deg)")
    axis.set_ylabel("Airspeed (m/s)")
    axis.set_title("Steady point-trim corridor and constraint screening")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    figure.suptitle(
        "Current aircraft model — evaluated grid points only",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "Not a continuous or dynamic reachable corridor. No optimized path is shown: "
        "the pitch-balanced trajectory is infeasible at the configured endpoint.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.05, 0.78, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_diagnostics(rows, path):
    trim_rows = [row for row in rows if _flag(row, "trim_feasible")]
    stability_rows = [
        row for row in trim_rows
        if _number(row, "airspeed_mps") >= 0.5
    ]
    tilt_values = np.array([_number(row, "tilt_deg") for row in trim_rows])
    norm = Normalize(vmin=0.0, vmax=90.0)
    cmap = plt.get_cmap("viridis")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    tail_axis, rpm_axis, residual_axis, stability_axis = axes.flat

    for tilt in sorted(set(tilt_values)):
        series = sorted(
            (row for row in trim_rows if _number(row, "tilt_deg") == tilt),
            key=lambda row: _number(row, "airspeed_mps"),
        )
        tail_axis.plot(
            [_number(row, "airspeed_mps") for row in series],
            [_number(row, "tail_deflection_deg") for row in series],
            "-o",
            color=cmap(norm(tilt)),
            label=f"{tilt:g}° tilt",
            markersize=5,
        )
    if trim_rows:
        tail_limit = max(
            abs(_number(row, "tail_deflection_deg"))
            + _number(row, "tail_authority_margin_deg")
            for row in trim_rows
        )
        tail_axis.axhline(tail_limit, color="black", linestyle=":", linewidth=1)
        tail_axis.axhline(-tail_limit, color="black", linestyle=":", linewidth=1)
    else:
        tail_axis.text(
            0.5,
            0.5,
            "No trim-feasible points",
            transform=tail_axis.transAxes,
            ha="center",
        )
    tail_axis.set(
        xlabel="Airspeed (m/s)",
        ylabel="Required V-tail deflection (deg)",
        title="Trim tail demand (feasible trim points only)",
    )
    tail_axis.grid(True, alpha=0.3)
    tail_axis.legend(fontsize=8)

    for row in trim_rows:
        speed = _number(row, "airspeed_mps")
        tilt = _number(row, "tilt_deg")
        color = cmap(norm(tilt))
        rpm_axis.scatter(
            speed,
            _number(row, "front_rpm_margin"),
            marker="o",
            color=color,
            edgecolors="black",
            s=55,
        )
        rpm_axis.scatter(
            speed,
            _number(row, "rear_rpm_margin"),
            marker="^",
            color=color,
            edgecolors="black",
            s=55,
        )
    rpm_axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    rpm_axis.set(
        xlabel="Airspeed (m/s)",
        ylabel="RPM margin to configured limit",
        title="RPM margin (rear limit includes reserve)",
    )
    rpm_axis.grid(True, alpha=0.3)
    rpm_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color="tab:gray", label="Front"),
            Line2D([], [], marker="^", linestyle="", color="tab:gray", label="Rear"),
        ],
        loc="best",
    )
    figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=rpm_axis,
        label="Nacelle tilt (deg)",
    )
    residual_values = np.array([
        _number(row, "max_normalized_residual") for row in rows
    ])
    residual_floor = max(float(np.nanmin(residual_values)), 1e-8)
    residual_ceiling = max(float(np.nanmax(residual_values)), residual_floor * 10)
    residual_scatter = residual_axis.scatter(
        [_number(row, "tilt_deg") for row in rows],
        [_number(row, "airspeed_mps") for row in rows],
        c=np.maximum(residual_values, residual_floor),
        norm=Normalize(vmin=residual_floor, vmax=residual_ceiling),
        cmap="magma",
        marker="s",
        s=65,
        edgecolors="black",
        linewidths=0.4,
    )
    residual_axis.set(
        xlabel="Nacelle tilt (deg)",
        ylabel="Airspeed (m/s)",
        title="Best-fit trim residual (acceptance threshold: 0.02)",
    )
    residual_axis.set_xticks(sorted({_number(row, "tilt_deg") for row in rows}))
    residual_axis.set_yticks(sorted({_number(row, "airspeed_mps") for row in rows}))
    residual_axis.grid(True, alpha=0.25)
    figure.colorbar(
        residual_scatter,
        ax=residual_axis,
        label="Maximum normalized residual (N and pitch moment)",
    )
    left_axis = stability_axis
    right_axis = left_axis.twinx()
    airspeeds = [_number(row, "airspeed_mps") for row in stability_rows]
    cl_beta_rows = [
        row for row in stability_rows if _flag(row, "cl_beta_assessed")
    ]
    left_axis.scatter(
        airspeeds,
        [_number(row, "cm_alpha_per_rad") for row in stability_rows],
        marker="o",
        color="tab:blue",
        label="Cm_alpha",
    )
    if cl_beta_rows:
        right_axis.scatter(
            [_number(row, "airspeed_mps") for row in cl_beta_rows],
            [_number(row, "cl_beta_per_rad") for row in cl_beta_rows],
            marker="s",
            color="tab:orange",
            label="Cl_beta",
        )
    else:
        right_axis.text(
            0.98,
            0.98,
            "Cl_beta unassessed\n(no roll-moment model)",
            transform=right_axis.transAxes,
            ha="right",
            va="top",
            color="tab:orange",
        )
    right_axis.scatter(
        airspeeds,
        [_number(row, "cn_beta_per_rad") for row in stability_rows],
        marker="^",
        color="tab:green",
        label="Cn_beta",
    )
    left_axis.axhline(0.0, color="tab:blue", linestyle=":", linewidth=0.8)
    right_axis.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    left_axis.set_xlabel("Airspeed (m/s)")
    left_axis.set_ylabel("Cm_alpha (1/rad)", color="tab:blue")
    right_axis.set_ylabel("Cl_beta / Cn_beta (1/rad)")
    left_axis.set_title(
        "Static derivatives at trim-assessed points (Cl_beta unassessed)"
    )
    left_axis.grid(True, alpha=0.3)
    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="best",
        fontsize=8,
    )
    figure.suptitle(
        "Pitch-trim and rotor diagnostics — sparse model grid",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    rows = _read_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corridor_path = OUTPUT_DIR / "pitch_trim_transition_corridor.png"
    diagnostics_path = OUTPUT_DIR / "pitch_trim_constraint_diagnostics.png"
    _plot_corridor(rows, corridor_path)
    _plot_diagnostics(rows, diagnostics_path)
    print(f"Plotted {len(rows)} speed/tilt grid points.")
    print(f"Corridor map: {corridor_path}")
    print(f"Constraint diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
