"""Diagnose pitch-trim authority at the optimizer's terminal condition."""

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.rotor import tilted_rotor_position
from src.dynamics.state import AircraftState
from src.dynamics.transition import solve_longitudinal_trim
from src.dynamics.nacelle import CRUISE_TILT_DEG, deflection_rad_from_tilt_deg


ENDPOINT_SPEED_MPS = 12.0
ENDPOINT_TILT_DEG = CRUISE_TILT_DEG
ADDITIONAL_ASSUMED_TAIL_LIMITS_DEG = (15.0, 20.0, 30.0, 45.0)


def _moment_budget(aircraft, dynamics, trim, speed_mps, tilt_deg, tail_limit_deg):
    tilt_rad = deflection_rad_from_tilt_deg(tilt_deg)
    alpha_rad = np.deg2rad(trim["alpha_deg"])
    tail_rad = np.deg2rad(trim["tail_deflection_deg"])
    front_omega = float(trim["front_motor_rpm"]) * 2.0 * np.pi / 60.0
    rear_omega = float(trim["rear_motor_rpm"]) * 2.0 * np.pi / 60.0
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([speed_mps, 0.0, 0.0]),
        att=np.array([0.0, alpha_rad, 0.0]),
        tilt=tilt_rad,
    )

    rotor_moments = {"front": 0.0, "rear": 0.0}
    rotor_thrusts = []
    for position in aircraft.motor_positions:
        x, y, z, is_tilting = position
        rotor_tilt = tilt_rad if is_tilting else 0.0
        omega = front_omega if is_tilting else rear_omega
        thrust = dynamics.rotor.compute_thrust(omega, height_above_ground=100.0)
        rotor_thrusts.append(thrust)
        if is_tilting:
            location = tilted_rotor_position(
                position,
                rotor_tilt,
                aircraft.tilt_pivot_xz_m,
            )
        else:
            location = np.array([x, y, z], dtype=float)
        rotor_force = thrust * np.array([
            np.sin(rotor_tilt),
            0.0,
            -np.cos(rotor_tilt),
        ])
        rotor_moments["front" if is_tilting else "rear"] += float(
            np.cross(location, rotor_force)[1]
        )

    wing_force, wing_moment, _, _, _ = dynamics.aero.compute_forces_and_moments(
        state.vel,
        np.zeros(3),
        state.att,
        float(sum(rotor_thrusts)),
        tilt_rad,
    )
    wing_force_arm_moment = float(np.cross(
        np.array([aircraft.wing_aerodynamic_center_x_m, 0.0, 0.0]),
        wing_force,
    )[1])
    wing_coefficient_moment = float(
        wing_moment[1] - wing_force_arm_moment
    )
    _, tail_moment = dynamics.aero.compute_tail_forces(
        state.vel,
        np.zeros(3),
        state.att,
        tail_rad,
        tail_rad,
    )
    negative_limit_moment = dynamics.aero.compute_tail_forces(
        state.vel,
        np.zeros(3),
        state.att,
        -np.deg2rad(tail_limit_deg),
        -np.deg2rad(tail_limit_deg),
    )[1][1]
    positive_limit_moment = dynamics.aero.compute_tail_forces(
        state.vel,
        np.zeros(3),
        state.att,
        np.deg2rad(tail_limit_deg),
        np.deg2rad(tail_limit_deg),
    )[1][1]

    moment_total = (
        float(wing_moment[1])
        + rotor_moments["front"]
        + rotor_moments["rear"]
        + float(tail_moment[1])
    )
    normalized_residual = np.asarray(trim["normalized_residual"], dtype=float)
    scale = aircraft.mass * aircraft.gravity * dynamics.aero.chord
    if not np.isclose(
        moment_total,
        normalized_residual[2] * scale,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise RuntimeError(
            "component pitch moments do not reproduce the trim solver residual"
        )
    effective_tail_alpha_rad = (
        alpha_rad
        + tail_rad * np.cos(np.deg2rad(aircraft.vtail_dihedral_deg))
    )
    tail_lift_coefficient = float(np.clip(
        aircraft.vtail_lift_slope * effective_tail_alpha_rad,
        -aircraft.vtail_max_lift_coefficient,
        aircraft.vtail_max_lift_coefficient,
    ))
    required_tail_moment = -(
        float(wing_moment[1])
        + rotor_moments["front"]
        + rotor_moments["rear"]
    )
    positive_tail_authority = max(
        float(negative_limit_moment),
        float(positive_limit_moment),
        0.0,
    )

    return {
        "endpoint_airspeed_mps": float(speed_mps),
        "endpoint_tilt_deg": float(tilt_deg),
        "assumed_tail_limit_deg": float(tail_limit_deg),
        "trim_feasible": bool(trim["feasible"]),
        "alpha_deg": float(trim["alpha_deg"]),
        "tail_deflection_deg": float(trim["tail_deflection_deg"]),
        "front_motor_rpm": float(trim["front_motor_rpm"]),
        "rear_motor_rpm": float(trim["rear_motor_rpm"]),
        "wing_pitch_moment_n_m": float(wing_moment[1]),
        "wing_force_arm_moment_n_m": wing_force_arm_moment,
        "wing_coefficient_moment_n_m": wing_coefficient_moment,
        "front_rotor_pitch_moment_n_m": rotor_moments["front"],
        "rear_rotor_pitch_moment_n_m": rotor_moments["rear"],
        "vtail_pitch_moment_n_m": float(tail_moment[1]),
        "vtail_effective_alpha_deg": float(np.rad2deg(effective_tail_alpha_rad)),
        "vtail_lift_coefficient": tail_lift_coefficient,
        "required_tail_moment_for_balance_n_m": float(required_tail_moment),
        "positive_tail_authority_at_limit_n_m": float(positive_tail_authority),
        "positive_tail_authority_shortfall_n_m": float(
            max(required_tail_moment - positive_tail_authority, 0.0)
        ),
        "pitch_moment_residual_n_m": moment_total,
        "tail_moment_at_negative_limit_n_m": float(negative_limit_moment),
        "tail_moment_at_positive_limit_n_m": float(positive_limit_moment),
        "longitudinal_force_residual_n": float(
            aircraft.mass * trim["acceleration_ned"][0]
        ),
        "vertical_force_residual_down_n": float(
            aircraft.mass * trim["acceleration_ned"][2]
        ),
        "longitudinal_residual_normalized": float(normalized_residual[0]),
        "vertical_residual_normalized": float(normalized_residual[1]),
        "pitch_moment_residual_normalized": float(normalized_residual[2]),
        "max_normalized_residual": float(trim["max_normalized_residual"]),
        "solver_message": str(trim["solver_message"]),
    }


def build_endpoint_budget():
    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft)
    tail_limits = sorted(set((
        aircraft.vtail_max_deflection_deg,
        *ADDITIONAL_ASSUMED_TAIL_LIMITS_DEG,
    )))
    rows = []
    for tail_limit_deg in tail_limits:
        trim = solve_longitudinal_trim(
            dynamics,
            ENDPOINT_SPEED_MPS,
            deflection_rad_from_tilt_deg(ENDPOINT_TILT_DEG),
            tail_limit_rad=np.deg2rad(tail_limit_deg),
        )
        rows.append(_moment_budget(
            aircraft,
            dynamics,
            trim,
            ENDPOINT_SPEED_MPS,
            ENDPOINT_TILT_DEG,
            tail_limit_deg,
        ))
    return aircraft, rows


def write_budget_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_budget(aircraft, rows, path):
    current = min(
        rows,
        key=lambda row: abs(
            row["assumed_tail_limit_deg"] - aircraft.vtail_max_deflection_deg
        ),
    )
    component_labels = (
        "Wing force arm",
        "Wing Cm",
        "Front rotors",
        "Rear rotors",
        "V-tail",
    )
    component_keys = (
        "wing_force_arm_moment_n_m",
        "wing_coefficient_moment_n_m",
        "front_rotor_pitch_moment_n_m",
        "rear_rotor_pitch_moment_n_m",
        "vtail_pitch_moment_n_m",
    )
    components = [current[key] for key in component_keys]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    moment_axis, sensitivity_axis = axes
    bars = moment_axis.bar(
        component_labels,
        components,
        color=("tab:blue", "tab:orange", "tab:green", "tab:purple"),
        edgecolor="black",
    )
    moment_axis.axhline(0.0, color="black", linewidth=1)
    moment_axis.axhline(
        current["pitch_moment_residual_n_m"],
        color="tab:red",
        linestyle="--",
        label=f"Net residual: {current['pitch_moment_residual_n_m']:.2f} N m",
    )
    moment_axis.set_ylabel("Pitch moment about CG (N m)")
    moment_axis.set_title(
        f"Endpoint moment budget at configured ±{aircraft.vtail_max_deflection_deg:.2f}°"
    )
    moment_axis.tick_params(axis="x", rotation=15)
    moment_axis.grid(axis="y", alpha=0.3)
    moment_axis.legend()
    for bar, value in zip(bars, components):
        moment_axis.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4 if value >= 0.0 else -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    limits = np.array([row["assumed_tail_limit_deg"] for row in rows])
    residuals = np.array([
        row["pitch_moment_residual_n_m"] for row in rows
    ])
    nose_up_authority = np.array([
        row["tail_moment_at_negative_limit_n_m"] for row in rows
    ])
    nose_down_authority = np.array([
        row["tail_moment_at_positive_limit_n_m"] for row in rows
    ])
    sensitivity_axis.axhspan(
        -0.02 * aircraft.mass * aircraft.gravity * aircraft.wing_mean_chord_m,
        0.02 * aircraft.mass * aircraft.gravity * aircraft.wing_mean_chord_m,
        color="tab:green",
        alpha=0.12,
        label=(
            "Trim residual tolerance "
            f"(reference chord {aircraft.wing_mean_chord_m:.2f} m)"
        ),
    )
    sensitivity_axis.plot(
        limits,
        residuals,
        "-o",
        color="tab:red",
        label="Best-fit net pitch-moment residual",
    )
    sensitivity_axis.plot(
        limits,
        nose_up_authority,
        "--s",
        color="tab:blue",
        label="Tail moment at negative travel limit",
    )
    sensitivity_axis.plot(
        limits,
        nose_down_authority,
        "--^",
        color="tab:purple",
        label="Tail moment at positive travel limit",
    )
    sensitivity_axis.axhline(0.0, color="black", linewidth=1)
    sensitivity_axis.set_xlabel("Assumed maximum V-tail travel (deg)")
    sensitivity_axis.set_ylabel("Pitch moment (N m)")
    sensitivity_axis.set_title("Endpoint trim sensitivity to assumed travel")
    sensitivity_axis.grid(True, alpha=0.3)
    sensitivity_axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        f"Best-fit trim diagnostics at {ENDPOINT_SPEED_MPS:g} m/s, "
        f"{ENDPOINT_TILT_DEG:g}° tilt",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Travel cases above the configured limit are hypothetical only; "
        "best-fit points are not feasible trims unless their residuals meet tolerance.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    aircraft, rows = build_endpoint_budget()
    output_dir = Path(__file__).resolve().parents[1] / "figures"
    csv_path = output_dir / "pitch_trim_endpoint_moment_budget.csv"
    plot_path = output_dir / "pitch_trim_endpoint_moment_budget.png"
    write_budget_csv(rows, csv_path)
    plot_budget(aircraft, rows, plot_path)
    print(
        f"Endpoint: {ENDPOINT_SPEED_MPS:g} m/s, "
        f"{ENDPOINT_TILT_DEG:g}° tilt; configured tail travel "
        f"+/-{aircraft.vtail_max_deflection_deg:.2f}°."
    )
    for row in rows:
        print(
            f"Assumed +/-{row['assumed_tail_limit_deg']:.2f} deg: "
            f"trim={row['trim_feasible']}, "
            f"tail={row['tail_deflection_deg']:.2f} deg, "
            f"pitch residual={row['pitch_moment_residual_n_m']:.2f} N m, "
            f"max normalized residual={row['max_normalized_residual']:.3f}"
        )
    configured = min(
        rows,
        key=lambda row: abs(
            row["assumed_tail_limit_deg"] - aircraft.vtail_max_deflection_deg
        ),
    )
    print(
        "Configured-case moments (N m): "
        f"wing={configured['wing_pitch_moment_n_m']:.3f}, "
        f"front={configured['front_rotor_pitch_moment_n_m']:.3f}, "
        f"rear={configured['rear_rotor_pitch_moment_n_m']:.3f}, "
        f"tail={configured['vtail_pitch_moment_n_m']:.3f}, "
        f"net={configured['pitch_moment_residual_n_m']:.3f}."
    )
    print(
        "At this best-fit state, tail balance requires "
        f"{configured['required_tail_moment_for_balance_n_m']:.3f} N m; "
        f"available nose-up tail moment is "
        f"{configured['positive_tail_authority_at_limit_n_m']:.3f} N m "
        f"(shortfall {configured['positive_tail_authority_shortfall_n_m']:.3f} N m)."
    )
    print(
        "Force residuals at the best-fit state: "
        f"forward={configured['longitudinal_force_residual_n']:.3f} N, "
        f"vertical-down={configured['vertical_force_residual_down_n']:.3f} N."
    )
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
