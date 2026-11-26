"""Plot force and acceleration curves from the 6-DOF tiltrotor model."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.frames import rotation_body_to_earth
from src.dynamics.nacelle import TILT_CONVENTION, deflection_rad_from_tilt_deg
from src.dynamics.rotor import tilted_rotor_position


def build_dynamics():
    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft)
    rotor = dynamics.rotor
    hover_omega = np.sqrt(dynamics.mass * aircraft.gravity / (len(aircraft.motor_positions) * rotor.k_thrust))
    if hover_omega > rotor.max_omega:
        raise ValueError("estimated maximum rotor thrust is insufficient for hover")
    return dynamics, hover_omega, aircraft


def evaluate_point(dynamics, omega, airspeed, tilt_rad, aircraft):
    attitude = np.zeros(3)
    velocity_earth = np.array([airspeed, 0.0, 0.0])
    rotor_force_body = np.zeros(3)
    rotor_moment_body = np.zeros(3)
    thrust_each = dynamics.rotor.compute_thrust(omega, height_above_ground=100.0)
    for position in aircraft.motor_positions:
        x, y, z = position[:3]
        tilts = position[3]
        rotor_tilt = tilt_rad if tilts else 0.0
        if tilts:
            x, y, z = tilted_rotor_position(position, rotor_tilt, aircraft.tilt_pivot_xz_m)
        rotor_force = thrust_each * np.array([
            np.sin(rotor_tilt),
            0.0,
            -np.cos(rotor_tilt),
        ])
        rotor_force_body += rotor_force
        rotor_moment_body += np.cross(np.array([x, y, z]), rotor_force)

    aero_force_body, aero_moment_body, _, _, _ = (
        dynamics.aero.compute_forces_and_moments(
            velocity_earth,
            np.zeros(3),
            attitude,
            total_thrust=0.0,
            tilt_angle=tilt_rad,
        )
    )
    total_force_body = rotor_force_body + aero_force_body
    force_ned = rotation_body_to_earth(*attitude) @ total_force_body
    acceleration_ned = force_ned / dynamics.mass + np.array(
        [0.0, 0.0, aircraft.gravity]
    )
    moment_body = rotor_moment_body + aero_moment_body
    return {
        "rotor_forward_n": float(rotor_force_body[0]),
        "rotor_up_n": float(-rotor_force_body[2]),
        "total_forward_n": float(force_ned[0]),
        "total_up_n": float(-force_ned[2]),
        "accel_forward_mps2": float(acceleration_ned[0]),
        "accel_up_mps2": float(-acceleration_ned[2]),
        "pitch_moment_nm": float(moment_body[1]),
    }


def main():
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    dynamics, hover_omega, aircraft = build_dynamics()
    tilt_deg = np.linspace(0.0, 90.0, 46)
    tilt_rad = deflection_rad_from_tilt_deg(tilt_deg)
    airspeeds = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    results = {
        key: np.empty((airspeeds.size, tilt_deg.size), dtype=float)
        for key in (
            "rotor_forward_n",
            "rotor_up_n",
            "total_forward_n",
            "total_up_n",
            "accel_forward_mps2",
            "accel_up_mps2",
            "pitch_moment_nm",
        )
    }

    for speed_index, airspeed in enumerate(airspeeds):
        for tilt_index, tilt in enumerate(tilt_rad):
            point = evaluate_point(dynamics, hover_omega, airspeed, tilt, aircraft)
            for key, value in point.items():
                results[key][speed_index, tilt_index] = value

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for speed_index, airspeed in enumerate(airspeeds):
        label = f"{airspeed:g} m/s"
        axes[0, 0].plot(
            tilt_deg, results["rotor_forward_n"][speed_index], label=label
        )
        axes[0, 1].plot(
            tilt_deg, results["rotor_up_n"][speed_index], label=label
        )
        axes[1, 0].plot(
            tilt_deg, results["total_forward_n"][speed_index], label=label
        )
        axes[1, 1].plot(
            tilt_deg, results["total_up_n"][speed_index], label=label
        )

    axes[0, 0].set_title("Tilting-rotor forward thrust")
    axes[0, 1].set_title("Total rotor upward thrust")
    axes[1, 0].set_title("Total forward force")
    axes[1, 1].set_title("Total upward force")
    axes[0, 0].set_ylabel("Force (N)")
    axes[0, 1].set_ylabel("Force (N)")
    axes[1, 0].set_ylabel("Force (N)")
    axes[1, 1].set_ylabel("Force (N)")
    for axis in axes.flat:
        axis.set_xlabel(f"Rotor angle ({TILT_CONVENTION})")
        axis.grid      (True, alpha=0.3)
        axis.legend    (title="Airspeed")
    fig.suptitle(
        "Force-model curves at fixed hover RPM (level attitude; wing-only aerodynamics)"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "transition_force_curves.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    speed_grid, tilt_grid = np.meshgrid(airspeeds, tilt_deg, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, key, title, color_map in (
        (axes[0], "accel_forward_mps2", "Forward acceleration", "coolwarm"),
        (axes[1], "accel_up_mps2", "Upward acceleration", "coolwarm"),
        (axes[2], "pitch_moment_nm", "Pitch moment", "viridis"),
    ):
        contour = axis.contourf(
            tilt_grid,
            speed_grid,
            results[key],
            levels=18,
            cmap=color_map,
        )
        fig.colorbar(contour, ax=axis, label=(
            "Moment (N m)" if key == "pitch_moment_nm" else "Acceleration (m/s²)"
        ))
        axis.set_title(title)
        axis.set_xlabel("Tilt (deg)")
        axis.grid(True, alpha=0.2)
    axes[0].set_ylabel("Airspeed (m/s)")
    fig.suptitle(
        "Force-model maps (fixed hover RPM; no propwash; not a trim corridor)"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "transition_force_model_maps.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved figures/transition_force_curves.png")
    print("Saved figures/transition_force_model_maps.png")
    print(
        f"Assumptions: {dynamics.mass:g} kg, {dynamics.aero.wing_area:g} m² wing, "
        f"zero wind, level attitude, no propwash, "
        f"{len(aircraft.motor_positions)} rotors fixed at {hover_omega:.1f} rad/s."
    )
    print("These are instantaneous model outputs, not a trimmed or flight-qualified corridor.")


if __name__ == "__main__":
    main()
