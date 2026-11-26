"""Plot provisional vertical-support boundaries across assumed rotor thrust limits."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.transition import rear_thrust_sensitivity_sweep


def main():
    aircraft = load_aircraft_parameters()
    result = rear_thrust_sensitivity_sweep(aircraft)
    thrust = np.asarray(result["max_thrust_per_rotor_n"], dtype=float)
    tilt = np.asarray(result["tilt_deg"], dtype=float)
    minimum_speed = np.asarray(result["minimum_speed_mps"], dtype=float)
    vertical_margin = np.asarray(result["vertical_margin_at_zero_speed_n"], dtype=float)
    feasible = np.asarray(result["feasible"], dtype=bool)
    threshold = float(result["rear_only_hover_threshold_per_rotor_n"])
    configured = float(result["configured_estimate_per_rotor_n"])

    output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "rear_thrust_sensitivity.csv")
    rows = np.column_stack([
        np.repeat(thrust, tilt.size),
        np.tile(tilt, thrust.size),
        minimum_speed.ravel(),
        vertical_margin.ravel(),
        feasible.ravel().astype(float),
    ])
    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header=(
            "max_thrust_per_rotor_n,tilt_deg,minimum_speed_mps,"
            "vertical_margin_at_zero_speed_n,envelope_feasible"
        ),
        comments="",
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    mesh = axes[0].pcolormesh(
        tilt,
        thrust,
        minimum_speed,
        shading="auto",
        cmap="viridis",
    )
    figure.colorbar(mesh, ax=axes[0], label="Minimum speed (m/s)")
    axes[0].axhline(threshold, color="white", linestyle="--", label="Rear-only threshold")
    axes[0].axhline(configured, color="red", linestyle=":", label="YAML-derived estimate")
    axes[0].set_xlabel("Nacelle tilt (deg)")
    axes[0].set_ylabel("Assumed max thrust per rotor (N)")
    axes[0].set_title("Provisional minimum-speed boundary")
    axes[0].legend()

    axes[1].plot(thrust, minimum_speed[:, -1], marker=".")
    axes[1].axvline(threshold, color="black", linestyle="--", label="Rear-only threshold")
    axes[1].axvline(configured, color="red", linestyle=":", label="YAML-derived estimate")
    axes[1].set_xlabel("Assumed max thrust per rotor (N)")
    axes[1].set_ylabel("Minimum speed at 90° tilt (m/s)")
    axes[1].set_title("Full-tilt boundary sensitivity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("Thrust-limit sensitivity only — not bench-validated")
    figure.tight_layout()
    plot_path = os.path.join(output_dir, "rear_thrust_sensitivity.png")
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Rear-only threshold: {threshold:.2f} N per rotor")
    print(f"YAML-derived estimate: {configured:.2f} N per rotor")
    print(f"Minimum speed at 90 degrees (YAML estimate): {minimum_speed[np.argmin(np.abs(thrust - configured)), -1]:.2f} m/s")
    print("Sensitivity results are not measured thrust or flight-validation evidence.")
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    return result


if __name__ == "__main__":
    main()
