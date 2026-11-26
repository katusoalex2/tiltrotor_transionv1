"""Generate the analytical tilt-angle versus velocity envelope plot."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dynamics.transition import transition_speed_envelope
from src.dynamics.nacelle import tilt_deg_from_deflection_deg
from src.configuration.aircraft_parameters import load_aircraft_parameters


# Edit these strings to change the wording displayed on the graph.
PLOT_TITLE = "Tilt Angle vs Velocity"
X_AXIS_LABEL = "Forward speed (m/s)"
Y_AXIS_LABEL = "Tilt angle (deg; 90 = hover, 0 = cruise)"
CONVENTION_NOTE = (
    "Displayed angle: 90 deg hover to 0 deg cruise. " )
    

OUTPUT = ROOT / "figures" / "tilt_angle_vs_velocity.png"


def build_plot() -> Path:
    aircraft = load_aircraft_parameters()
    max_lift_coefficient = aircraft.wing_lift_coefficient_zero + aircraft.wing_lift_slope * np.deg2rad(aircraft.wing_stall_angle_deg)

    envelope = transition_speed_envelope(
        mass=aircraft.mass,
        wing_area=aircraft.wing_area,
        max_thrust_per_rotor=aircraft.max_thrust_per_rotor,
        rotor_tilts=[position[3] for position in aircraft.motor_positions],
        rho=aircraft.air_density,
        cl_max=max_lift_coefficient,
        cd_0=aircraft.wing_drag_coeff,
        aspect_ratio=aircraft.wing_aspect_ratio,
        oswald_efficiency=aircraft.oswald_efficiency,
        samples=901,
        gravity=aircraft.gravity,
    )

    displayed_tilt_deg = tilt_deg_from_deflection_deg(envelope["tilt_deg"])
    if not np.any(envelope["feasible"]):
        raise ValueError("no feasible points in the analytical envelope for the configured aircraft")
    max_speed = float(np.nanmax(envelope["maximum_speed_mps"]))
    speed_grid = np.linspace(0.0, max_speed, 1000)
    lower_tilt = np.full(speed_grid.shape, np.nan)
    upper_tilt = np.full(speed_grid.shape, np.nan)

    for index, speed in enumerate(speed_grid):
        feasible = (
            envelope["feasible"]
            & (speed >= envelope["minimum_speed_mps"])
            & (speed <= envelope["maximum_speed_mps"])
        )
        if np.any(feasible):
            angles = displayed_tilt_deg[feasible]
            lower_tilt[index] = float(np.min(angles))
            upper_tilt[index] = float(np.max(angles))

    valid = np.isfinite(lower_tilt) & np.isfinite(upper_tilt)
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    axis.fill_between(
        speed_grid[valid],
        lower_tilt[valid],
        upper_tilt[valid],
        color="0.78",
        label="Analytical rotor/wing envelope",
    )
    axis.plot(
        speed_grid[valid],
        lower_tilt[valid],
        "--",
        color="#2878b5",
        linewidth=2,
        label="Lower tilt bound",
    )
    axis.plot(
        speed_grid[valid],
        upper_tilt[valid],
        "--",
        color="#d64a4a",
        linewidth=2,
        label="Upper tilt bound",
    )

    reference_speed = np.linspace(0.0, 20.0, 20)
    reference_tilt = 90.0 * (1.0 - reference_speed / 20.0)
    axis.plot(
        reference_speed,
        reference_tilt,
        ":",
        color="black",
        linewidth=2.5,
        label="Project tilt-scaled reference (0-20 m/s)",
    )

    axis.set_xlabel(X_AXIS_LABEL)
    axis.set_ylabel(Y_AXIS_LABEL)
    axis.set_title(PLOT_TITLE)
    axis.set_xlim(0.0, max_speed)
    axis.set_ylim(0.0, 90.0)
    axis.set_yticks(np.arange(0.0, 91.0, 10.0))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left", framealpha=0.95)
    axis.text(
        0.99,
        0.02,
        f"{CONVENTION_NOTE}\n"
        f"Estimated speed={aircraft.max_omega:.0f} rad/s; "
        f"thrust={aircraft.max_thrust_per_rotor:.1f} N/rotor. Not flight validated.\n",
    
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.6",
            "alpha": 0.96,
        },
    )

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    print(build_plot())
