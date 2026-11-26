"""Sweep transition speed-envelope sensitivity to wing lift and rotor thrust."""

import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.transition import transition_speed_envelope
from src.dynamics.nacelle import tilt_deg_from_deflection_deg


def wing_lift_thrust_sensitivity(
    aircraft,
    *,
    lift_scales=(0.8, 1.0, 1.2),
    thrust_scales=(0.4, 0.55, 0.7, 1.0),
    samples=46,
):
    """Return quasi-steady envelope sensitivities, not a trajectory or validation."""
    lift_scales = np.asarray(lift_scales, dtype=float)
    thrust_scales = np.asarray(thrust_scales, dtype=float)
    if (
        lift_scales.ndim != 1
        or thrust_scales.ndim != 1
        or lift_scales.size == 0
        or thrust_scales.size == 0
        or not np.isfinite(lift_scales).all()
        or not np.isfinite(thrust_scales).all()
        or np.any(lift_scales <= 0.0)
        or np.any(thrust_scales <= 0.0)
    ):
        raise ValueError("lift and thrust scales must be finite positive 1D sequences")

    cl_max = (
        aircraft.wing_lift_coefficient_zero
        + aircraft.wing_lift_slope * np.deg2rad(aircraft.wing_stall_angle_deg)
    )
    envelopes = {}
    for lift_scale in lift_scales:
        for thrust_scale in thrust_scales:
            envelopes[(float(lift_scale), float(thrust_scale))] = (
                transition_speed_envelope(
                    mass=aircraft.mass,
                    wing_area=aircraft.wing_area,
                    max_thrust_per_rotor=(
                        aircraft.max_thrust_per_rotor * thrust_scale
                    ),
                    rotor_tilts=[
                        position[3] for position in aircraft.motor_positions
                    ],
                    rho=aircraft.air_density,
                    cl_max=cl_max * lift_scale,
                    cd_0=aircraft.wing_drag_coeff,
                    aspect_ratio=aircraft.wing_aspect_ratio,
                    oswald_efficiency=aircraft.oswald_efficiency,
                    samples=samples,
                    gravity=aircraft.gravity,
                    rear_thrust_reserve_fraction=(
                        aircraft.rear_thrust_reserve_fraction
                    ),
                )
            )

    return {
        "lift_scales": lift_scales,
        "thrust_scales": thrust_scales,
        "tilt_deg": next(iter(envelopes.values()))["tilt_deg"],
        "minimum_speed_mps": np.stack([
            envelopes[(float(lift), float(thrust))]["minimum_speed_mps"]
            for lift in lift_scales
            for thrust in thrust_scales
        ]).reshape(lift_scales.size, thrust_scales.size, samples),
        "feasible": np.stack([
            envelopes[(float(lift), float(thrust))]["feasible"]
            for lift in lift_scales
            for thrust in thrust_scales
        ]).reshape(lift_scales.size, thrust_scales.size, samples),
    }


def write_sensitivity_csv(result, path):
    """Write one row per lift/thrust assumption and tilt sample."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "lift_scale",
            "thrust_scale",
            "tilt_deg",
            "minimum_speed_mps",
            "feasible",
        ])
        for lift_index, lift_scale in enumerate(result["lift_scales"]):
            for thrust_index, thrust_scale in enumerate(result["thrust_scales"]):
                for tilt_index, tilt_deg in enumerate(result["tilt_deg"]):
                    speed = result["minimum_speed_mps"][
                        lift_index, thrust_index, tilt_index
                    ]
                    writer.writerow([
                        lift_scale,
                        thrust_scale,
                        tilt_deg_from_deflection_deg(tilt_deg),
                        speed,
                        result["feasible"][
                            lift_index, thrust_index, tilt_index
                        ],
                    ])


def main():
    aircraft = load_aircraft_parameters()
    result = wing_lift_thrust_sensitivity(aircraft)
    output_path = Path("figures") / "wing_lift_thrust_sensitivity.csv"
    write_sensitivity_csv(result, output_path)
    print(f"Saved {output_path}")
    print(
        "Sensitivity model: quasi-steady speed envelope with scaled CLmax "
        "and estimated per-rotor thrust; not measured aircraft performance."
    )


if __name__ == "__main__":
    main()
