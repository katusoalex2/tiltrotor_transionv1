"""Map steady longitudinal pitch-trim feasibility over speed and tilt."""

import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.transition import constrained_transition_corridor


AIR_SPEEDS_MPS = np.arange(0.0, 12.0 + 1e-9, 2.0)
TILT_ANGLES_DEG = np.arange(0.0, 90.0 + 1e-9, 15.0)


def build_pitch_trim_map():
    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft)
    tilt_rate_deg_s = float(np.rad2deg(aircraft.max_tilt_rate))
    transition_duration_s = 90.0 / tilt_rate_deg_s
    cl_max = (
        aircraft.wing_lift_coefficient_zero
        + aircraft.wing_lift_slope * np.deg2rad(aircraft.wing_stall_angle_deg)
    )
    corridor = constrained_transition_corridor(
        dynamics,
        max_thrust_per_rotor=aircraft.max_thrust_per_rotor,
        rotor_tilts=[position[3] for position in aircraft.motor_positions],
        airspeeds_mps=AIR_SPEEDS_MPS,
        tilt_angles_deg=TILT_ANGLES_DEG,
        max_tilt_rate_deg_s=tilt_rate_deg_s,
        transition_duration_s=transition_duration_s,
        rear_thrust_reserve_fraction=aircraft.rear_thrust_reserve_fraction,
        electrical_parameters=aircraft.electrical_parameters,
        tail_limit_rad=aircraft.vtail_max_deflection_rad,
        cl_max=cl_max,
        cd_0=aircraft.wing_drag_coeff,
        aspect_ratio=aircraft.wing_aspect_ratio,
        oswald_efficiency=aircraft.oswald_efficiency,
    )
    stability = corridor["stability_map"]
    rows = []
    for speed_index, airspeed_mps in enumerate(AIR_SPEEDS_MPS):
        for tilt_index, tilt_deg in enumerate(TILT_ANGLES_DEG):
            trim_feasible = bool(stability["trim_feasible"][speed_index, tilt_index])
            rows.append({
                "airspeed_mps": float(airspeed_mps),
                "tilt_deg": float(tilt_deg),
                "trim_feasible": trim_feasible,
                "alpha_deg": float(
                    stability["trim_alpha_deg"][speed_index, tilt_index]
                ),
                "tail_deflection_deg": float(
                    stability["trim_tail_deflection_deg"][speed_index, tilt_index]
                ),
                "tail_authority_margin_deg": float(
                    stability["tail_authority_margin_deg"][speed_index, tilt_index]
                ),
                "tail_deflection_saturated": bool(
                    stability["tail_deflection_saturated"][speed_index, tilt_index]
                ),
                "front_motor_rpm": float(
                    stability["trim_front_motor_rpm"][speed_index, tilt_index]
                ),
                "front_rpm_limit": float(
                    corridor["trim_front_rpm_limit"][speed_index, tilt_index]
                ),
                "front_rpm_margin": float(
                    corridor["trim_front_rpm_margin"][speed_index, tilt_index]
                ),
                "front_rpm_feasible": bool(
                    corridor["front_rpm_mask"][speed_index, tilt_index]
                ),
                "rear_motor_rpm": float(
                    stability["trim_rear_motor_rpm"][speed_index, tilt_index]
                ),
                "rear_rpm_limit_reserve_adjusted": float(
                    corridor["trim_rear_rpm_limit"][speed_index, tilt_index]
                ),
                "rear_rpm_margin": float(
                    corridor["trim_rear_rpm_margin"][speed_index, tilt_index]
                ),
                "rear_rpm_feasible": bool(
                    corridor["rear_rpm_mask"][speed_index, tilt_index]
                ),
                "rear_thrust_per_rotor_n": float(
                    corridor["trim_rear_thrust_per_rotor_n"][speed_index, tilt_index]
                ),
                "rear_thrust_limit_per_rotor_n": float(
                    corridor["rear_thrust_limit_per_rotor_n"]
                ),
                "rear_thrust_margin_per_rotor_n": float(
                    corridor["rear_thrust_margin_per_rotor_n"][speed_index, tilt_index]
                ),
                "rear_thrust_reserve_feasible": bool(
                    corridor["rear_thrust_mask"][speed_index, tilt_index]
                ),
                "electrical_assessed": bool(
                    corridor["electrical_assessed_mask"][speed_index, tilt_index]
                ),
                "electrical_feasible": bool(
                    corridor["electrical_feasible_mask"][speed_index, tilt_index]
                ),
                "electrical_missing_parameters": ";".join(
                    corridor["electrical_missing_parameters"]
                ),
                "point_screen_status": str(
                    corridor["point_screen_status"][speed_index, tilt_index]
                ),
                "point_screen_reasons": str(
                    corridor["point_screen_reasons"][speed_index, tilt_index]
                ),
                "trim_and_rotor_constraints_feasible": bool(
                    trim_feasible
                    and corridor["rotor_rpm_mask"][speed_index, tilt_index]
                    and corridor["rear_thrust_mask"][speed_index, tilt_index]
                    and corridor["electrical_assessed_mask"][speed_index, tilt_index]
                    and corridor["electrical_feasible_mask"][speed_index, tilt_index]
                ),
                "corridor_qualified": bool(
                    corridor["qualified_mask"][speed_index, tilt_index]
                ),
                "analytic_envelope_pass": bool(
                    corridor["analytic_mask"][speed_index, tilt_index]
                ),
                "gust_margin_pass": bool(
                    corridor["gust_mask"][speed_index, tilt_index]
                ),
                "tail_authority_pass": bool(
                    corridor["tail_authority_mask"][speed_index, tilt_index]
                ),
                "rotor_thrust_and_reserve_pass": bool(
                    corridor["rotor_thrust_mask"][speed_index, tilt_index]
                ),
                "pitch_moment_residual_pass": bool(
                    corridor["pitch_moment_mask"][speed_index, tilt_index]
                ),
                "static_stability_pass": bool(
                    corridor["static_stability_mask"][speed_index, tilt_index]
                ),
                "static_stability_assessed": bool(
                    corridor["static_stability_assessed_mask"][
                        speed_index, tilt_index
                    ]
                ),
                "static_stability_status": str(
                    corridor["static_stability_status"][speed_index, tilt_index]
                ),
                "known_static_stability_pass": bool(
                    corridor["known_static_stability_mask"][
                        speed_index, tilt_index
                    ]
                ),
                "cl_beta_assessed": bool(
                    stability["cl_beta_assessed"][speed_index, tilt_index]
                ),
                "tilt_rate_reachable": bool(
                    corridor["actuator_rate_mask"][speed_index, tilt_index]
                ),
                "known_corridor_constraints_pass": bool(
                    corridor["known_constraints_mask"][speed_index, tilt_index]
                ),
                "cm_alpha_per_rad": float(
                    stability["cm_alpha_per_rad"][speed_index, tilt_index]
                ),
                "cl_beta_per_rad": float(
                    stability["cl_beta_per_rad"][speed_index, tilt_index]
                ),
                "cn_beta_per_rad": float(
                    stability["cn_beta_per_rad"][speed_index, tilt_index]
                ),
                "motor_slew_assessed": bool(
                    corridor["motor_slew_assessed_mask"][speed_index, tilt_index]
                ),
                "motor_slew_feasible": bool(
                    corridor["motor_slew_feasible_mask"][speed_index, tilt_index]
                ),
                "longitudinal_residual_normalized": float(
                    stability["trim_longitudinal_residual"][speed_index, tilt_index]
                ),
                "vertical_residual_normalized": float(
                    stability["trim_vertical_residual"][speed_index, tilt_index]
                ),
                "pitch_moment_residual_normalized": float(
                    stability["trim_pitch_moment_residual"][speed_index, tilt_index]
                ),
                "max_normalized_residual": float(
                    stability["trim_residual"][speed_index, tilt_index]
                ),
                "trim_solver_converged": bool(
                    stability["trim_solver_converged"][speed_index, tilt_index]
                ),
                "solver_message": str(
                    stability["trim_solver_message"][speed_index, tilt_index]
                ),
            })
    return aircraft, rows, corridor


def write_pitch_trim_map(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    aircraft, rows, corridor = build_pitch_trim_map()
    output = Path("figures") / "pitch_trim_authority.csv"
    write_pitch_trim_map(rows, output)
    feasible = sum(row["trim_feasible"] for row in rows)
    rotor_feasible = sum(
        row["trim_and_rotor_constraints_feasible"] for row in rows
    )
    corridor_feasible = int(np.count_nonzero(corridor["qualified_mask"]))
    known_constraints_pass = int(
        np.count_nonzero(corridor["known_constraints_mask"])
    )
    static_stability_assessed = int(
        np.count_nonzero(corridor["static_stability_assessed_mask"])
    )
    cl_beta_unassessed = int(np.count_nonzero(
        corridor["stability_map"]["assessment_mask"]
        & ~corridor["stability_map"]["cl_beta_assessed"]
    ))
    point_screen_counts = {
        status: sum(row["point_screen_status"] == status for row in rows)
        for status in ("pass", "fail", "incomplete")
    }
    electrical_assessed = int(
        np.count_nonzero(corridor["electrical_assessed_mask"])
    )
    print(
        f"Pitch trim feasible: {feasible}/{len(rows)} speed/tilt samples; "
        f"configured provisional V-tail travel: "
        f"+/-{aircraft.vtail_max_deflection_deg:.2f} deg"
    )
    print(
        "Point screen (steady point constraints only): "
        f"{point_screen_counts['pass']} pass, "
        f"{point_screen_counts['fail']} fail, "
        f"{point_screen_counts['incomplete']} incomplete."
    )
    print(
        f"Trim + rotor constraints (reserve, RPM, electrical) feasible: "
        f"{rotor_feasible}/{len(rows)}; electrical screen assessed at "
        f"{electrical_assessed}/{len(rows)} points."
    )
    print(
        f"Full static-stability assessment: {static_stability_assessed}/"
        f"{len(rows)} points; Cl_beta is unassessed at {cl_beta_unassessed} "
        "trim-assessed points. "
        f"Available static/actuator/trim constraints pass at "
        f"{known_constraints_pass}/{len(rows)} points."
    )
    print(
        f"Fully qualified corridor: {corridor_feasible}/{len(rows)}. "
        "Full qualification remains fail-closed when Cl_beta, motor slew, "
        "or electrical feasibility is unassessed."
    )
    print(
        "Model scope: quasi-steady grid and estimated electrical screen; "
        "not measured propulsion performance or dynamic pitch/control response."
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
