"""Compute finer static trim and classify independent replay evidence."""

import csv
from concurrent.futures import ProcessPoolExecutor

import casadi as ca
import numpy as np
from scipy.optimize import least_squares

from scripts.transition_study import (
    LOW_SPEED_FULL_LOAD, OUTPUT, TARGET_SPEED, TRIM_TOLERANCE, study_limits, write_csv,
)
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.nacelle import deflection_rad_from_tilt_deg
from src.dynamics.transition_trajectory import RPM_PER_RAD_S, longitudinal_loads


REPORT_DIR = OUTPUT / "refined_corridor"
SPEED_WIDTH = 0.5
ANGLE_WIDTH = 2.0
SPEEDS = np.arange(0, TARGET_SPEED + SPEED_WIDTH, SPEED_WIDTH)
ANGLES = np.arange(0, 90 + ANGLE_WIDTH, ANGLE_WIDTH)


def load_coarse_seeds():
    with (OUTPUT / "static_corridor.csv").open(newline="", encoding="utf-8") as stream:
        return {
            (int(row["airspeed_mps"]), int(row["front_tilt_deg"])): (
                np.deg2rad(float(row["pitch_deg"])),
                float(row["front_rpm_per_motor"]) / RPM_PER_RAD_S,
                float(row["rear_rpm_per_motor"]) / RPM_PER_RAD_S,
                np.deg2rad(float(row["vtail_deg"])),
            )
            for row in csv.DictReader(stream)
        }


def trim_slice(task):
    """Solve one row with exact longitudinal load derivatives and continuation."""
    angle, coarse_seeds = task
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    load = longitudinal_loads(aircraft, LOW_SPEED_FULL_LOAD)
    controls = ca.SX.sym("controls", 4)
    airspeed = ca.SX.sym("airspeed")
    tilt = deflection_rad_from_tilt_deg(angle)
    forces = load(ca.vertcat(airspeed, 0, 0, controls[0], 0), ca.vertcat(tilt, controls[1:]))
    trim_scale = [aircraft.gravity, aircraft.gravity, aircraft.mass * aircraft.gravity * aircraft.wing_mean_chord_m]
    residual = forces[:3] / ca.DM(trim_scale)
    value_fn = ca.Function("trim_residual", [controls, airspeed], [residual])
    jac_fn = ca.Function("trim_jacobian", [controls, airspeed], [ca.jacobian(residual, controls)])
    angle_bound = np.deg2rad(aircraft.wing_stall_angle_deg)
    lower = np.array([-angle_bound, 0.0, 0.0, -limits.tail_rad])
    upper = np.array([angle_bound, limits.front_omega, limits.rear_omega, limits.tail_rad])
    hover = np.sqrt(aircraft.mass * aircraft.gravity / (4 * aircraft.k_thrust))
    previous = np.array([0.0, hover, hover, 0.0])
    rows = []
    for speed in SPEEDS:
        neighbor = coarse_seeds[round(speed), round(angle / 5) * 5]
        attempts = []
        for seed in (neighbor, previous, (0.0, hover, hover, 0.0), (0.1, hover / 2, hover / 2, -0.1)):
            candidate = least_squares(
                lambda x: np.asarray(value_fn(x, speed)).ravel(),
                np.clip(seed, lower, upper),
                jac=lambda x: np.asarray(jac_fn(x, speed)),
                bounds=(lower, upper), x_scale=[angle_bound, hover, hover, limits.tail_rad],
                max_nfev=120, gtol=1e-9, ftol=1e-9, xtol=1e-9,
            )
            error = float(np.max(np.abs(candidate.fun)))
            attempts.append((error, candidate))
            if error <= TRIM_TOLERANCE:
                previous = candidate.x
                break
        error, best = min(attempts, key=lambda attempt: attempt[0])
        pitch, front, rear, tail = best.x
        rows.append({
            "airspeed_mps": speed,
            "front_tilt_deg": angle,
            "trim_found": error <= TRIM_TOLERANCE,
            "normalized_residual": error,
            "forward_force_residual_n": best.fun[0] * aircraft.mass * aircraft.gravity,
            "vertical_force_residual_n": best.fun[1] * aircraft.mass * aircraft.gravity,
            "pitch_moment_residual_nm": best.fun[2] * trim_scale[2],
            "pitch_deg": np.rad2deg(pitch),
            "front_pair_thrust_n": 2 * aircraft.k_thrust * front**2,
            "rear_pair_thrust_n": 2 * aircraft.k_thrust * rear**2,
            "front_rpm_per_motor": RPM_PER_RAD_S * front,
            "rear_rpm_per_motor": RPM_PER_RAD_S * rear,
            "vtail_deg": np.rad2deg(tail),
            "solver_converged": bool(best.success),
            "starts_attempted": len(attempts),
        })
    return rows


def cell(speed, angle):
    speed_index = int(np.floor(speed / SPEED_WIDTH + 0.5))
    angle_index = int(np.floor(angle / ANGLE_WIDTH + 0.5))
    if 0 <= speed_index < len(SPEEDS) and 0 <= angle_index < len(ANGLES):
        return SPEEDS[speed_index], ANGLES[angle_index]
    return None


def coverage(rows):
    """Return the bins with an actual accepted-route sample and their directed links."""
    samples = {}
    edges = set()
    prior = None
    for index, row in enumerate(rows):
        current = cell(row["airspeed_mps"], row["front_tilt_deg"])
        if current is not None:
            samples.setdefault(current, index)
            if prior is not None and current != prior:
                edges.add((prior, current))
        prior = current
    return samples, edges


def compute_static(workers=3):
    seeds = load_coarse_seeds()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for batch in pool.map(trim_slice, [(angle, seeds) for angle in ANGLES]) for row in batch]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_DIR / "static_trim.csv", rows)
    print(f"Static trim found at {sum(row['trim_found'] for row in rows)} / {len(rows)} centers.", flush=True)


if __name__ == "__main__":
    compute_static()
