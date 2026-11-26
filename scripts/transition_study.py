"""Build a static trim corridor and a dynamically constrained transition."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import lsq_linear

from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.nacelle import (
    CRUISE_TILT_DEG, TILT_CONVENTION, deflection_rad_from_tilt_deg,
    tilt_deg_from_deflection_deg, tilt_deg_from_deflection_rad,
)
from src.dynamics.state import AircraftState
from src.dynamics.transition import assess_rotor_electrical_point, solve_longitudinal_trim
from src.dynamics.transition_trajectory import (
    RPM_PER_RAD_S,
    TransitionLimits,
    longitudinal_loads,
    optimize_trajectory,
)


OUTPUT = Path(__file__).resolve().parents[1] / "figures" / "transition_study"
TARGET_SPEED = 30.0
DURATION = 32.0
TRIM_TOLERANCE = 0.005
LOW_SPEED_FULL_LOAD = 2.5


def study_limits(aircraft):
    """Derate the predicted loaded RPM with all four motors at equal speed."""
    lower, upper = 0.0, aircraft.estimated_motor_rpm
    for _ in range(38):
        rpm = (lower + upper) / 2
        point = assess_rotor_electrical_point(np.full(4, rpm), aircraft.electrical_parameters)
        if not point["assessed"]:
            raise ValueError(f"Missing electrical parameters: {point['missing_parameters']}")
        if point["feasible"]:
            lower = rpm
        else:
            upper = rpm
    front_omega = 0.90 * lower / RPM_PER_RAD_S
    return TransitionLimits(
        front_omega=front_omega,
        rear_omega=front_omega * np.sqrt(1.0 - aircraft.rear_thrust_reserve_fraction),
        tail_rad=aircraft.vtail_max_deflection_rad - np.deg2rad(1.0),
    )


def trim_guess(trim):
    return (
        np.deg2rad(trim["alpha_deg"]),
        trim["front_motor_rpm"] / RPM_PER_RAD_S,
        trim["rear_motor_rpm"] / RPM_PER_RAD_S,
        np.deg2rad(trim["tail_deflection_deg"]),
    )


def trim_point(dynamics, limits, speed, tilt, seeds=()):
    """Try the adjacent trim solutions when the default seed misses a point."""
    best = None
    for seed in (None, *seeds):
        result = solve_longitudinal_trim(
            dynamics, float(speed), float(tilt),
            tail_limit_rad=limits.tail_rad, max_rotor_omega=limits.front_omega,
            max_rear_rotor_omega=limits.rear_omega, residual_tolerance=TRIM_TOLERANCE,
            initial_guess=seed,
        )
        if best is None or result["max_normalized_residual"] < best["max_normalized_residual"]:
            best = result
        if best["feasible"]:
            break
    return best


def trajectory_guess(dynamics, limits, *, target_speed=TARGET_SPEED, steps=96):
    fraction = np.linspace(0.0, 1.0, steps + 1)
    speed = target_speed * fraction**2 * (3.0 - 2.0 * fraction)
    tilt = np.deg2rad(np.interp(speed, [0, 2, 4, 8, 12, 20, target_speed], [0, 12, 22, 41, 57, 78, 90]))
    motion = np.zeros((5, steps + 1))
    motion[0] = speed
    controls = np.empty((4, steps + 1))
    controls[0] = tilt
    seed = None
    for index in range(steps + 1):
        trim = trim_point(dynamics, limits, speed[index], tilt[index], (seed,) if seed else ())
        seed = trim_guess(trim)
        motion[3, index] = seed[0]
        controls[1:, index] = seed[1:]
    return np.vstack((motion, controls))


def solve_study_trajectory(aircraft, dynamics, limits, *, steps=96):
    guess = trajectory_guess(dynamics, limits, steps=steps)
    return optimize_trajectory(
        aircraft, limits, TARGET_SPEED, guess, duration_s=DURATION,
        low_speed_full_load_mps=dynamics.aero.low_speed_full_load_mps,
    )


def static_corridor(aircraft, dynamics, limits):
    speeds = range(35)
    tilts_deg = range(0, 91, 5)
    load = longitudinal_loads(aircraft, dynamics.aero.low_speed_full_load_mps)
    rows = []
    previous = {}
    for tilt_deg in tilts_deg:
        tilt = np.deg2rad(tilt_deg)
        adjacent = None
        for speed in speeds:
            seeds = [seed for seed in (adjacent, previous.get(speed)) if seed is not None]
            trim = trim_point(dynamics, limits, speed, tilt, seeds)
            pitch, front, rear, tail = trim_guess(trim)
            if trim["feasible"]:
                adjacent = previous[speed] = (pitch, front, rear, tail)
            result = np.asarray(load([speed, 0, 0, pitch, 0], [tilt, front, rear, tail])).ravel()
            measured_electrical = assess_rotor_electrical_point(
                [front * RPM_PER_RAD_S] * 2 + [rear * RPM_PER_RAD_S] * 2,
                aircraft.electrical_parameters,
            )
            rows.append({
                "airspeed_mps": speed,
                "front_tilt_deg": int(tilt_deg_from_deflection_deg(tilt_deg)),
                "static_trim_found": bool(trim["feasible"] and measured_electrical["feasible"]),
                "normalized_trim_residual": trim["max_normalized_residual"],
                "forward_force_residual_n": trim["acceleration_ned"][0] * aircraft.mass,
                "vertical_force_residual_up_n": -trim["acceleration_ned"][2] * aircraft.mass,
                "pitch_moment_residual_nm": trim["moment_body"][1],
                "pitch_deg": trim["alpha_deg"],
                "effective_alpha_deg": np.rad2deg(result[4]) if speed >= 0.5 else np.nan,
                "front_pair_thrust_n": result[5],
                "rear_pair_thrust_n": result[6],
                "front_rpm_per_motor": trim["front_motor_rpm"],
                "rear_rpm_per_motor": trim["rear_motor_rpm"],
                "vtail_deg": trim["tail_deflection_deg"],
                "tail_travel_remaining_deg": aircraft.vtail_max_deflection_deg - abs(trim["tail_deflection_deg"]),
                "electrical_pass": measured_electrical["feasible"],
            })
    return rows


def replay_trajectory(
    aircraft, dynamics, limits, time, motion, controls, *,
    step_s=0.02, planned_full_load_speed=LOW_SPEED_FULL_LOAD, stop_on_limit=False,
    plant_aircraft=None, wind_at=None, tilt_response_s=0.0, tail_reserve_rad=0.0, availability_at=None,
):
    """Track the plan in the rigid-body integrator using bounded load feedback."""
    load = longitudinal_loads(aircraft, dynamics.aero.low_speed_full_load_mps)
    planned_load = longitudinal_loads(aircraft, planned_full_load_speed)
    plant_load = load if plant_aircraft is None else longitudinal_loads(
        plant_aircraft, dynamics.aero.low_speed_full_load_mps, exact_kinematics=True,
    )
    state_symbol = ca.SX.sym("state", 5)
    actuator_symbol = ca.SX.sym("actuator", 4)
    gradient = ca.Function("load_gradient", [state_symbol, actuator_symbol], [
        ca.jacobian(load(state_symbol, actuator_symbol)[:3], actuator_symbol)[:, 1:],
    ])
    planned_loads = np.asarray(planned_load.map(time.size)(motion, controls))
    motion_rate = np.vstack((
        planned_loads[:2], motion[1], motion[4], planned_loads[2] / aircraft.inertia_yy,
    ))
    reference = CubicHermiteSpline(time, motion.T, motion_rate.T)
    sample_times = np.linspace(0.0, time[-1], int(np.ceil(time[-1] / step_s)) + 1)
    dt = sample_times[1]
    n = len(sample_times)
    actual_motion = np.empty((5, n))
    actual_control = np.empty((4, n))
    actual_load = np.empty((plant_load.size_out(0)[0] + 1, n))
    nominal_state = reference(sample_times).T
    nominal_rate = reference.derivative()(sample_times).T
    nominal_controls = np.vstack([np.interp(sample_times, time, control) for control in controls])
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))
    wind = np.zeros(3)
    previous_command = controls[1:, 0].copy()
    tail_bound = limits.tail_rad - tail_reserve_rad
    if not 0 < tail_bound <= limits.tail_rad:
        raise ValueError("tail reserve must be nonnegative and leave usable travel")
    static_lower = np.array([0.0, 0.0, -tail_bound])
    static_upper = np.array([limits.front_omega, limits.rear_omega, tail_bound])
    motor_rate = limits.motor_slew_rpm_s / RPM_PER_RAD_S
    permitted_change = dt * np.array([motor_rate, motor_rate, np.deg2rad(limits.tail_rate_deg_s)])
    command_scale = np.array([200.0, 200.0, 0.04])
    output_weights = np.array([1.0, 2.0, 10.0])
    regularizer = 0.01 * np.eye(3)
    model_discrepancy = 0.0
    for index in range(n):
        actual = np.array([state.vel[0], -state.vel[2], -state.pos[2] - 100.0, state.att[1], state.omega[1]])
        if wind_at is not None:
            wind[0] = wind_at(sample_times[index])
        air_state = actual.copy()
        air_state[0] -= wind[0]
        command = nominal_controls[:, index].copy()
        if tilt_response_s > 0:
            rate = (command[0] - state.tilt) * -np.expm1(-dt / tilt_response_s) / dt
            command[0] = state.tilt + dt * np.clip(rate, -aircraft.max_tilt_rate, aircraft.max_tilt_rate)
        if index:
            error = nominal_state[:, index] - actual
            target = nominal_rate[[0, 1, 4], index].copy()
            target[2] *= aircraft.inertia_yy
            target += [1.0 * error[0], 2.5 * error[1] + error[2],
                       aircraft.inertia_yy * (8.0 * error[4] + 14.0 * error[3])]
            lower = np.maximum(static_lower, previous_command - permitted_change)
            upper = np.minimum(static_upper, previous_command + permitted_change)
            command[1:] = np.clip(command[1:], lower, upper)
            for _ in range(2):
                local_load = np.asarray(load(air_state, command)).ravel()[:3]
                derivative = np.asarray(gradient(air_state, command)) * command_scale
                adjustment = lsq_linear(
                    np.vstack((output_weights[:, None] * derivative, regularizer)),
                    np.r_[output_weights * (target - local_load), np.zeros(3)],
                    bounds=((lower - command[1:]) / command_scale, (upper - command[1:]) / command_scale),
                ).x
                command[1:] += adjustment * command_scale
        previous_command = command[1:]
        applied = command.copy()
        if availability_at is not None:
            applied[1:3] *= np.sqrt(availability_at(sample_times[index]))
        tilt, front, rear, tail = applied
        next_state = dynamics.step(state, wind, front, front, rear, rear, tilt, tail, tail, dt)
        diag = dynamics.last_diagnostics
        forces = np.asarray(plant_load(air_state, applied)).ravel()
        measured = np.array([diag["acceleration_ned"][0], -diag["acceleration_ned"][2], diag["moment_body"][1]])
        model_discrepancy = max(model_discrepancy, float(np.max(np.abs(measured - forces[:3]))))
        forces[:3] = measured
        actual_motion[:, index] = actual
        actual_control[:, index] = command
        actual_load[:-1, index] = forces
        actual_load[-1, index] = (next_state.omega[1] - state.omega[1]) / dt
        if stop_on_limit and (
            abs(actual[4]) > np.deg2rad(limits.pitch_rate_deg_s)
            or abs(measured[0]) > limits.longitudinal_accel_mps2
            or abs(measured[1]) > limits.vertical_accel_mps2
            or abs(actual[2]) > limits.altitude_loss_m
        ):
            return (
                sample_times[:index + 1], actual_motion[:, :index + 1],
                actual_control[:, :index + 1], actual_load[:, :index + 1], model_discrepancy,
            )
        state = next_state
    return sample_times, actual_motion, actual_control, actual_load, model_discrepancy


def pitch_reserve(aircraft, limits, controls, loads):
    cosine = np.maximum(np.cos(controls[0]), 1e-10)
    front_max = 2 * aircraft.k_thrust * limits.front_omega**2
    rear_max = 2 * aircraft.k_thrust * limits.rear_omega**2
    front, rear, arm = loads[5:8]
    nose_up = np.minimum(front_max - front, rear / cosine) * arm
    nose_down = np.minimum(front, (rear_max - rear) / cosine) * arm
    return np.minimum(nose_up, nose_down)


def path_rows(aircraft, limits, time, motion, controls, loads, *, wind_at=None):
    time_delta = np.diff(time, prepend=time[0] - (time[1] - time[0]))
    rates = np.diff(controls, axis=1, prepend=controls[:, :1]) / time_delta
    wind = np.zeros_like(time) if wind_at is None else np.array([wind_at(t) for t in time])
    speed = np.hypot(motion[0] - wind, motion[1])
    reserve = pitch_reserve(aircraft, limits, controls, loads)
    return [{
        "time_s": time[i],
        "airspeed_mps": speed[i],
        "forward_speed_mps": motion[0, i],
        "wind_along_track_mps": wind[i],
        "climb_rate_mps": motion[1, i],
        "altitude_change_m": motion[2, i],
        "pitch_deg": np.rad2deg(motion[3, i]),
        "pitch_rate_deg_s": np.rad2deg(motion[4, i]),
        "front_tilt_deg": tilt_deg_from_deflection_rad(controls[0, i]),
        "tilt_rate_deg_s": -np.rad2deg(rates[0, i]),
        "front_rpm_per_motor": controls[1, i] * RPM_PER_RAD_S,
        "rear_rpm_per_motor": controls[2, i] * RPM_PER_RAD_S,
        "front_slew_rpm_s": rates[1, i] * RPM_PER_RAD_S,
        "rear_slew_rpm_s": rates[2, i] * RPM_PER_RAD_S,
        "vtail_deg": np.rad2deg(controls[3, i]),
        "vtail_rate_deg_s": np.rad2deg(rates[3, i]),
        "forward_acceleration_mps2": loads[0, i],
        "upward_acceleration_mps2": loads[1, i],
        "pitch_moment_nm": loads[2, i],
        "freestream_alpha_deg": np.rad2deg(loads[3, i]) if speed[i] >= 0.5 else np.nan,
        "effective_alpha_deg": np.rad2deg(loads[4, i]) if speed[i] >= 0.5 else np.nan,
        "net_front_pair_thrust_n": loads[5, i],
        "net_rear_pair_thrust_n": loads[6, i],
        "front_vertical_thrust_n": loads[5, i] * np.cos(controls[0, i] - motion[3, i]),
        "rear_vertical_thrust_n": loads[6, i] * np.cos(motion[3, i]),
        "wing_vertical_lift_n": loads[9, i],
        "wing_vertical_drag_n": loads[10, i],
        "vtail_vertical_force_n": loads[11, i],
        "wing_pitch_moment_nm": loads[12, i],
        "front_pitch_moment_nm": loads[13, i],
        "rear_pitch_moment_nm": loads[14, i],
        "vtail_pitch_moment_nm": loads[15, i],
        "pitch_acceleration_rad_s2": loads[16, i] if len(loads) > 16 else float("nan"),
        "differential_pitch_reserve_nm": reserve[i],
    } for i in range(len(time))]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dynamic_summary(aircraft, limits, rows):
    def peak(key):
        return max(abs(row[key]) for row in rows)

    last = rows[-1]
    wingborne_rows = [row for row in rows if row["airspeed_mps"] >= limits.wingborne_speed_mps]
    effective_stall_margin = aircraft.wing_stall_angle_deg - max(
        abs(row["effective_alpha_deg"]) for row in wingborne_rows
    )
    summary = {
        "altitude_loss_m": max(0.0, -min(row["altitude_change_m"] for row in rows)),
        "altitude_gain_m": max(row["altitude_change_m"] for row in rows),
        "max_climb_or_descent_mps": peak("climb_rate_mps"),
        "max_forward_acceleration_mps2": peak("forward_acceleration_mps2"),
        "max_vertical_acceleration_mps2": peak("upward_acceleration_mps2"),
        "max_pitch_deg": peak("pitch_deg"),
        "max_pitch_rate_deg_s": peak("pitch_rate_deg_s"),
        "max_tilt_rate_deg_s": peak("tilt_rate_deg_s"),
        "max_motor_slew_rpm_s": max(peak("front_slew_rpm_s"), peak("rear_slew_rpm_s")),
        "max_vtail_rate_deg_s": peak("vtail_rate_deg_s"),
        "max_vtail_deg": peak("vtail_deg"),
        "front_rpm_headroom": limits.front_omega * RPM_PER_RAD_S - peak("front_rpm_per_motor"),
        "rear_rpm_headroom": limits.rear_omega * RPM_PER_RAD_S - peak("rear_rpm_per_motor"),
        "minimum_pitch_reserve_nm": min(row["differential_pitch_reserve_nm"] for row in rows),
        "effective_stall_margin_wingborne_deg": effective_stall_margin,
        "effective_alpha_max_below_wingborne_deg": max(
            abs(row["effective_alpha_deg"]) for row in rows if 0.5 <= row["airspeed_mps"] < limits.wingborne_speed_mps
        ),
        "terminal_speed_mps": last["airspeed_mps"],
        "terminal_tilt_deg": last["front_tilt_deg"],
        "terminal_altitude_change_m": last["altitude_change_m"],
        "terminal_climb_rate_mps": last["climb_rate_mps"],
        "terminal_pitch_rate_deg_s": last["pitch_rate_deg_s"],
        "terminal_forward_acceleration_mps2": last["forward_acceleration_mps2"],
    }
    tolerance = 1e-5
    checks = {
        "altitude": max(summary["altitude_loss_m"], summary["altitude_gain_m"]) <= limits.altitude_loss_m + tolerance,
        "climb_rate": summary["max_climb_or_descent_mps"] <= limits.climb_rate_mps + tolerance,
        "forward_acceleration": summary["max_forward_acceleration_mps2"] <= limits.longitudinal_accel_mps2 + tolerance,
        "vertical_acceleration": summary["max_vertical_acceleration_mps2"] <= limits.vertical_accel_mps2 + tolerance,
        "pitch": summary["max_pitch_deg"] <= limits.pitch_limit_deg + tolerance,
        "pitch_rate": summary["max_pitch_rate_deg_s"] <= limits.pitch_rate_deg_s + tolerance,
        "tilt_rate": summary["max_tilt_rate_deg_s"] <= np.rad2deg(aircraft.max_tilt_rate) + tolerance,
        "motor_slew": summary["max_motor_slew_rpm_s"] <= limits.motor_slew_rpm_s + tolerance,
        "vtail_rate": summary["max_vtail_rate_deg_s"] <= limits.tail_rate_deg_s + tolerance,
        "vtail_travel": summary["max_vtail_deg"] <= np.rad2deg(limits.tail_rad) + tolerance,
        "motor_headroom": min(summary["front_rpm_headroom"], summary["rear_rpm_headroom"]) >= -tolerance,
        "pitch_reserve": summary["minimum_pitch_reserve_nm"] >= limits.pitch_reserve_nm - tolerance,
        "stall_margin": effective_stall_margin >= limits.stall_reserve_deg - tolerance,
        "terminal_tracking": (
            abs(last["airspeed_mps"] - TARGET_SPEED) <= 0.15
            and abs(last["front_tilt_deg"] - CRUISE_TILT_DEG) <= tolerance
            and abs(last["altitude_change_m"]) <= 0.15
            and abs(last["climb_rate_mps"]) <= 0.10
            and abs(last["pitch_rate_deg_s"]) <= 1.0
            and abs(last["forward_acceleration_mps2"]) <= 0.15
        ),
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    return summary


def check_static_path(dynamics, limits, rows, *, stride=1):
    """Re-solve trim at sampled path coordinates without rounding to grid cells."""
    indices = sorted({*range(0, len(rows), stride), len(rows) - 1})
    seed = ()
    for index in indices:
        row = rows[index]
        speed = row["airspeed_mps"]
        tilt = deflection_rad_from_tilt_deg(np.clip(row["front_tilt_deg"], 0, 90))
        motion_seed = (np.deg2rad(row["pitch_deg"]), row["front_rpm_per_motor"] / RPM_PER_RAD_S,
                       row["rear_rpm_per_motor"] / RPM_PER_RAD_S, np.deg2rad(row["vtail_deg"]))
        trim = trim_point(dynamics, limits, speed, tilt, (*seed, motion_seed))
        if trim["feasible"]:
            seed = (trim_guess(trim),)
        row["static_trim_found_at_sample"] = bool(trim["feasible"])
        row["static_trim_residual_at_sample"] = trim["max_normalized_residual"]
    for row in rows:
        row.setdefault("static_trim_found_at_sample", "")
        row.setdefault("static_trim_residual_at_sample", "")
    failed = [rows[index]["time_s"] for index in indices if not rows[index]["static_trim_found_at_sample"]]
    return {"points_checked": len(indices), "trim_found": len(indices) - len(failed), "failed_times_s": failed}


def plot_study(path, corridor, plan, replay):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    passed = [row for row in corridor if row["static_trim_found"]]
    failed = [row for row in corridor if not row["static_trim_found"]]
    axis = axes[0, 0]
    for rows, color, label in ((failed, "#d4dce2", "Trim not found"), (passed, "#459781", "Static trim")):
        axis.scatter([row["airspeed_mps"] for row in rows], [row["front_tilt_deg"] for row in rows],
                     s=12, color=color, label=label)
    axis.plot([row["airspeed_mps"] for row in plan], [row["front_tilt_deg"] for row in plan], color="#142b54", label="Optimized path")
    axis.set(xlabel="True airspeed (m/s)", ylabel="Front rotor angle (°)",
             title="Static corridor · 90° hover to 0° cruise")
    axis.legend(loc="upper right", fontsize=8)
    t = [row["time_s"] for row in replay]
    axes[0, 1].plot(t, [row["altitude_change_m"] for row in replay], color="#142b54")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Altitude change (m)", title="Rigid-body replay")
    for key, label in (("forward_acceleration_mps2", "Forward"), ("upward_acceleration_mps2", "Upward")):
        axes[1, 0].plot(t, [row[key] for row in replay], label=label)
    axes[1, 0].set(xlabel="Time (s)", ylabel="Acceleration (m/s²)")
    axes[1, 0].legend()
    for key, label in (("pitch_rate_deg_s", "Pitch rate"), ("tilt_rate_deg_s", "Tilt rate"), ("vtail_rate_deg_s", "V-tail rate")):
        axes[1, 1].plot(t, [row[key] for row in replay], label=label)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Angular rate (deg/s)")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.18)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    dynamics = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    baseline = build_aircraft_dynamics(aircraft)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    corridor = static_corridor(aircraft, dynamics, limits)
    for row in corridor:
        speed = row["airspeed_mps"]
        unchanged = speed == 0 or speed >= LOW_SPEED_FULL_LOAD
        baseline_trim = row["static_trim_found"] if unchanged else trim_point(
            baseline, limits, speed, deflection_rad_from_tilt_deg(row["front_tilt_deg"]),
        )["feasible"]
        row["original_cutoff_trim_found"] = bool(baseline_trim)
    write_csv(OUTPUT / "static_corridor.csv", corridor)
    print(f"Static trim grid saved ({len(corridor)} points).", flush=True)

    time, motion, controls = solve_study_trajectory(aircraft, dynamics, limits)
    load = longitudinal_loads(aircraft, LOW_SPEED_FULL_LOAD)
    plan = path_rows(aircraft, limits, time, motion, controls, np.asarray(load.map(time.size)(motion, controls)))
    check_static_path(dynamics, limits, plan)
    write_csv(OUTPUT / "optimized_trajectory.csv", plan)

    replay_time, actual, commands, actual_load, discrepancy = replay_trajectory(
        aircraft, dynamics, limits, time, motion, controls, step_s=0.01,
    )
    replay = path_rows(aircraft, limits, replay_time, actual, commands, actual_load)
    replay_results = dynamic_summary(aircraft, limits, replay)
    along_path = check_static_path(dynamics, limits, replay, stride=10)
    write_csv(OUTPUT / "rigid_body_replay.csv", replay)

    raw_time, raw_motion, raw_commands, raw_loads, _ = replay_trajectory(
        aircraft, baseline, limits, time, motion, controls, step_s=0.01, stop_on_limit=True,
    )
    raw = path_rows(aircraft, limits, raw_time, raw_motion, raw_commands, raw_loads)
    write_csv(OUTPUT / "original_cutoff_replay.csv", raw)
    endpoint_12 = next(row for row in corridor
                       if row["airspeed_mps"] == 12 and row["front_tilt_deg"] == CRUISE_TILT_DEG)
    cruise_speeds = [row["airspeed_mps"] for row in corridor
                     if row["front_tilt_deg"] == CRUISE_TILT_DEG and row["static_trim_found"]]
    summary = {
        "tilt_convention": TILT_CONVENTION,
        "static_points": len(corridor),
        "static_trim_found": sum(row["static_trim_found"] for row in corridor),
        "original_cutoff_static_trim_found": sum(row["original_cutoff_trim_found"] for row in corridor),
        "static_grid_resolution": {"airspeed_mps": 1, "tilt_deg": 5},
        "forward_rotor_static_grid_speeds_mps": cruise_speeds,
        "corner_12_mps_0_deg": endpoint_12,
        "duration_s": DURATION,
        "target_speed_mps": TARGET_SPEED,
        "low_speed_load_blend_mps": [0.5, LOW_SPEED_FULL_LOAD],
        "limits": asdict(limits),
        "front_rpm_limit": limits.front_omega * RPM_PER_RAD_S,
        "rear_rpm_limit": limits.rear_omega * RPM_PER_RAD_S,
        "tilt_rate_deg_s_limit": np.rad2deg(aircraft.max_tilt_rate),
        "replay": replay_results,
        "replay_step_s": 0.01,
        "model_max_load_difference": discrepancy,
        "static_trim_along_replay": along_path,
        "original_cutoff_replay_last_sample": raw[-1],
    }
    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=lambda value: value.item())
        stream.write("\n")
    plot_study(OUTPUT / "corridor_and_trajectory.png", corridor, plan, replay)
    print(f"Study saved in {OUTPUT}", flush=True)
    if not replay_results["passed"] or along_path["failed_times_s"] or discrepancy > 1e-6:
        raise RuntimeError("The candidate did not pass all rigid-body and corridor verification checks")


if __name__ == "__main__":
    main()
