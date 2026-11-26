"""Audit fine trim, replay full disturbance cases, and try a margin-first route."""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from scripts.corridor_uncertainty import run_case, scenarios
from scripts.corridor_validation import read_rows, replay_nominal, write_reports
from scripts.dynamic_corridor import load_study, route_plan
from scripts.refined_corridor import REPORT_DIR
from scripts.transition_study import DURATION, LOW_SPEED_FULL_LOAD, TARGET_SPEED, study_limits
from src.aero.wing_polar import low_speed_aerodynamic_factor, wing_lift_drag_coefficients
from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.nacelle import deflection_rad_from_tilt_deg
from src.dynamics.state import AircraftState
from src.dynamics.transition import assess_rotor_electrical_point
from src.dynamics.transition_trajectory import RPM_PER_RAD_S, longitudinal_loads, optimize_trajectory

OUTPUT = REPORT_DIR / "followup"


def audit_trim():
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    dynamics = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    load = longitudinal_loads(aircraft, LOW_SPEED_FULL_LOAD, exact_kinematics=True)
    output = []
    for raw in read_rows(REPORT_DIR / "static_trim.csv"):
        speed, angle, pitch, tail, front_rpm, rear_rpm = (float(raw[key]) for key in (
            "airspeed_mps", "front_tilt_deg", "pitch_deg", "vtail_deg", "front_rpm_per_motor", "rear_rpm_per_motor",
        ))
        tilt, pitch_rad, delta = deflection_rad_from_tilt_deg(angle), np.deg2rad(pitch), np.deg2rad(tail)
        front, rear = np.array([front_rpm, rear_rpm]) / RPM_PER_RAD_S
        state = AircraftState(pos=np.array([0., 0., -100.]), vel=np.array([speed, 0., 0.]), att=np.array([0., pitch_rad, 0.]), tilt=tilt)
        dynamics.step(state, np.zeros(3), front, front, rear, rear, tilt, delta, delta, 0.01)
        diagnostic = dynamics.last_diagnostics
        ax, az, moment = diagnostic["acceleration_ned"][0], -diagnostic["acceleration_ned"][2], diagnostic["moment_body"][1]
        loads = np.asarray(load([speed, 0, 0, pitch_rad, 0], [tilt, front, rear, delta])).ravel()
        parity = max(abs(loads[:3] - [ax, az, moment]))
        electrical = assess_rotor_electrical_point([front_rpm] * 2 + [rear_rpm] * 2, aircraft.electrical_parameters)
        residual = max(abs(ax) / aircraft.gravity, abs(az) / aircraft.gravity,
                       abs(moment) / (aircraft.mass * aircraft.gravity * aircraft.wing_mean_chord_m))
        controls_ok = (
            0 <= angle <= 90 and 0 <= front <= limits.front_omega + 1e-6
            and 0 <= rear <= limits.rear_omega + 1e-6 and abs(delta) <= limits.tail_rad + 1e-8
        )
        effective_alpha = np.rad2deg(loads[4])
        wingborne = speed >= limits.wingborne_speed_mps
        stall_ok = max(abs(pitch), abs(effective_alpha)) <= aircraft.wing_stall_angle_deg
        result = (
            "trim_not_found" if raw["trim_found"] != "True" or residual > 0.005000001 else
            "control_or_power_limit" if not controls_ok or not electrical["feasible"] else
            "wingborne_incidence_limit" if wingborne and not stall_ok else
            "passes_wingborne_screen" if wingborne else "below_wingborne_gate"
        )
        output.append(raw | {
            "candidate_audit": result, "trim_replay_max_normalized_residual": residual,
            "load_model_max_difference": parity, "effective_alpha_deg": effective_alpha if speed >= .5 else "",
            "electrical_pass": bool(electrical["feasible"]), "battery_power_w": electrical["battery_power_w"],
            "motor_voltage_headroom_v": electrical["battery_voltage_v"] - electrical["maximum_required_motor_voltage_v"],
            "wingborne_incidence_checked": wingborne,
        })
    write_reports(OUTPUT / "static_trim_audit.csv", output)
    return output


def make_plan(task):
    route_id, time, guess = task
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    try:
        if route_id != 12:
            from scripts.dynamic_corridor import SCHEDULES
            return route_id, route_plan(aircraft, limits, SCHEDULES[route_id], time, guess), ""
        design = replace(limits, tail_rad=limits.tail_rad - np.deg2rad(2.0), stall_reserve_deg=5,
                         pitch_reserve_nm=1.2, pitch_rate_deg_s=14, altitude_loss_m=0.65,
                         front_omega=0.85 * limits.front_omega, rear_omega=0.85 * limits.rear_omega)
        plan = optimize_trajectory(
            replace(aircraft, max_tilt_rate=0.85 * aircraft.max_tilt_rate), design, TARGET_SPEED, guess,
            duration_s=30., low_speed_full_load_mps=LOW_SPEED_FULL_LOAD, tilt_schedule_exponent=.85,
            flow_guard_extra_deg=1.0, margin_objective=True,
        )
        t, x, u = plan
        return route_id, (np.r_[t, DURATION], np.column_stack((x, x[:, -1])),
                          np.column_stack((u, u[:, -1]))), ""
    except RuntimeError as error:
        return route_id, None, str(error).splitlines()[-1]


def balance_report(aircraft, rows, route_id):
    disk_area = np.pi * (aircraft.prop_diameter_m / 2.0)**2
    for row in rows:
        upward = sum(row[key] for key in (
            "front_vertical_thrust_n", "rear_vertical_thrust_n", "wing_vertical_lift_n",
            "wing_vertical_drag_n", "vtail_vertical_force_n",
        )) - aircraft.mass * aircraft.gravity
        moment = sum(row[key] for key in (
            "front_pitch_moment_nm", "rear_pitch_moment_nm", "wing_pitch_moment_nm", "vtail_pitch_moment_nm",
        ))
        speed = row["airspeed_mps"]
        tilt = deflection_rad_from_tilt_deg(row["front_tilt_deg"])
        alpha = np.deg2rad(row["freestream_alpha_deg"]) if speed >= 0.5 else 0.0
        induced_speed = np.sqrt(
            (row["net_front_pair_thrust_n"] + row["net_rear_pair_thrust_n"]) / (8 * aircraft.air_density * disk_area),
        )
        effective_speed = np.hypot(speed * np.cos(alpha) + induced_speed * np.sin(tilt),
                                   speed * np.sin(alpha) + induced_speed * np.cos(tilt))
        alpha_eff = np.deg2rad(row["effective_alpha_deg"]) if speed >= 0.5 else 0.0
        cl, _ = wing_lift_drag_coefficients(
            alpha_eff, cl_zero=aircraft.wing_lift_coefficient_zero, lift_slope_per_rad=aircraft.wing_lift_slope,
            stall_angle_rad=np.deg2rad(aircraft.wing_stall_angle_deg), cd_zero=aircraft.wing_drag_coeff,
            aspect_ratio=aircraft.wing_aspect_ratio, oswald_efficiency=aircraft.oswald_efficiency,
        )
        pressure = 0.5 * aircraft.air_density * effective_speed**2 * low_speed_aerodynamic_factor(speed, LOW_SPEED_FULL_LOAD)
        vertical_lift = pressure * aircraft.wing_area * float(cl) * np.cos(np.deg2rad(row["pitch_deg"]) - alpha)
        row.update(
            net_vertical_force_n=upward, integrated_vertical_force_n=aircraft.mass * row["upward_acceleration_mps2"],
            vertical_force_closure_n=upward - aircraft.mass * row["upward_acceleration_mps2"],
            net_pitch_moment_nm=moment, integrated_pitch_moment_nm=aircraft.inertia_yy * row["pitch_acceleration_rad_s2"],
            pitch_moment_closure_nm=moment - aircraft.inertia_yy * row["pitch_acceleration_rad_s2"],
            modeled_wing_effective_airspeed_mps=effective_speed, modeled_wing_lift_coefficient=float(cl),
            wing_vertical_lift_reconstruction_error_n=vertical_lift - row["wing_vertical_lift_n"],
        )
    write_reports(OUTPUT / f"route_{route_id}_balance.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = [r["time_s"] for r in rows]
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True, layout="constrained")
    for ax, first, second, label, a, b in (
        (axs[0], "net_vertical_force_n", "integrated_vertical_force_n", "Vertical force (N)", "Upward forces minus weight", "Mass × upward acceleration"),
        (axs[1], "net_pitch_moment_nm", "integrated_pitch_moment_nm", "Pitch moment (N·m)", "Sum of component moments", "Pitch inertia × dq/dt"),
    ):
        ax.plot(time, [r[first] for r in rows], label=a)
        ax.plot(time, [r[second] for r in rows], "--", label=b)
        ax.set_ylabel(label)
        ax.legend()
    for key, label in (("vertical_force_closure_n", "Vertical closure (N)"), ("pitch_moment_closure_nm", "Pitch closure (N·m)")):
        axs[2].plot(time, [r[key] for r in rows], label=label)
    axs[2].set(xlabel="Time (s)", title="Difference between load sum and integrated motion")
    axs[2].legend()
    axs[0].set_title(f"Route {route_id} force and pitch balance")
    fig.savefig(OUTPUT / f"route_{route_id}_balance.png", dpi=155)
    plt.close(fig)
    return {key: max(abs(r[key]) for r in rows) for key in (
        "vertical_force_closure_n", "pitch_moment_closure_nm", "wing_vertical_lift_reconstruction_error_n",
    )}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    static = audit_trim()
    print("Static audit complete.", flush=True)
    _, time, guess = load_study()
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    plant = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    diagnostics, cases = {}, []
    with ProcessPoolExecutor(max_workers=4) as pool:
        plans = {}
        for route_id, plan, solver_error in pool.map(make_plan, [(i, time, guess) for i in (0, 2, 3, 12)]):
            print(f"Plan {route_id} {'solved' if plan is not None else solver_error}", flush=True)
            diagnostics[route_id] = {"solver_error": solver_error}
            if plan is not None:
                plans[route_id] = plan
        trials = [pool.submit(run_case, (i, plan, case)) for i, plan in plans.items() for case in scenarios()]
        for index, trial in enumerate(as_completed(trials), 1):
            cases.append(trial.result()[0])
            if index % 11 == 0:
                print(f"Completed {index}/{len(trials)} full-route tests.", flush=True)
    for route_id in (3, 12):
        if route_id in plans:
            rows, metrics, failed = replay_nominal(aircraft, plant, limits, plans[route_id], .01)
            closure = balance_report(aircraft, rows, route_id)
            diagnostics[route_id] |= metrics | closure | {"nominal_failed_checks": failed}
    write_reports(OUTPUT / "disturbance_cases.csv", sorted(cases, key=lambda item: (item["route_id"], item["name"])))
    counts = {i: sum(r["passed"] for r in cases if r["route_id"] == i) for i in plans}
    summary = {
        "static_audit_counts": Counter(r["candidate_audit"] for r in static),
        "static_max_load_model_difference": max(r["load_model_max_difference"] for r in static),
        "static_gates": "Thrust/RPM and tail travel are bounded in the trim solve. Power and wingborne incidence are screened on the stored solution. Tilt rate is assessed in replay.",
        "cases_per_plan": len(scenarios()), "case_pass_counts": counts,
        "full_sequence_passes": [i for i, count in counts.items() if count == len(scenarios())],
        "first_violations": Counter(r.get("first_violated_constraint", "passed") for r in cases),
        "routes": diagnostics,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda value: value.item()), encoding="utf-8")


if __name__ == "__main__":
    main()
