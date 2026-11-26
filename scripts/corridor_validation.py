"""Evaluate static trim, replay reachability, and sampled disturbance margins."""

import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from scripts.corridor_uncertainty import INNER_MARGINS, SEED, fill_margins, run_case, scenarios, screen_route
from scripts.dynamic_corridor import SCHEDULES, load_study, route_plan
from scripts.refined_corridor import ANGLES, ANGLE_WIDTH, REPORT_DIR, SPEEDS, SPEED_WIDTH, compute_static, coverage
from scripts.transition_study import LOW_SPEED_FULL_LOAD, check_static_path, path_rows, replay_trajectory, study_limits, write_csv
from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.nacelle import TILT_CONVENTION, deflection_rad_from_tilt_deg


ROUTES = (0, 2, 3, 4, 5, 6, 7, 8, 9)
FINE_STEP = 0.01
COARSE_STEP = 0.02


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_reports(path, rows):
    columns = dict.fromkeys(key for row in rows for key in row)
    write_csv(path, [columns | row for row in rows])


def check_physics(aircraft, dynamics):
    """Check rotor axes, moment arms, and symmetric V-tail loads independently."""
    max_rotor_error = 0.0
    thrust = 8.0
    omega = np.sqrt(thrust / aircraft.k_thrust)
    locations = sorted(aircraft.motor_positions, key=lambda r: (-r[0], r[1]))
    pivot_x, pivot_z = aircraft.tilt_pivot_xz_m
    for angle in (90.0, 45.0, 0.0):
        for index, (x, y, z, tilts) in enumerate(locations):
            deflection = deflection_rad_from_tilt_deg(angle if tilts else 90.0)
            c, s = np.cos(deflection), np.sin(deflection)
            expected_force = thrust * np.array([s, 0, -c])
            expected_xyzm = np.array([-y * thrust * c, 0.0, -y * thrust * s])
            expected_xyzm[1] = thrust * (
                (x - pivot_x) + pivot_x * c + pivot_z * s if tilts else x
            )
            command = np.zeros(4)
            command[index] = omega
            force, moment, _ = dynamics._rotor_forces_and_moments(command, deflection_rad_from_tilt_deg(angle), 100)
            max_rotor_error = max(max_rotor_error, np.max(np.abs(force - expected_force)),
                                  np.max(np.abs(moment - expected_xyzm)))
    max_tail_error = 0.0
    projected = np.cos(np.deg2rad(aircraft.vtail_dihedral_deg))
    pitch = np.deg2rad(3.0)
    for speed in (0.0, 10.0, 20.0, 30.0):
        for tail in (-aircraft.vtail_max_deflection_rad, 0.0, aircraft.vtail_max_deflection_rad):
            lift_coefficient = np.clip(
                aircraft.vtail_lift_slope * (pitch + tail * projected),
                -aircraft.vtail_max_lift_coefficient, aircraft.vtail_max_lift_coefficient,
            )
            lift = 0.5 * aircraft.air_density * speed**2 * aircraft.vtail_area * projected * lift_coefficient
            force, moment = dynamics.aero.compute_tail_forces(
                np.array([speed, 0, 0]), np.zeros(3), np.array([0, pitch, 0]), tail, tail,
            )
            max_tail_error = max(max_tail_error, abs(force[2] + lift),
                                 abs(moment[1] + aircraft.vtail_arm_length * lift),
                                 abs(force[1]), abs(moment[2]))
    if max(max_rotor_error, max_tail_error) > 1e-8:
        raise RuntimeError("Independent rotor or V-tail balance check failed")
    return {"rotor_force_or_moment_max_error": max_rotor_error, "vtail_force_or_moment_max_error": max_tail_error}


def replay_nominal(aircraft, dynamics, limits, plan, step):
    time, motion, controls, loads, parity = replay_trajectory(
        aircraft, dynamics, limits, *plan, step_s=step, stop_on_limit=True,
    )
    rows = path_rows(aircraft, limits, time, motion, controls, loads)
    fill_margins(aircraft, limits, rows)
    summary, failures = screen_route(aircraft, limits, rows)
    force_error = max(abs(
        row["front_vertical_thrust_n"] + row["rear_vertical_thrust_n"]
        + row["wing_vertical_lift_n"] + row["wing_vertical_drag_n"] + row["vtail_vertical_force_n"]
        - aircraft.mass * (aircraft.gravity + row["upward_acceleration_mps2"])
    ) for row in rows)
    moment_error = max(abs(
        row["front_pitch_moment_nm"] + row["rear_pitch_moment_nm"] + row["wing_pitch_moment_nm"]
        + row["vtail_pitch_moment_nm"] - row["pitch_moment_nm"]
    ) for row in rows)
    if max(parity, force_error, moment_error) > 1e-6:
        failures.append("force_or_moment_closure")
    return rows, summary | {"vertical_force_closure_n": force_error, "pitch_moment_closure_nm": moment_error}, failures


def nominal_trial(task):
    route_id, time, guess = task
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    dynamics = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    report = {"route_id": route_id, "schedule_exponent": SCHEDULES[route_id]}
    try:
        plan = route_plan(aircraft, limits, SCHEDULES[route_id], time, guess)
    except RuntimeError as error:
        return report | {"status": "solver_failed", "failed_checks": str(error).splitlines()[-1]}, None, [], {}, set()
    fine, summary, failed = replay_nominal(aircraft, dynamics, limits, plan, FINE_STEP)
    coarse, coarse_summary, coarse_failed = replay_nominal(aircraft, dynamics, limits, plan, COARSE_STEP)
    fine_cells, edges = coverage(fine)
    coarse_cells, _ = coverage(coarse)
    report |= summary | {
        "coarse_timestep_passed": not coarse_failed,
        "coarse_failed_checks": " ".join(coarse_failed),
        "coarse_fine_cell_overlap_fraction": len(set(fine_cells) & set(coarse_cells)) / len(set(fine_cells) | set(coarse_cells)),
        "coarse_fine_peak_altitude_change_difference_m": abs(
            summary["altitude_deviation_m"] - coarse_summary["altitude_deviation_m"]
        ),
    }
    if failed:
        return report | {"status": "replay_failed", "failed_checks": " ".join(failed)}, None, [], {}, set()
    # A green cell must have an equilibrium check at its exact witness. Also
    # independently recheck trim every 0.1 s along the accepted route.
    selected = sorted(set(range(0, len(fine), 10)) | set(fine_cells.values()) | {len(fine) - 1})
    samples = [fine[i] for i in selected]
    trim = check_static_path(dynamics, limits, samples)
    if trim["failed_times_s"]:
        return report | {
            "status": "trim_recheck_failed", "failed_checks": str(trim["failed_times_s"]),
        }, None, [], {}, set()
    for row in fine:
        row["route_id"] = route_id
    _, inner_failures = screen_route(aircraft, limits, fine, inner=True)
    report |= {"status": "constraint_checked", "trim_samples": trim["points_checked"],
               "inner_nominal_passed": not inner_failures, "inner_nominal_failures": " ".join(inner_failures)}
    return report, plan, samples, {key: fine[i] for key, i in fine_cells.items()}, edges


def plot_maps(static, witnessed, robust, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    base = np.array([[int(static[speed, angle]) for speed in SPEEDS] for angle in ANGLES])
    dynamic = np.array([[2 if (speed, angle) in witnessed else base[j, i]
                         for i, speed in enumerate(SPEEDS)] for j, angle in enumerate(ANGLES)])
    inner = np.array([[3 if (speed, angle) in robust else 2 if (speed, angle) in witnessed else 0
                       for speed in SPEEDS] for angle in ANGLES])
    colors = ["#f0f2f4", "#8baecb", "#24876e", "#4d438c"]
    captions = ["Trim not found", "Static trim at cell center", "Constraint-checked replay witness",
                "Reached in every tested case on a passing route"]
    stages = (
        ("static_trim.png", "Static trim", base, (0, 1)),
        ("dynamic_reachability.png", "Constraint-checked dynamic reachability", dynamic, (0, 1, 2)),
        ("tested_margin_corridor.png", "Candidate disturbance-tested transition corridor", inner, (2, 3)),
    )
    for name, title, matrix, indices in stages:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.pcolormesh(
            np.r_[SPEEDS - SPEED_WIDTH / 2, SPEEDS[-1] + SPEED_WIDTH / 2],
            np.r_[ANGLES - ANGLE_WIDTH / 2, ANGLES[-1] + ANGLE_WIDTH / 2],
            matrix, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
            shading="flat",
        )
        handles = [Patch(color=colors[i], label=captions[i]) for i in indices]
        if name == "tested_margin_corridor.png" and selected:
            line, = ax.plot([row["airspeed_mps"] for row in selected], [row["front_tilt_deg"] for row in selected],
                            color="#efb44f", lw=1.8, label="Selected nominal replay")
            handles.append(line)
        ax.legend(handles=handles, loc="upper right", framealpha=0.96, fontsize=8)
        if name == "tested_margin_corridor.png" and not robust:
            ax.text(0.03, 0.06, "No route passed every tested case", transform=ax.transAxes, weight="bold")
        ax.set(xlabel="True airspeed (m/s)", ylabel="Front rotor angle (°)", title=title,
               xlim=(-SPEED_WIDTH / 2, SPEEDS[-1] + SPEED_WIDTH / 2), ylim=(-1, 91))
        fig.savefig(REPORT_DIR / name, dpi=165)
        plt.close(fig)


def plot_loads(rows):
    import matplotlib.pyplot as plt

    time = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for key, label in (
        ("front_vertical_thrust_n", "Front rotors"), ("rear_vertical_thrust_n", "Rear rotors"),
        ("wing_vertical_lift_n", "Wing lift"), ("vtail_vertical_force_n", "V-tail net force"),
    ):
        axes[0].plot(time, [row[key] for row in rows], label=label)
    for key, label in (
        ("front_pitch_moment_nm", "Front rotors"), ("rear_pitch_moment_nm", "Rear rotors"),
        ("wing_pitch_moment_nm", "Wing"), ("vtail_pitch_moment_nm", "V-tail"),
    ):
        axes[1].plot(time, [row[key] for row in rows], label=label)
    axes[2].plot(time, [row["front_rpm_per_motor"] for row in rows], label="Front motors")
    axes[2].plot(time, [row["rear_rpm_per_motor"] for row in rows], label="Rear motors")
    axes[0].set(ylabel="Vertical force (N)", title="Selected replay force and moment balance")
    axes[1].set_ylabel("Pitch moment (N·m)")
    axes[2].set(xlabel="Time (s)", ylabel="RPM per motor")
    for ax in axes:
        ax.legend(ncols=4, fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(REPORT_DIR / "propulsion_and_pitch_history.png", dpi=165)
    plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    static_path = REPORT_DIR / "static_trim.csv"
    if not static_path.exists():
        compute_static()
    static_rows = read_rows(static_path)
    static = {(float(r["airspeed_mps"]), float(r["front_tilt_deg"])): r["trim_found"] == "True" for r in static_rows}
    coarse_static, time, guess = load_study()
    overlap = {key for key in coarse_static if key in static}
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    dynamics = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    physical = check_physics(aircraft, dynamics)
    reports, accepted, witnesses, edges = {}, {}, defaultdict(list), defaultdict(list)
    with ProcessPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(nominal_trial, (route_id, time, guess)) for route_id in ROUTES]
        for future in as_completed(futures):
            report, plan, samples, cells, links = future.result()
            route_id = report["route_id"]
            reports[route_id] = report
            if plan is not None:
                accepted[route_id] = (plan, samples, set(cells))
                for key, row in cells.items():
                    witnesses[key].append(row)
                for key in links:
                    edges[key].append(route_id)
            print(f"Nominal route {route_id} {report['status']} {report.get('failed_checks', '')}", flush=True)
        write_reports(REPORT_DIR / "nominal_route_checks.csv", [reports[i] for i in sorted(reports)])
        if not accepted:
            raise RuntimeError("No routes passed the fine-timestep replay and exact-point trim checks")
        eligible = [i for i in accepted if reports[i]["inner_nominal_passed"]]
        eligible.sort(key=lambda i: reports[i]["energy_wh"])
        diversity = sorted(eligible, key=lambda i: SCHEDULES[i] or 0.55)
        candidates = list(dict.fromkeys([*eligible[:1], *diversity[:1], *diversity[-1:]]))
        candidates.extend(i for i in eligible if i not in candidates)
        candidates = candidates[:3]
        experiment = scenarios()
        tests = [workers.submit(run_case, (i, accepted[i][0], scenario)) for i in candidates for scenario in experiment]
        tested = defaultdict(list)
        failed_criteria = Counter()
        for index, future in enumerate(as_completed(tests), start=1):
            result, cells = future.result()
            tested[result["route_id"]].append((result, cells))
            failed_criteria.update(result["failed_checks"].split())
            if index % 6 == 0 or index == len(tests):
                print(f"Completed {index} / {len(tests)} disturbance and uncertainty replays.", flush=True)

    case_rows = [report for route_id in sorted(tested)
                 for report, _ in sorted(tested[route_id], key=lambda item: item[0]["name"])]
    if case_rows:
        write_reports(REPORT_DIR / "scenario_replays.csv", case_rows)
    passing_counts = {i: sum(report["passed"] for report, _ in cases) for i, cases in sorted(tested.items())}
    robust_routes = [i for i, count in passing_counts.items() if count == len(experiment)]
    robust = defaultdict(list)
    for route_id in robust_routes:
        common = accepted[route_id][2].intersection(*(cells for _, cells in tested[route_id]))
        for key in common:
            robust[key].append(route_id)

    selected_id = min(robust_routes, key=lambda i: reports[i]["energy_wh"]) if robust_routes else min(
        accepted, key=lambda i: (-passing_counts.get(i, -1), reports[i]["energy_wh"]),
    )
    plan, selected_samples, _ = accepted[selected_id]
    full_rows, _, _ = replay_nominal(aircraft, dynamics, limits, plan, FINE_STEP)
    write_csv(REPORT_DIR / "selected_replay.csv", full_rows)
    write_csv(REPORT_DIR / "observed_edges.csv", [{
        "from_speed_mps": source[0], "from_angle_deg": source[1],
        "to_speed_mps": target[0], "to_angle_deg": target[1],
        "route_ids": " ".join(map(str, sorted(ids))),
    } for (source, target), ids in sorted(edges.items())])
    rows = []
    for key, trimmed in sorted(static.items()):
        evidence = sorted(witnesses.get(key, []), key=lambda row: row["route_id"])
        row = evidence[0] if evidence else {}
        rows.append({
            "airspeed_cell_center_mps": key[0], "tilt_cell_center_deg": key[1],
            "static_center_trim_found": trimmed,
            "nominal_evidence": "constraint_checked" if row else "no_witness",
            "tested_inner_evidence": "all_cases_observed" if key in robust else "unverified",
            "route_ids": " ".join(str(item["route_id"]) for item in evidence),
            "all_case_route_ids": " ".join(map(str, sorted(robust.get(key, [])))),
            "witness_actual_airspeed_mps": row.get("airspeed_mps", ""),
            "witness_actual_tilt_deg": row.get("front_tilt_deg", ""),
            "witness_time_s": row.get("time_s", ""),
            "witness_trim_residual": row.get("static_trim_residual_at_sample", ""),
            "witness_altitude_deviation_m": row.get("altitude_change_m", ""),
            "witness_pitch_rate_deg_s": row.get("pitch_rate_deg_s", ""),
            "witness_rotor_pitch_reserve_nm": row.get("differential_pitch_reserve_nm", ""),
            "witness_motor_rpm_reserve_fraction": row.get("motor_rpm_reserve_fraction", ""),
            "witness_vtail_travel_reserve_deg": row.get("tail_travel_reserve_deg", ""),
            "witness_electrical_voltage_headroom_v": row.get("voltage_headroom_v", ""),
        })
    write_csv(REPORT_DIR / "corridor_evidence.csv", rows)
    plot_maps(static, witnesses, robust, selected_samples if selected_id in robust_routes else [])
    plot_loads(full_rows)
    summary = {
        "tilt_convention": TILT_CONVENTION,
        "scope": "Symmetric longitudinal dynamics in simulation with along-track gusts and nominal-model feedback.",
        "static_centers": len(static),
        "static_trim_found": sum(static.values()),
        "shared_coarse_centers": len(overlap),
        "static_classification_changes_at_shared_centers": sum(coarse_static[k] != static[k] for k in overlap),
        "air_speed_step_mps": SPEED_WIDTH, "front_angle_step_deg": ANGLE_WIDTH,
        "fine_replay_dt_s": FINE_STEP, "coarse_replay_dt_s": COARSE_STEP,
        "accepted_nominal_routes": len(accepted), "nominal_routes_attempted": len(reports),
        "nominal_witness_cells": len(witnesses), "nominal_directed_edges": len(edges),
        "nominal_static_center_cells_without_witness": sum(value and key not in witnesses for key, value in static.items()),
        "physics_checks": physical, "inner_screen_limits": INNER_MARGINS,
        "monte_carlo_seed": SEED, "scenario_replays": len(case_rows),
        "cases_per_tested_route": len(experiment) if candidates else 0,
        "scenario_inputs": [scenario.__dict__ for scenario in experiment],
        "cases_passed_by_route": passing_counts, "failed_checks_counts": failed_criteria,
        "routes_passing_every_tested_case": robust_routes, "all_case_witness_cells": len(robust),
        "selected_route": selected_id, "selected_route_passed_all_cases": selected_id in robust_routes,
        "selected_nominal_energy_wh": reports[selected_id]["energy_wh"],
        "remaining_unassessed": ["measured uncertainty ranges", "lateral and crosswind robustness",
                                  "hardware and aerodynamic validation"],
    }
    with (REPORT_DIR / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=lambda value: value.item())
        stream.write("\n")
    print(f"{len(witnesses)} nominal witnessed cells and {len(robust)} common all-case cells.", flush=True)


if __name__ == "__main__":
    main()
