"""Attempt routes through selected static-only speed and rotor-angle centers."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import json

import numpy as np

from scripts.corridor_validation import FINE_STEP, plot_maps, read_rows, replay_nominal, write_reports
from scripts.dynamic_corridor import TERMINAL_SETTLE_S, load_study
from scripts.refined_corridor import REPORT_DIR, coverage
from scripts.transition_study import DURATION, LOW_SPEED_FULL_LOAD, TARGET_SPEED, check_static_path, study_limits
from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.nacelle import deflection_rad_from_tilt_deg
from src.dynamics.transition_trajectory import optimize_trajectory


TARGETS = [(12.0, 80.0), (20.0, 6.0), (25.0, 70.0)]


def probe(task):
    route_id, speed, tilt, offset, guess = task
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    dynamics = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    end = DURATION - TERMINAL_SETTLE_S
    node_count = guess.shape[1] - 1
    node = int(np.clip(np.argmin(abs(guess[0] - speed)) + offset, 1, node_count - 1))
    result = {"route_id": route_id, "airspeed_target_mps": speed, "rotor_angle_target_deg": tilt,
              "arrival_time_s": end * node / node_count}
    try:
        time, motion, control = optimize_trajectory(
            aircraft, limits, TARGET_SPEED, guess, duration_s=end,
            low_speed_full_load_mps=LOW_SPEED_FULL_LOAD, flow_guard_extra_deg=1.0,
            waypoint=(node, speed, deflection_rad_from_tilt_deg(tilt)),
        )
    except RuntimeError as error:
        return result | {"status": "solver_did_not_find_route", "diagnostic": str(error).splitlines()[-1]}, [], set()
    plan = np.r_[time, DURATION], np.column_stack((motion, motion[:, -1])), np.column_stack((control, control[:, -1]))
    rows, _, failures = replay_nominal(aircraft, dynamics, limits, plan, FINE_STEP)
    cells, edges = coverage(rows)
    point = (speed, tilt)
    if failures or point not in cells:
        return result | {"status": "unverified_after_replay", "diagnostic": " ".join(failures) or "target_bin_missed"}, [], set()
    checked = [rows[index] for index in sorted(set(range(0, len(rows), 10)) | set(cells.values()) | {len(rows) - 1})]
    trim = check_static_path(dynamics, limits, checked)
    if trim["failed_times_s"]:
        return result | {"status": "unverified_after_replay", "diagnostic": "trim_recheck_failed"}, [], set()
    witness = rows[cells[point]]
    return result | {"status": "verified", "diagnostic": "",
                     "witness_speed_mps": witness["airspeed_mps"], "witness_angle_deg": witness["front_tilt_deg"]}, [
        {
            "route_id": route_id, "airspeed_cell_center_mps": key[0], "tilt_cell_center_deg": key[1],
            "witness_actual_airspeed_mps": rows[index]["airspeed_mps"],
            "witness_actual_tilt_deg": rows[index]["front_tilt_deg"],
            "witness_time_s": rows[index]["time_s"],
            "witness_trim_residual": rows[index]["static_trim_residual_at_sample"],
            "witness_altitude_deviation_m": rows[index]["altitude_change_m"],
            "witness_pitch_rate_deg_s": rows[index]["pitch_rate_deg_s"],
            "witness_rotor_pitch_reserve_nm": rows[index]["differential_pitch_reserve_nm"],
            "witness_motor_rpm_reserve_fraction": rows[index]["motor_rpm_reserve_fraction"],
            "witness_vtail_travel_reserve_deg": rows[index]["tail_travel_reserve_deg"],
            "witness_electrical_voltage_headroom_v": rows[index]["voltage_headroom_v"],
        }
        for key, index in cells.items()
    ], edges


def merge():
    """Include independently accepted targeted probes in the nominal evidence."""
    summary_path = REPORT_DIR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grid = read_rows(REPORT_DIR / "corridor_evidence.csv")
    probes = read_rows(REPORT_DIR / "static_only_probes.csv")
    by_cell = {(row["airspeed_cell_center_mps"], row["tilt_cell_center_deg"]): row for row in grid}
    witness_path = REPORT_DIR / "probe_witnesses.csv"
    if witness_path.exists():
        for witness in read_rows(witness_path):
            key = witness["airspeed_cell_center_mps"], witness["tilt_cell_center_deg"]
            row = by_cell[key]
            if row["nominal_evidence"] == "no_witness":
                row["nominal_evidence"] = "constraint_checked"
                row.update({field: value for field, value in witness.items() if field.startswith("witness_")})
            row["route_ids"] = " ".join(sorted(set(row["route_ids"].split()) | {witness["route_id"]}, key=int))
    write_reports(REPORT_DIR / "corridor_evidence.csv", grid)
    edge_path = REPORT_DIR / "probe_edges.csv"
    if edge_path.exists():
        edge_fields = ("from_speed_mps", "from_angle_deg", "to_speed_mps", "to_angle_deg")
        lookup = {tuple(row[key] for key in edge_fields): row for row in read_rows(REPORT_DIR / "observed_edges.csv")}
        for edge in read_rows(edge_path):
            key = tuple(edge[field] for field in edge_fields)
            row = lookup.setdefault(key, dict(edge))
            row["route_ids"] = " ".join(sorted(set(row["route_ids"].split()) | set(edge["route_ids"].split()), key=int))
        write_reports(REPORT_DIR / "observed_edges.csv", [lookup[key] for key in sorted(lookup)])
        summary["nominal_directed_edges"] = len(lookup)
    static = {(float(key[0]), float(key[1])): row["static_center_trim_found"] == "True" for key, row in by_cell.items()}
    witnessed = {(float(key[0]), float(key[1])) for key, row in by_cell.items() if row["nominal_evidence"] == "constraint_checked"}
    robust = {(float(key[0]), float(key[1])) for key, row in by_cell.items() if row["tested_inner_evidence"] == "all_cases_observed"}
    selected = [{key: float(row[key]) for key in ("airspeed_mps", "front_tilt_deg")}
                for row in read_rows(REPORT_DIR / "selected_replay.csv")] if summary["selected_route_passed_all_cases"] else []
    plot_maps(static, witnessed, robust, selected)
    summary["targeted_probe_attempts"] = len(probes)
    summary["targeted_probe_routes_verified"] = sum(row["status"] == "verified" for row in probes)
    summary["nominal_witness_cells"] = len(witnessed)
    summary["nominal_static_center_cells_without_witness"] = sum(value and key not in witnessed for key, value in static.items())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Nominal evidence now covers {len(witnessed)} cells with the targeted routes included.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge", action="store_true", help="merge existing accepted probes into the three corridor maps")
    if parser.parse_args().merge:
        merge()
        return
    static = {(float(row["airspeed_mps"]), float(row["front_tilt_deg"])) for row in
              read_rows(REPORT_DIR / "static_trim.csv") if row["trim_found"] == "True"}
    _, _, guess = load_study()
    positions = [(speed, tilt, offset) for speed, tilt in TARGETS if (speed, tilt) in static for offset in (-7, 7)]
    subjects = [(100 + index, speed, tilt, offset, guess) for index, (speed, tilt, offset) in enumerate(positions)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(probe, subjects))
    write_reports(REPORT_DIR / "static_only_probes.csv", [row for row, _, _ in results])
    witnesses = [w for _, points, _ in results for w in points]
    edges = [{
        "from_speed_mps": source[0], "from_angle_deg": source[1],
        "to_speed_mps": target[0], "to_angle_deg": target[1], "route_ids": row["route_id"],
    } for row, _, links in results for (source, target) in sorted(links)]
    if witnesses:
        write_reports(REPORT_DIR / "probe_witnesses.csv", witnesses)
        write_reports(REPORT_DIR / "probe_edges.csv", edges)
    for row, _, _ in results:
        print(row, flush=True)


if __name__ == "__main__":
    main()
