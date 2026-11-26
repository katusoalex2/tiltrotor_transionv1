"""Map speed–tilt cells reached by verified, complete transition trajectories."""

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scripts.transition_study import (
    DURATION,
    LOW_SPEED_FULL_LOAD,
    OUTPUT,
    TARGET_SPEED,
    check_static_path,
    dynamic_summary,
    path_rows,
    replay_trajectory,
    study_limits,
    write_csv,
)
from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.nacelle import TILT_CONVENTION, deflection_rad_from_tilt_deg
from src.dynamics.transition_trajectory import RPM_PER_RAD_S, optimize_trajectory


SCHEDULES = [None, 0.38, 0.60, 0.85, 1.10, 1.45, 1.90, 2.50, 3.30, 4.50, 6.50, 9.00]
REPLAY_DT = 0.02
TERMINAL_SETTLE_S = 2.0
FLOW_CHECK_TOLERANCE_DEG = 0.05
SPEED_CELL = 1
TILT_CELL = 5
CORRIDOR_OUTPUT = OUTPUT / "dynamic_corridor"


def load_study():
    with (OUTPUT / "summary.json").open(encoding="utf-8") as stream:
        if json.load(stream).get("tilt_convention") != TILT_CONVENTION:
            raise ValueError("Regenerate the transition study for the current rotor-angle convention")
    with (OUTPUT / "optimized_trajectory.csv").open(newline="", encoding="utf-8") as stream:
        plan = list(csv.DictReader(stream))
    with (OUTPUT / "static_corridor.csv").open(newline="", encoding="utf-8") as stream:
        static = {
            (int(row["airspeed_mps"]), int(row["front_tilt_deg"])): row["static_trim_found"] == "True"
            for row in csv.DictReader(stream)
        }
    names = (
        "forward_speed_mps", "climb_rate_mps", "altitude_change_m", "pitch_deg", "pitch_rate_deg_s",
        "front_tilt_deg", "front_rpm_per_motor", "rear_rpm_per_motor", "vtail_deg",
    )
    guess = np.array([[float(row[name]) for row in plan] for name in names])
    guess[[3, 4, 8]] = np.deg2rad(guess[[3, 4, 8]])
    guess[5] = deflection_rad_from_tilt_deg(guess[5])
    guess[6:8] /= RPM_PER_RAD_S
    time = np.array([float(row["time_s"]) for row in plan])
    if not np.isclose(time[-1], DURATION) or not np.isclose(guess[0, -1], TARGET_SPEED):
        raise ValueError("Regenerate the static study and nominal trajectory before exploring the corridor")
    return static, time, guess


def minimum_flow_guard_margin(aircraft, limits, rows):
    """Return the least margin to either speed-dependent incidence bound."""
    return float(np.min(flow_guard_margins(aircraft, limits, rows)))


def flow_guard_margins(aircraft, limits, rows):
    stall_guard = aircraft.wing_stall_angle_deg - limits.stall_reserve_deg
    speed = np.array([row["airspeed_mps"] for row in rows])
    free_limit = stall_guard + (180 - stall_guard) * np.clip((4.0 - speed) / 3.5, 0, 1)
    effective_limit = stall_guard + (90 - stall_guard) * np.clip((limits.wingborne_speed_mps - speed) / 8.0, 0, 1)
    free = np.abs([row["freestream_alpha_deg"] for row in rows])
    effective = np.abs([row["effective_alpha_deg"] for row in rows])
    return np.where(speed >= 0.5, np.minimum(free_limit - free, effective_limit - effective), np.inf)


def grid_cell(speed, tilt):
    """Return the center of the half-open bin containing the sampled state."""
    speed_cell = int(np.clip(np.floor(speed / SPEED_CELL + 0.5) * SPEED_CELL, 0, TARGET_SPEED))
    tilt_cell = int(np.clip(np.floor(tilt / TILT_CELL + 0.5) * TILT_CELL, 0, 90))
    return speed_cell, tilt_cell


def route_plan(aircraft, limits, exponent, time, guess):
    if exponent is None:
        return time, guess[:5], guess[5:]
    time, motion, control = optimize_trajectory(
        aircraft, limits, TARGET_SPEED, guess,
        duration_s=DURATION - TERMINAL_SETTLE_S, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD,
        tilt_schedule_exponent=exponent, flow_guard_extra_deg=1.0,
    )
    return np.r_[time, DURATION], np.column_stack((motion, motion[:, -1])), np.column_stack((control, control[:, -1]))


def explore_route(task):
    route_id, exponent, time, guess = task
    aircraft = load_aircraft_parameters()
    limits = study_limits(aircraft)
    dynamics = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    diagnostics = {"route_id": route_id, "tilt_schedule_exponent": exponent}
    try:
        time, motion, control = route_plan(aircraft, limits, exponent, time, guess)
    except RuntimeError as error:
        return diagnostics | {"status": "solver_failed", "reason": str(error).splitlines()[-1]}, [], {}, set()

    rt, motion, control, loads, discrepancy = replay_trajectory(
        aircraft, dynamics, limits, time, motion, control, step_s=REPLAY_DT, stop_on_limit=True,
    )
    rows = path_rows(aircraft, limits, rt, motion, control, loads)
    if rt[-1] < DURATION - 1e-6:
        reason = f"replay stopped at {rt[-1]:.2f} s after exceeding a dynamic limit"
        return diagnostics | {"status": "replay_failed", "reason": reason}, [], {}, set()
    metrics = dynamic_summary(aircraft, limits, rows)
    flow_margin = minimum_flow_guard_margin(aircraft, limits, rows)
    diagnostics["minimum_flow_guard_margin_deg"] = flow_margin
    checks = metrics["checks"] | {
        "flow_guards": flow_margin >= -FLOW_CHECK_TOLERANCE_DEG,
        "load_parity": discrepancy <= 1e-6,
    }
    diagnostics |= {key: value for key, value in metrics.items() if key not in ("checks", "passed")}
    violated = [key for key, value in checks.items() if not value]
    if violated:
        return diagnostics | {"status": "replay_failed", "reason": ", ".join(violated)}, [], {}, set()

    # Recheck trim along the route and at the first entry into each cell.
    cells = {}
    edges = set()
    prior = None
    for index, row in enumerate(rows):
        cell = grid_cell(row["airspeed_mps"], row["front_tilt_deg"])
        cells.setdefault(cell, index)
        if prior is not None and cell != prior:
            edges.add((prior, cell))
        prior = cell
    indices = sorted(set(range(0, len(rows), 5)) | set(cells.values()) | {len(rows) - 1})
    sample_rows = [rows[index] for index in indices]
    trim = check_static_path(dynamics, limits, sample_rows)
    if trim["failed_times_s"]:
        return diagnostics | {
            "status": "trim_recheck_failed",
            "reason": f"trim not found at {len(trim['failed_times_s'])} replay samples",
            "first_failed_trim_time_s": trim["failed_times_s"][0],
        }, [], {}, set()
    for row in sample_rows:
        row["route_id"] = route_id
    return diagnostics | {
        "status": "verified", "trim_checks": trim["points_checked"], "verified_cells": len(cells),
    }, sample_rows, {cell: rows[index] for cell, index in cells.items()}, edges


def corridor_grid(static, witnesses):
    rows = []
    for speed in range(0, int(TARGET_SPEED) + 1, SPEED_CELL):
        for tilt in range(0, 91, TILT_CELL):
            cell = speed, tilt
            evidence = witnesses.get(cell, [])
            first = evidence[0] if evidence else {}
            rows.append({
                "airspeed_cell_center_mps": speed,
                "tilt_cell_center_deg": tilt,
                "center_has_static_trim": static[cell],
                "dynamic_status": "witnessed" if evidence else "unverified",
                "verified_route_count": len(evidence),
                "witness_route_ids": " ".join(str(row["route_id"]) for row in evidence),
                "first_witness_speed_mps": first.get("airspeed_mps", ""),
                "first_witness_tilt_deg": first.get("front_tilt_deg", ""),
                "first_witness_time_s": first.get("time_s", ""),
                "first_witness_altitude_change_m": first.get("altitude_change_m", ""),
                "first_witness_pitch_rate_deg_s": first.get("pitch_rate_deg_s", ""),
                "first_witness_pitch_reserve_nm": first.get("differential_pitch_reserve_nm", ""),
            })
    return rows


def plot_corridor(path, cells, samples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    centers_x = np.arange(0, TARGET_SPEED + SPEED_CELL, SPEED_CELL)
    centers_y = np.arange(0, 90 + TILT_CELL, TILT_CELL)
    grid = np.zeros((len(centers_y), len(centers_x)))
    for row in cells:
        speed = row["airspeed_cell_center_mps"] // SPEED_CELL
        tilt = row["tilt_cell_center_deg"] // TILT_CELL
        grid[tilt, speed] = 2 if row["verified_route_count"] else int(row["center_has_static_trim"])
    colors = ["#f1f3f5", "#c5d4e4", "#2c987b"]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.pcolormesh(
        np.r_[centers_x - SPEED_CELL / 2, centers_x[-1] + SPEED_CELL / 2],
        np.r_[centers_y - TILT_CELL / 2, centers_y[-1] + TILT_CELL / 2],
        grid, cmap=ListedColormap(colors), norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
        edgecolors="white", linewidth=0.30,
    )
    routes = defaultdict(list)
    for row in samples:
        routes[row["route_id"]].append(row)
    for route in routes.values():
        ax.plot([row["airspeed_mps"] for row in route], [row["front_tilt_deg"] for row in route],
                color="#12374e", lw=0.8, alpha=0.50)
    ax.set(xlabel="True airspeed (m/s)", ylabel="Front rotor angle (°)",
           xlim=(-SPEED_CELL / 2, TARGET_SPEED + SPEED_CELL / 2), ylim=(-TILT_CELL / 2, 90 + TILT_CELL / 2),
           title="Dynamic corridor from hover to 30 m/s cruise")
    ax.set_xticks(np.arange(0, TARGET_SPEED + 1, 5))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.legend(handles=[Patch(color=color, label=label) for color, label in zip(colors, (
        "No static trim found at center", "Static trim at center, no dynamic witness", "Reached on a verified route",
    ))], loc="upper right", frameon=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, help="number of optimizer processes")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    static, time, guess = load_study()
    tasks = [(index, exponent, time, guess) for index, exponent in enumerate(SCHEDULES)]
    witnesses = defaultdict(list)
    adjacency = defaultdict(list)
    sample_rows = []
    route_rows = []
    CORRIDOR_OUTPUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report, samples, cells, edges in pool.map(explore_route, tasks):
            route_id = report["route_id"]
            route_rows.append(report)
            sample_rows.extend(samples)
            for cell, witness in cells.items():
                witnesses[cell].append(witness)
            for edge in edges:
                adjacency[edge].append(route_id)
            print(f"Route {route_id} {report['status']} {report.get('reason', '')}", flush=True)

    grid = corridor_grid(static, witnesses)
    report_fields = dict.fromkeys(key for row in route_rows for key in row)
    write_csv(CORRIDOR_OUTPUT / "route_verification.csv", [report_fields | row for row in route_rows])
    write_csv(CORRIDOR_OUTPUT / "dynamic_corridor.csv", grid)
    if not sample_rows:
        raise RuntimeError("No complete transitions passed verification")
    write_csv(CORRIDOR_OUTPUT / "verified_route_samples.csv", sample_rows)
    write_csv(CORRIDOR_OUTPUT / "observed_cell_transitions.csv", [{
        "from_speed_mps": source[0], "from_tilt_deg": source[1],
        "to_speed_mps": target[0], "to_tilt_deg": target[1],
        "witness_route_ids": " ".join(map(str, routes)),
    } for (source, target), routes in sorted(adjacency.items())])
    plot_corridor(CORRIDOR_OUTPUT / "dynamic_corridor.png", grid, sample_rows)

    summary = {
        "tilt_convention": TILT_CONVENTION,
        "routes_attempted": len(route_rows),
        "routes_verified": sum(row["status"] == "verified" for row in route_rows),
        "grid_cells": len(grid),
        "witnessed_cells": len(witnesses),
        "witnessed_cells_with_static_center_trim": sum(static[cell] for cell in witnesses),
        "observed_directed_cell_transitions": len(adjacency),
        "speed_cell_width_mps": SPEED_CELL,
        "tilt_cell_width_deg": TILT_CELL,
        "target_speed_mps": TARGET_SPEED,
        "transition_duration_s": DURATION,
        "alternate_route_terminal_hold_s": TERMINAL_SETTLE_S,
        "replay_step_s": REPLAY_DT,
        "flow_guard_comparison_tolerance_deg": FLOW_CHECK_TOLERANCE_DEG,
        "aerodynamic_load_onset_mps": [0.5, LOW_SPEED_FULL_LOAD],
        "dynamic_cell_definition": "At least one exact sampled state in the cell has trim and lies on a verified full route.",
    }
    with (CORRIDOR_OUTPUT / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"Verified {summary['routes_verified']} routes and {summary['witnessed_cells']} cells.", flush=True)


if __name__ == "__main__":
    main()
