"""Rank complete transition replays and test the selected route on held-out gusts."""

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scripts.corridor_followup import make_plan
from scripts.corridor_uncertainty import INNER_MARGINS, Scenario, fill_margins, run_case, scenarios, screen_route
from scripts.corridor_validation import read_rows, write_reports
from scripts.dynamic_corridor import load_study
from scripts.refined_corridor import ANGLES, REPORT_DIR, SPEEDS, coverage
from scripts.transition_study import DURATION, LOW_SPEED_FULL_LOAD, check_static_path, path_rows, replay_trajectory, study_limits
from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.wind.gust import DrydenGust


OUTPUT = REPORT_DIR / "transition_review"
TAIL_RESERVE = np.deg2rad(1.6)
DRYDEN_DT = 0.01


def dryden_trace(seed):
    """Precompute a common, repeatable along-track gust for all candidates."""
    time = np.linspace(0, DURATION, round(DURATION / DRYDEN_DT) + 1)
    model = DrydenGust(sigma=(0.35, 0.0, 0.0), airspeed=15.0, tau=1.0, seed=seed)
    gust = np.array([model.step(DRYDEN_DT)[0] for _ in time])
    # Start from a still-air trim and test terminal settling after the gust.
    envelope = np.clip((time - 3.0) / 2.0, 0, 1) * np.clip((29.0 - time) / 2.0, 0, 1)
    return time, gust * envelope


def trial(task):
    name, plan, reserve, scenario, seed = task
    wind = None
    if seed is not None:
        t, gust = dryden_trace(seed)
        wind = lambda moment: np.interp(moment, t, gust)
    result, _ = run_case((name, plan, scenario), wind_at=wind, tail_reserve_rad=reserve)
    return result | {"wind_family": "Dryden-like" if seed is not None else "prescribed",
                     "wind_seed": seed, "controller_tail_reserve_deg": np.rad2deg(reserve)}


def setup():
    aircraft = load_aircraft_parameters()
    return aircraft, study_limits(aircraft), build_aircraft_dynamics(
        aircraft, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD,
    )


def nominal_check(plan, reserve):
    aircraft, limits, plant = setup()
    t, x, u, loads, discrepancy = replay_trajectory(
        aircraft, plant, limits, *plan, step_s=.01, stop_on_limit=True, tail_reserve_rad=reserve,
    )
    rows = path_rows(aircraft, limits, t, x, u, loads)
    fill_margins(aircraft, limits, rows)
    metrics, failed = screen_route(aircraft, limits, rows, inner=True)
    if discrepancy > 1e-6:
        failed.append("load_parity")
    return rows, metrics, failed


def fault_report(plan, reserve):
    aircraft, limits, plant = setup()
    reports = []
    for pair in (0, 1):
        def availability(time):
            values = np.ones(2)
            if time >= 12:
                values[pair] = 0
            return values

        t, x, u, loads, _ = replay_trajectory(
            aircraft, plant, limits, *plan, step_s=.02, stop_on_limit=True,
            availability_at=availability, tail_reserve_rad=reserve,
        )
        rows = path_rows(aircraft, limits, t, x, u, loads)
        fill_margins(aircraft, limits, rows)
        metrics, failed = screen_route(aircraft, limits, rows, inner=True)
        reports.append({"failed_pair": ("front", "rear")[pair], "injection_time_s": 12.,
                        "completed_time_s": metrics["completed_time_s"], "passed": not failed,
                        "failed_checks": " ".join(failed),
                        "first_violated_constraint": metrics.get("first_violated_constraint", ""),
                        "first_violation_time_s": metrics.get("first_violation_time_s", ""),
                        "airspeed_at_injection_mps": rows[np.searchsorted(t, 12)]["airspeed_mps"] if t[-1] >= 12 else None})
    return reports


def plot_review(rows, overlap, ranking, selected, holdout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axes[0, 0]
    ax.pcolormesh(SPEEDS, ANGLES, [[(s, a) in overlap for s in SPEEDS] for a in ANGLES],
                  cmap=matplotlib.colors.ListedColormap(["#eef2f6", "#bdd9d2"]), shading="nearest")
    ax.plot([r["airspeed_mps"] for r in rows], [r["front_tilt_deg"] for r in rows], color="#113550", lw=2,
            label="Selected nominal replay")
    ax.set(xlabel="True airspeed (m/s)", ylabel="Front rotor angle (°)", title="Static and nominal dynamic overlap")
    ax.legend(handles=[ax.lines[-1], Patch(facecolor="#bdd9d2", label="Previously mapped overlap")], fontsize=8)
    t = [row["time_s"] for row in rows]
    ax = axes[0, 1]
    for key, label in (("front_vertical_thrust_n", "Front thrust"), ("rear_vertical_thrust_n", "Rear thrust"),
                       ("wing_vertical_lift_n", "Wing lift"), ("vtail_vertical_force_n", "V-tail force")):
        ax.plot(t, [row[key] for row in rows], label=label)
    ax.set(xlabel="Time (s)", ylabel="Vertical force (N)", title="Rotor-to-wing load transfer")
    ax.legend(ncols=2, fontsize=8)
    ax = axes[1, 0]
    ax.plot(t, [r["tail_travel_reserve_deg"] for r in rows], label="V-tail travel reserve")
    ax.plot(t, [r["stall_margin_deg"] if r["airspeed_mps"] >= 12 else np.nan for r in rows], label="Wingborne stall margin")
    ax.axhline(INNER_MARGINS["tail_travel_reserve_deg"], color="#377eb8", ls=":")
    ax.axhline(INNER_MARGINS["wingborne_stall_margin_deg"], color="#e98830", ls=":")
    ax.set(xlabel="Time (s)", ylabel="Margin (°)", title="Nominal margins and inner test limits")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    names = [r["candidate"] for r in ranking] + ["Selected holdout"]
    totals = [r["cases"] for r in ranking] + [len(holdout)]
    passed = [r["passed"] for r in ranking] + [sum(r["passed"] for r in holdout)]
    y = np.arange(len(names))
    ax.barh(y, totals, color="#ebc7bf", label="Cases attempted")
    ax.barh(y, passed, color="#25836b", label="Passed")
    for idx, (ok, total) in enumerate(zip(passed, totals)):
        ax.text(total + .4, idx, f"{ok}/{total}", va="center")
    ax.set(yticks=y, yticklabels=names, xlim=(0, max(totals) + 5), xlabel="Complete-replay test cases",
           title="Scenario outcomes")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    fig.suptitle(f"Forward-only tiltrotor · {selected} · simulation comparison", fontsize=16)
    fig.savefig(OUTPUT / "transition_review.png", dpi=175)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, time, guess = load_study()
    with ProcessPoolExecutor(max_workers=4) as pool:
        plans = dict((i, plan) for i, plan, _ in pool.map(make_plan, [(i, time, guess) for i in (3, 12)]) if plan is not None)
        choices = {}
        if 3 in plans:
            choices["Reference 3"] = plans[3], 0.
        if 12 in plans:
            choices |= {"Margin plan 12": (plans[12], 0.), "Reserve-aware 12": (plans[12], TAIL_RESERVE)}
        bank = [(scenario, None) for scenario in scenarios()] + [
            (Scenario(f"dryden_{seed}"), seed) for seed in (701, 702, 703, 704)
        ]
        nominal = {name: nominal_check(plan, reserve) for name, (plan, reserve) in choices.items()}
        tasks = [(name, plan, reserve, scenario, seed) for name, (plan, reserve) in choices.items()
                 for scenario, seed in bank]
        cases = list(pool.map(trial, tasks))
        print(f"Completed {len(cases)} comparison replays.", flush=True)
        ranking = []
        for name in choices:
            runs = [row for row in cases if row["route_id"] == name]
            _, metrics, failures = nominal[name]
            ranking.append({"candidate": name, "cases": len(runs), "passed": sum(r["passed"] for r in runs),
                            "dryden_passed": sum(r["passed"] for r in runs if r["wind_family"] == "Dryden-like"),
                            "nominal_inner_pass": not failures, "nominal_failures": " ".join(failures), **metrics})
        ranking.sort(key=lambda r: (not r["nominal_inner_pass"], -r["passed"], r["energy_wh"]))
        selected = ranking[0]["candidate"]
        plan, reserve = choices[selected]
        holdout = list(pool.map(trial, [(selected, plan, reserve, Scenario(f"holdout_{seed}"), seed)
                                        for seed in (901, 902, 903, 904)]))
    faults = fault_report(plan, reserve)
    rows = nominal[selected][0]
    _, limits, dynamics = setup()
    cells, _ = coverage(rows)
    samples = [rows[i] for i in sorted(set(cells.values()) | set(range(0, len(rows), 10)) | {len(rows) - 1})]
    trim = check_static_path(dynamics, limits, samples)
    grid = read_rows(REPORT_DIR / "corridor_evidence.csv")
    overlap = {(float(r["airspeed_cell_center_mps"]), float(r["tilt_cell_center_deg"])) for r in grid
               if r["static_center_trim_found"] == "True" and r["nominal_evidence"] == "constraint_checked"}
    write_reports(OUTPUT / "selected_nominal_trace.csv", rows)
    write_reports(OUTPUT / "candidate_comparison.csv", ranking)
    write_reports(OUTPUT / "scenario_results.csv", cases + holdout)
    write_reports(OUTPUT / "pair_failure_results.csv", faults)
    plot_review(rows, overlap, ranking, selected, holdout)
    summary = {
        "selected": selected, "ranking_rule": "Nominal inner pass, comparison pass count, then modeled energy.",
        "candidate_scores": ranking, "held_out_passed": sum(r["passed"] for r in holdout),
        "held_out_cases": len(holdout), "selected_exact_trim_rechecks": trim,
        "selected_cells": len(cells), "selected_cells_in_preexisting_overlap": len(set(cells) & overlap),
        "first_violations_selected": Counter(r.get("first_violated_constraint", "pass") for r in cases if r["route_id"] == selected),
        "pair_failures": faults, "dryden_sigma_along_track_mps": .35, "dryden_tau_s": 1.,
        "scope": "Modeled symmetric longitudinal feedback and prescribed tilt schedule. No fault detection or recovery law.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda value: value.item()), encoding="utf-8")
    print(f"Selected {selected}. Held-out pass count {summary['held_out_passed']}/{len(holdout)}.", flush=True)


if __name__ == "__main__":
    main()
