"""Re-solve transition across assumed loaded rotor-speed limits."""

import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import optimize_longitudinal_transition as optimizer
from src.configuration.aircraft_parameters import load_aircraft_parameters


DEFAULT_SPEED_LIMIT_FRACTIONS = (
    1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.5,
)


def run_sensitivity_sweep(speed_limit_fractions=DEFAULT_SPEED_LIMIT_FRACTIONS):
    """Solve each transition with a loaded-RPM cap relative to the YAML estimate.

    Fractions are scenario assumptions, not measured motor performance.
    """
    aircraft = load_aircraft_parameters()
    fractions = np.asarray(speed_limit_fractions, dtype=float)
    if (
        fractions.ndim != 1
        or fractions.size == 0
        or not np.isfinite(fractions).all()
        or np.any(fractions <= 0.0)
        or np.any(fractions > 1.0)
        or np.unique(fractions).size != fractions.size
    ):
        raise ValueError("speed-limit fractions must be unique finite values in (0, 1]")

    results = []
    for fraction in fractions:
        assumed_front_limit = aircraft.estimated_motor_rpm * fraction
        assumed_rear_limit = assumed_front_limit * np.sqrt(
            1.0 - aircraft.rear_thrust_reserve_fraction
        )
        maximum_hover_support = (
            2.0 * aircraft.max_thrust_per_rotor * fraction**2
            * (2.0 - aircraft.rear_thrust_reserve_fraction)
        )
        if maximum_hover_support < aircraft.mass * aircraft.gravity:
            results.append({
                "speed_limit_fraction": float(fraction),
                "rpm_limit_basis": "fraction_of_yaml_kv_nominal_estimate_not_measured",
                "assumed_front_rpm_limit": assumed_front_limit,
                "assumed_rear_rpm_limit": assumed_rear_limit,
                "status": "infeasible_hover_thrust",
                "feasible": False,
                "failure_reason": (
                    f"maximum hover rotor support "
                    f"{maximum_hover_support:.2f} N is below aircraft weight "
                    f"{aircraft.mass * aircraft.gravity:.2f} N"
                ),
            })
            continue
        try:
            data = optimizer.add_model_outputs(
                optimizer.solve_transition(float(fraction))
            )
        except RuntimeError as error:
            results.append({
                "speed_limit_fraction": float(fraction),
                "rpm_limit_basis": "fraction_of_yaml_kv_nominal_estimate_not_measured",
                "assumed_front_rpm_limit": assumed_front_limit,
                "assumed_rear_rpm_limit": assumed_rear_limit,
                "status": "solver_failed",
                "feasible": False,
                "failure_reason": str(error),
            })
            continue

        results.append({
            "speed_limit_fraction": float(fraction),
            "rpm_limit_basis": "fraction_of_yaml_kv_nominal_estimate_not_measured",
            "assumed_front_rpm_limit": assumed_front_limit,
            "assumed_rear_rpm_limit": assumed_rear_limit,
            "status": "solved",
            "feasible": True,
            "failure_reason": "",
            "max_front_rotor_rpm": float(data["front_rotor_rpm"].max()),
            "max_rear_rotor_rpm": float(data["rear_rotor_rpm"].max()),
            "minimum_front_rpm_margin": float(data["front_rpm_margin"].min()),
            "minimum_rear_rpm_margin": float(data["rear_rpm_margin"].min()),
            "final_speed_mps": float(data["vx_mps"][-1]),
            "minimum_altitude_m": float(data["altitude_m"].min()),
            "maximum_altitude_loss_m": float(data["max_altitude_loss_m"]),
        })
    return results


def write_results_csv(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "speed_limit_fraction",
        "rpm_limit_basis",
        "assumed_front_rpm_limit",
        "assumed_rear_rpm_limit",
        "status",
        "feasible",
        "max_front_rotor_rpm",
        "max_rear_rotor_rpm",
        "minimum_front_rpm_margin",
        "minimum_rear_rpm_margin",
        "final_speed_mps",
        "minimum_altitude_m",
        "maximum_altitude_loss_m",
        "failure_reason",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({
                column: result.get(column, "")
                for column in columns
            })


def main():
    aircraft = load_aircraft_parameters()
    results = run_sensitivity_sweep()
    output_path = Path("figures") / "rotor_speed_limit_sensitivity.csv"
    write_results_csv(results, output_path)
    print(f"Nominal YAML-derived RPM estimate: {aircraft.estimated_motor_rpm:.1f}")
    print("Assumed loaded-RPM fractions are sensitivities, not motor measurements.")
    for result in results:
        fraction = result["speed_limit_fraction"]
        if result["feasible"]:
            print(
                f"{fraction:.0%} ({result['assumed_front_rpm_limit']:.0f} RPM): "
                f"solved; peak front/rear RPM="
                f"{result['max_front_rotor_rpm']:.0f}/"
                f"{result['max_rear_rotor_rpm']:.0f}; "
                f"min altitude={result['minimum_altitude_m']:.2f} m"
            )
        else:
            status_label = (
                "infeasible before solve"
                if result["status"] == "infeasible_hover_thrust"
                else "solver failed"
            )
            print(
                f"{fraction:.0%} ({result['assumed_front_rpm_limit']:.0f} RPM): "
                f"{status_label}: {result['failure_reason']}"
            )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
