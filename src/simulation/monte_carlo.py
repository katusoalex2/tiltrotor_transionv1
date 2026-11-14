"""Seeded Monte Carlo scenario generation and safety-result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from src.simulation.failures import ACTUATOR_CHANNELS, ActuatorFailure
from src.wind.gust import DrydenGust


@dataclass(frozen=True)
class MonteCarloScenario:
    trial_index: int
    seed: int
    gust_sigma_mps: tuple[float, float, float]
    actuator_failure: ActuatorFailure | None

    def make_gust(self, airspeed_mps: float = 15.0) -> DrydenGust:
        """Create this scenario's reproducible gust model."""
        return DrydenGust(
            sigma=self.gust_sigma_mps,
            airspeed=airspeed_mps,
            seed=self.seed,
        )


@dataclass(frozen=True)
class MonteCarloTrialResult:
    scenario: MonteCarloScenario
    metrics: Mapping[str, float | bool]
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MonteCarloReport:
    seed: int
    trials: tuple[MonteCarloTrialResult, ...]
    success_rate: float
    metric_percentiles: Mapping[str, tuple[float, float, float]]


class MonteCarloValidator:
    """Run reproducible environmental/failure cases through a caller-supplied simulation."""

    DEFAULT_LIMITS = {
        "max_abs_pitch_rad": ("max", float(np.deg2rad(60.0))),
        "max_abs_alpha_rad": ("max", float(np.deg2rad(25.0))),
        "min_altitude_m": ("min", 0.0),
    }

    def __init__(
        self,
        trials: int = 100,
        seed: int = 0,
        *,
        gust_sigma_max_mps: tuple[float, float, float] = (3.0, 3.0, 2.0),
        failure_probability: float = 0.1,
        failure_time_range_s: tuple[float, float] = (1.0, 20.0),
        safety_limits: Mapping[str, tuple[str, float]] | None = None,
    ):
        if not isinstance(trials, int) or trials <= 0:
            raise ValueError("trial count must be a positive integer")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        gust_max = np.asarray(gust_sigma_max_mps, dtype=float)
        if gust_max.shape != (3,) or not np.all(np.isfinite(gust_max)) or np.any(gust_max < 0.0):
            raise ValueError("maximum gust sigma must be a finite non-negative 3-vector")
        if not np.isfinite(failure_probability) or not 0.0 <= failure_probability <= 1.0:
            raise ValueError("failure probability must be between 0 and 1")
        time_range = np.asarray(failure_time_range_s, dtype=float)
        if (
            time_range.shape != (2,)
            or not np.all(np.isfinite(time_range))
            or time_range[0] < 0.0
            or time_range[1] < time_range[0]
        ):
            raise ValueError("failure time range must be finite, ordered, and non-negative")

        self.trials = trials
        self.seed = seed
        self.gust_sigma_max_mps = gust_max
        self.failure_probability = float(failure_probability)
        self.failure_time_range_s = (float(time_range[0]), float(time_range[1]))
        self.safety_limits = dict(
            self.DEFAULT_LIMITS if safety_limits is None else safety_limits
        )
        for name, (direction, limit) in self.safety_limits.items():
            if direction not in ("max", "min") or not np.isfinite(limit):
                raise ValueError(f"invalid safety limit for metric {name!r}")

    def run(
        self,
        simulate_trial: Callable[
            [MonteCarloScenario], Mapping[str, float | int | bool]
        ],
    ) -> MonteCarloReport:
        results: list[MonteCarloTrialResult] = []
        expected_metric_names: set[str] | None = None
        child_seeds = np.random.SeedSequence(self.seed).spawn(self.trials)

        for trial_index, child_seed in enumerate(child_seeds):
            rng = np.random.default_rng(child_seed)
            trial_seed = int(child_seed.generate_state(1, dtype=np.uint32)[0])
            gust_sigma = tuple(
                float(value)
                for value in rng.uniform(0.0, self.gust_sigma_max_mps, size=3)
            )
            actuator_failure = None
            if rng.random() < self.failure_probability:
                channel = str(rng.choice(ACTUATOR_CHANNELS))
                failure_time = float(rng.uniform(*self.failure_time_range_s))
                actuator_failure = ActuatorFailure(
                    channel=channel,
                    start_time_s=failure_time,
                    remaining_effectiveness=float(rng.uniform(0.0, 0.5)),
                )
            scenario = MonteCarloScenario(
                trial_index=trial_index,
                seed=trial_seed,
                gust_sigma_mps=gust_sigma,
                actuator_failure=actuator_failure,
            )

            raw_metrics = simulate_trial(scenario)
            if not isinstance(raw_metrics, Mapping):
                raise TypeError("simulation callback must return a metric mapping")
            metrics: dict[str, float | bool] = {}
            for name, value in raw_metrics.items():
                if isinstance(value, (bool, np.bool_)):
                    metrics[name] = bool(value)
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_value = float(value)
                    if not np.isfinite(numeric_value):
                        raise ValueError(f"simulation metric {name!r} must be finite")
                    metrics[name] = numeric_value
                else:
                    raise TypeError(f"simulation metric {name!r} must be numeric or boolean")

            if expected_metric_names is None:
                expected_metric_names = set(metrics)
                missing_limits = set(self.safety_limits) - expected_metric_names
                if missing_limits:
                    raise ValueError(
                        f"simulation metrics are missing safety fields: {sorted(missing_limits)}"
                    )
            elif set(metrics) != expected_metric_names:
                raise ValueError("simulation callback must return the same metric names for every trial")

            reasons: list[str] = []
            if "completed" in metrics:
                if not isinstance(metrics["completed"], bool):
                    raise TypeError("simulation metric 'completed' must be boolean")
                if not metrics["completed"]:
                    reasons.append("simulation_incomplete")
            for name, (direction, limit) in self.safety_limits.items():
                if isinstance(metrics[name], bool):
                    raise TypeError(f"safety metric {name!r} must be numeric")
                value = float(metrics[name])
                violated = value > limit if direction == "max" else value < limit
                if violated:
                    reasons.append(f"{name}_{direction}_limit")
            results.append(
                MonteCarloTrialResult(
                    scenario=scenario,
                    metrics=metrics,
                    passed=not reasons,
                    failure_reasons=tuple(reasons),
                )
            )

        numeric_names = [
            name
            for name in (expected_metric_names or set())
            if all(not isinstance(result.metrics[name], bool) for result in results)
        ]
        percentiles = {
            name: tuple(
                float(value)
                for value in np.percentile(
                    [float(result.metrics[name]) for result in results],
                    [5.0, 50.0, 95.0],
                )
            )
            for name in numeric_names
        }
        success_rate = sum(result.passed for result in results) / len(results)
        return MonteCarloReport(
            seed=self.seed,
            trials=tuple(results),
            success_rate=float(success_rate),
            metric_percentiles=percentiles,
        )
