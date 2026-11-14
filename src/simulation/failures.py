"""Deterministic actuator-failure injection for simulation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


ACTUATOR_CHANNELS = (
    "omega_fl",
    "omega_fr",
    "omega_rl",
    "omega_rr",
    "tilt",
    "delta_left",
    "delta_right",
)


@dataclass(frozen=True)
class ActuatorFailure:
    channel: str
    start_time_s: float
    remaining_effectiveness: float = 0.0
    stuck_value: float | None = None

    def __post_init__(self) -> None:
        if self.channel not in ACTUATOR_CHANNELS:
            raise ValueError(f"unsupported actuator channel: {self.channel}")
        if not np.isfinite(self.start_time_s) or self.start_time_s < 0.0:
            raise ValueError("failure start time must be finite and non-negative")
        if (
            not np.isfinite(self.remaining_effectiveness)
            or not 0.0 <= self.remaining_effectiveness <= 1.0
        ):
            raise ValueError("remaining effectiveness must be between 0 and 1")
        if self.stuck_value is not None and not np.isfinite(self.stuck_value):
            raise ValueError("stuck value must be finite")
        if self.stuck_value is not None and self.remaining_effectiveness != 0.0:
            raise ValueError("a stuck failure cannot also specify remaining effectiveness")

    def apply(self, commands: Mapping[str, float], time_s: float) -> dict[str, float]:
        time_s = float(time_s)
        if not np.isfinite(time_s):
            raise ValueError("failure application time must be finite")
        if self.channel not in commands:
            raise KeyError(f"actuator command is missing channel {self.channel!r}")
        result = {name: float(value) for name, value in commands.items()}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("actuator commands must be finite")
        if time_s < self.start_time_s:
            return result
        if self.stuck_value is not None:
            result[self.channel] = float(self.stuck_value)
        else:
            result[self.channel] *= self.remaining_effectiveness
        return result
