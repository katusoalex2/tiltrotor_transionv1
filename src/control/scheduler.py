"""Blend controller gains through hover, transition, and cruise."""

from __future__ import annotations

import numpy as np


class GainScheduler:
    def __init__(
        self,
        hover_gains: dict[str, float] | None = None,
        cruise_gains: dict[str, float] | None = None,
        cruise_speed: float = 20.0,
    ):
        self.hover_gains = hover_gains or {"kp": 1.2, "ki": 0.1, "kd": 0.3}
        self.cruise_gains = cruise_gains or {"kp": 0.2, "ki": 0.01, "kd": 0.05}
        self.cruise_speed = float(cruise_speed)
        if self.cruise_speed <= 0.0:
            raise ValueError("cruise speed must be positive")
        for gains in (self.hover_gains, self.cruise_gains):
            if set(gains) != {"kp", "ki", "kd"} or not np.all(
                np.isfinite([gains["kp"], gains["ki"], gains["kd"]])
            ):
                raise ValueError("gain sets must contain finite kp, ki, and kd values")

    def transition_blend(self, current_speed: float, tilt_rad: float | None = None) -> float:
        """Return a linear 0–1 blend from speed and nacelle-tilt progress."""
        if not np.isfinite(current_speed):
            raise ValueError("current speed must be finite")
        speed_blend = np.clip(abs(current_speed) / self.cruise_speed, 0.0, 1.0)
        if tilt_rad is None:
            return float(speed_blend)
        if not np.isfinite(tilt_rad) or not 0.0 <= tilt_rad <= np.pi / 2.0:
            raise ValueError("tilt must be between 0 and 90 degrees")
        tilt_blend = tilt_rad / (np.pi / 2.0)
        return float(0.5 * (speed_blend + tilt_blend))

    def get_gains(self, current_speed: float, tilt_rad: float | None = None) -> dict[str, float]:
        """Interpolate PID gains across the current transition condition."""
        blend = self.transition_blend(current_speed, tilt_rad)
        return {
            key: float((1.0 - blend) * self.hover_gains[key] + blend * self.cruise_gains[key])
            for key in ("kp", "ki", "kd")
        }
