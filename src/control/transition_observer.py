"""Filtered transition-state estimates for monitoring and protection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dynamics.frames import rotation_earth_to_body
from src.dynamics.state import AircraftState


@dataclass(frozen=True)
class TransitionStateEstimate:
    airspeed_mps: float
    alpha_rad: float
    beta_rad: float
    altitude_m: float
    climb_rate_mps: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    p_rad_s: float
    q_rad_s: float
    r_rad_s: float
    tilt_rad: float
    transition_blend: float
    phase: str


class TransitionStateObserver:
    """Low-pass observer for air-relative and transition-progress state."""

    _angle_fields = {"alpha_rad", "beta_rad", "roll_rad", "pitch_rad", "yaw_rad"}

    def __init__(self, time_constant_s: float = 0.25, cruise_speed_mps: float = 20.0):
        self.time_constant_s = float(time_constant_s)
        self.cruise_speed_mps = float(cruise_speed_mps)
        if not np.isfinite(self.time_constant_s) or self.time_constant_s <= 0.0:
            raise ValueError("observer time constant must be positive and finite")
        if not np.isfinite(self.cruise_speed_mps) or self.cruise_speed_mps <= 0.0:
            raise ValueError("cruise speed must be positive and finite")
        self._filtered: dict[str, float] | None = None

    def update(
        self,
        state: AircraftState,
        wind_earth: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> TransitionStateEstimate:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("observer time step must be positive and finite")

        position = self._vector(state.pos, "position")
        velocity = self._vector(state.vel, "velocity")
        attitude = self._vector(state.att, "attitude")
        body_rates = self._vector(state.omega, "body rates")
        wind = np.zeros(3, dtype=float) if wind_earth is None else self._vector(wind_earth, "wind")
        tilt = float(state.tilt)
        if not np.isfinite(tilt) or not 0.0 <= tilt <= np.pi / 2.0:
            raise ValueError("tilt must be finite and between 0 and 90 degrees")

        air_velocity_body = rotation_earth_to_body(*attitude) @ (velocity - wind)
        airspeed = float(np.linalg.norm(air_velocity_body))
        alpha = float(np.arctan2(air_velocity_body[2], air_velocity_body[0]))
        beta = float(np.arcsin(np.clip(air_velocity_body[1] / max(airspeed, 1e-9), -1.0, 1.0)))
        observed = {
            "airspeed_mps": airspeed,
            "alpha_rad": alpha,
            "beta_rad": beta,
            "altitude_m": float(-position[2]),
            "climb_rate_mps": float(-velocity[2]),
            "roll_rad": float(attitude[0]),
            "pitch_rad": float(attitude[1]),
            "yaw_rad": float(attitude[2]),
            "p_rad_s": float(body_rates[0]),
            "q_rad_s": float(body_rates[1]),
            "r_rad_s": float(body_rates[2]),
            "tilt_rad": tilt,
        }

        if self._filtered is None:
            self._filtered = observed.copy()
        else:
            blend = 1.0 - np.exp(-dt / self.time_constant_s)
            for name, value in observed.items():
                previous = self._filtered[name]
                difference = value - previous
                if name in self._angle_fields:
                    difference = float(np.arctan2(np.sin(difference), np.cos(difference)))
                self._filtered[name] = previous + blend * difference

        airspeed_blend = np.clip(
            self._filtered["airspeed_mps"] / self.cruise_speed_mps, 0.0, 1.0
        )
        tilt_blend = np.clip(self._filtered["tilt_rad"] / (np.pi / 2.0), 0.0, 1.0)
        transition_blend = float(0.5 * (airspeed_blend + tilt_blend))
        phase = "hover" if transition_blend < 0.2 else (
            "cruise" if transition_blend > 0.8 else "transition"
        )
        return TransitionStateEstimate(
            **self._filtered,
            transition_blend=transition_blend,
            phase=phase,
        )

    def reset(self) -> None:
        self._filtered = None

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite 3-vector")
        return vector
