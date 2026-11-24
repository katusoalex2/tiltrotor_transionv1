"""Reference limiting against transition and attitude safety bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.control.transition_observer import TransitionStateEstimate


@dataclass(frozen=True)
class ProtectedReference:
    speed_mps: float
    tilt_rad: float
    protected: bool
    within_envelope: bool
    reasons: tuple[str, ...]
    minimum_speed_mps: float
    maximum_speed_mps: float


class EnvelopeProtection:
    """Limit speed and tilt references using a computed transition envelope."""

    def __init__(
        self,
        envelope: Mapping[str, np.ndarray],
        *,
        max_alpha_rad: float = np.deg2rad(15.0),
        max_pitch_rad: float = np.deg2rad(60.0),
        max_body_rate_rad_s: float = np.deg2rad(180.0),
        speed_margin_mps: float = 0.5,
    ):
        tilt_deg = np.asarray(envelope["tilt_deg"], dtype=float)
        minimum = np.asarray(envelope["minimum_speed_mps"], dtype=float)
        maximum = np.asarray(envelope["maximum_speed_mps"], dtype=float)
        feasible = np.asarray(envelope["feasible"], dtype=bool)
        if not (tilt_deg.ndim == minimum.ndim == maximum.ndim == feasible.ndim == 1):
            raise ValueError("transition envelope arrays must be one-dimensional")
        if not (tilt_deg.shape == minimum.shape == maximum.shape == feasible.shape):
            raise ValueError("transition envelope arrays must have matching shapes")
        if tilt_deg.size == 0 or not np.all(np.isfinite(tilt_deg)):
            raise ValueError("transition envelope must contain finite tilt samples")
        if np.any(np.diff(tilt_deg) <= 0.0):
            raise ValueError("transition envelope tilt samples must be strictly increasing")

        valid = feasible & np.isfinite(minimum) & np.isfinite(maximum)
        if not np.any(valid):
            raise ValueError("transition envelope has no feasible speed samples")
        if np.any(minimum[valid] < 0.0) or np.any(maximum[valid] < minimum[valid]):
            raise ValueError("transition envelope speed bounds are invalid")
        self._tilt_deg = tilt_deg[valid]
        self._minimum_speed = minimum[valid]
        self._maximum_speed = maximum[valid]

        limits = (max_alpha_rad, max_pitch_rad, max_body_rate_rad_s, speed_margin_mps)
        if not all(np.isfinite(value) for value in limits):
            raise ValueError("protection limits must be finite")
        if max_alpha_rad <= 0.0 or max_pitch_rad <= 0.0 or max_body_rate_rad_s <= 0.0:
            raise ValueError("attitude and rate limits must be positive")
        if speed_margin_mps < 0.0:
            raise ValueError("speed margin cannot be negative")
        self.max_alpha_rad = float(max_alpha_rad)
        self.max_pitch_rad = float(max_pitch_rad)
        self.max_body_rate_rad_s = float(max_body_rate_rad_s)
        self.speed_margin_mps = float(speed_margin_mps)

    def apply(
        self,
        target_speed_mps: float,
        target_tilt_rad: float,
        estimate: TransitionStateEstimate,
        available_control_authority: float = 1.0,
    ) -> ProtectedReference:
        target_speed = float(target_speed_mps)
        target_tilt = float(target_tilt_rad)
        available_control_authority = float(available_control_authority)
        if not np.isfinite(target_speed) or target_speed < 0.0:
            raise ValueError("target speed must be finite and non-negative")
        if not np.isfinite(target_tilt) or not 0.0 <= target_tilt <= np.pi / 2.0:
            raise ValueError("target tilt must be finite and between 0 and 90 degrees")
        if (
            not np.isfinite(available_control_authority)
            or not 0.0 <= available_control_authority <= 1.0
        ):
            raise ValueError("available control authority must be between 0 and 1")

        reasons: list[str] = []
        protected_tilt = estimate.tilt_rad + available_control_authority * (
            target_tilt - estimate.tilt_rad
        )
        if available_control_authority < 1.0:
            reasons.append("control_authority_limited")
        requested_min, requested_max = self._limits_at(target_tilt)
        measured_speed = float(estimate.airspeed_mps)

        rates = (estimate.p_rad_s, estimate.q_rad_s, estimate.r_rad_s)
        estimate_values = (
            measured_speed,
            estimate.alpha_rad,
            estimate.pitch_rad,
            estimate.tilt_rad,
            *rates,
        )
        if not all(np.isfinite(value) for value in estimate_values):
            raise ValueError("observer estimate must contain finite safety-state values")
        if not 0.0 <= estimate.tilt_rad <= np.pi / 2.0:
            raise ValueError("observer tilt must be between 0 and 90 degrees")
        if measured_speed < 0.0:
            raise ValueError("observer airspeed cannot be negative")
        if abs(estimate.alpha_rad) > self.max_alpha_rad:
            protected_tilt = min(protected_tilt, estimate.tilt_rad)
            reasons.append("angle_of_attack_limit")
        if abs(estimate.pitch_rad) > self.max_pitch_rad:
            protected_tilt = min(protected_tilt, estimate.tilt_rad)
            reasons.append("pitch_limit")
        if max(abs(rate) for rate in rates) > self.max_body_rate_rad_s:
            protected_tilt = min(protected_tilt, estimate.tilt_rad)
            reasons.append("body_rate_limit")

        if measured_speed < requested_min - self.speed_margin_mps:
            protected_tilt = min(protected_tilt, estimate.tilt_rad)
            reasons.append("airspeed_below_transition_envelope")
        elif measured_speed > requested_max + self.speed_margin_mps:
            protected_tilt = min(protected_tilt, estimate.tilt_rad)
            reasons.append("airspeed_above_transition_envelope")

        minimum_speed, maximum_speed = self._limits_at(protected_tilt)
        margin = self.speed_margin_mps if protected_tilt > 1e-6 else 0.0
        lower_speed = minimum_speed + margin
        upper_speed = maximum_speed - margin
        if lower_speed > upper_speed:
            lower_speed = minimum_speed
            upper_speed = maximum_speed
            reasons.append("narrow_transition_envelope")
        protected_speed = float(np.clip(target_speed, lower_speed, upper_speed))
        if not np.isclose(protected_speed, target_speed):
            reasons.append("speed_reference_limited")

        within_envelope = (
            minimum_speed - self.speed_margin_mps
            <= measured_speed
            <= maximum_speed + self.speed_margin_mps
        )
        return ProtectedReference(
            speed_mps=protected_speed,
            tilt_rad=float(protected_tilt),
            protected=bool(reasons),
            within_envelope=bool(within_envelope),
            reasons=tuple(dict.fromkeys(reasons)),
            minimum_speed_mps=float(minimum_speed),
            maximum_speed_mps=float(maximum_speed),
        )

    def _limits_at(self, tilt_rad: float) -> tuple[float, float]:
        tilt_deg = float(np.rad2deg(np.clip(tilt_rad, 0.0, np.pi / 2.0)))
        minimum = float(np.interp(tilt_deg, self._tilt_deg, self._minimum_speed))
        maximum = float(np.interp(tilt_deg, self._tilt_deg, self._maximum_speed))
        return minimum, maximum
