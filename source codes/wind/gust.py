"""Atmospheric wind models used by the tilt-rotor simulation."""

from __future__ import annotations

import numpy as np


class WindField:
    """Base class for gust models."""

    def __init__(self, mean_speed=5.0, mean_direction_deg=0.0,
                 turbulence_intensity=0.2, ref_altitude=10.0, surface_roughness=0.03):
        self.mean_speed = float(mean_speed)
        self.mean_direction = np.deg2rad(mean_direction_deg)
        self.turbulence = float(turbulence_intensity)
        self.ref_alt = float(ref_altitude)
        self.z0 = float(surface_roughness)
        self.time = 0.0
        self.gust = np.zeros(3, dtype=float)

    def get_wind(self, t: float, altitude: float = 100.0) -> np.ndarray:
        raise NotImplementedError

    def step(self, dt: float) -> np.ndarray:
        raise NotImplementedError


class ConstantGust(WindField):
    """Simple constant gust used in the tests and demos."""

    def __init__(self, u: float = 0.0, v: float = 0.0, w: float = 0.0):
        super().__init__()
        self.gust = np.array([u, v, w], dtype=float)
        self.vector = self.gust.copy()

    def step(self, dt: float) -> np.ndarray:
        self.time += float(dt)
        return self.gust.copy()

    def get_wind(self, t: float, altitude: float = 100.0) -> np.ndarray:
        return self.gust.copy()


class DiscreteGust(WindField):
    """Finite-duration half-cosine gust with NED wind components."""

    def __init__(
        self,
        start_time_s: float,
        duration_s: float,
        peak_wind_ned: tuple[float, float, float],
    ):
        super().__init__()
        self.start_time_s = float(start_time_s)
        self.duration_s = float(duration_s)
        self.peak_wind_ned = np.asarray(peak_wind_ned, dtype=float)
        if not np.isfinite(self.start_time_s) or self.start_time_s < 0.0:
            raise ValueError("gust start time must be finite and non-negative")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("gust duration must be positive and finite")
        if self.peak_wind_ned.shape != (3,) or not np.all(np.isfinite(self.peak_wind_ned)):
            raise ValueError("peak gust wind must be a finite NED 3-vector")

    def get_wind(self, t: float, altitude: float = 100.0) -> np.ndarray:
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("gust time must be finite")
        if t < self.start_time_s or t > self.start_time_s + self.duration_s:
            return np.zeros(3, dtype=float)
        phase = (t - self.start_time_s) / self.duration_s
        scale = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        return scale * self.peak_wind_ned

    def step(self, dt: float) -> np.ndarray:
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("time step must be finite and non-negative")
        self.time += dt
        return self.get_wind(self.time)

    def reset(self) -> None:
        self.time = 0.0


class DrydenGust(WindField):
    """A seeded first-order Dryden-like turbulence approximation."""

    def __init__(self, sigma=(1.0, 1.0, 1.0), airspeed=15.0, seed=None, tau=1.0):
        super().__init__()
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.shape == ():
            self.sigma = np.full(3, float(self.sigma))
        if self.sigma.shape != (3,) or not np.all(np.isfinite(self.sigma)) or np.any(self.sigma < 0.0):
            raise ValueError("gust sigma must contain three finite non-negative values")
        self.airspeed = float(airspeed)
        self.tau = float(tau)
        if not np.isfinite(self.airspeed) or self.airspeed < 0.0:
            raise ValueError("airspeed must be finite and non-negative")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("gust time constant must be positive and finite")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.gust = np.zeros(3, dtype=float)
        self.time = 0.0

    def step(self, dt: float) -> np.ndarray:
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("time step must be finite and non-negative")
        if dt == 0.0:
            return self.gust.copy()
        correlation_time = self.tau * 15.0 / max(self.airspeed, 0.1)
        decay = np.exp(-dt / correlation_time)
        innovation_std = self.sigma * np.sqrt(1.0 - decay**2)
        self.gust = decay * self.gust + self.rng.normal(0.0, innovation_std)
        self.time += dt
        return self.gust.copy()

    def get_wind(self, t: float, altitude: float = 100.0) -> np.ndarray:
        return self.gust.copy()

    def reset(self):
        self.gust = np.zeros(3, dtype=float)
        self.time = 0.0
        self.rng = np.random.default_rng(self.seed)