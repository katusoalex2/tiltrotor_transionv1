"""Atmospheric wind models used by the tilt-rotor simulation."""
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


class DrydenGust(WindField):
    """A lightweight Dryden-like gust model with smooth, reproducible samples."""

    def __init__(self, sigma=(0.0, 0.0, 0.0), airspeed=15.0, seed=None, tau=1.0):
        super().__init__()
        self.sigma = np.asarray(sigma, dtype=float)
        if self.sigma.shape == ():
            self.sigma = np.full(3, float(self.sigma))
        if np.allclose(self.sigma, 0.0):
            self.sigma = np.ones(3, dtype=float)
        self.airspeed = float(airspeed)
        self.tau = float(tau)
        self.rng = np.random.default_rng(seed)
        self.gust = np.zeros(3, dtype=float)
        self.time = 0.0

    def step(self, dt: float) -> np.ndarray:
        dt = float(dt)
        if dt <= 0.0:
            return self.gust.copy()

        alpha = 0.12
        noise = self.rng.normal(0.0, self.sigma * 4.0)
        self.gust = (1.0 - alpha) * self.gust + alpha * noise
        self.time += dt
        return self.gust.copy()

    def get_wind(self, t: float, altitude: float = 100.0) -> np.ndarray:
        return self.gust.copy()

    def reset(self):
        self.gust = np.zeros(3, dtype=float)
        self.time = 0.0