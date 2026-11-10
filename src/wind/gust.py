"""Wind gust models: Dryden turbulence and constant gust."""

import numpy as np


class WindField:
    """Base class for wind fields. Subclasses must implement step()."""

    def __init__(self):
        self.time = 0.0

    def step(self, dt: float, pos: np.ndarray = None) -> np.ndarray:
        """Advance the wind field by dt seconds and return wind velocity (3,).

        Args:
            dt: time step (s)
            pos: aircraft position (optional, used for spatial gusts)

        Returns:
            wind velocity vector in world frame [u, v, w] (m/s)
        """
        raise NotImplementedError

    def reset(self):
        """Reset internal state (used between simulation runs)."""
        self.time = 0.0


class ConstantGust(WindField):
    """Constant wind (useful for basic tests and troubleshooting)."""

    def __init__(self, u: float = 0.0, v: float = 0.0, w: float = 0.0):
        super().__init__()
        self.vector = np.array([u, v, w], dtype=float)

    def step(self, dt: float, pos: np.ndarray = None) -> np.ndarray:
        self.time += dt
        return self.vector.copy()


class DrydenGust(WindField):
    """Dryden turbulence model with first-order filters.

    The Dryden model approximates atmospheric turbulence as filtered white noise.
    For each axis (u, v, w), we apply:

        u_gust[k+1] = u_gust[k] + (sigma_u * sqrt(2*dt/tau) * noise - u_gust[k]*dt/tau)

    where tau = L / V is the turbulence time constant.

    For simplicity we use zero-mean white Gaussian noise as the driver.

    Args:
        sigma: (sigma_u, sigma_v, sigma_w) turbulence intensity, m/s
        length_scale: (Lu, Lv, Lw) turbulence length scales, m
        airspeed: nominal forward speed V (m/s), used for time constant
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        sigma: tuple = (2.0, 2.0, 2.0),
        length_scale: tuple = (200.0, 200.0, 200.0),
        airspeed: float = 15.0,
        seed: int = 42,
    ):
        super().__init__()
        self.sigma = np.array(sigma, dtype=float)
        self.length_scale = np.array(length_scale, dtype=float)
        self.airspeed = airspeed
        self.rng = np.random.default_rng(seed)

        # Time constants tau = L / V (clamped to avoid division issues)
        self.tau = np.clip(self.length_scale / max(self.airspeed, 1.0), 0.5, 10.0)

        # State: filtered gust velocity
        self.gust = np.zeros(3)

    def step(self, dt: float, pos: np.ndarray = None) -> np.ndarray:
        """Advance the Dryden filter by dt seconds."""
        # White noise, zero mean, unit variance
        noise = self.rng.standard_normal(3)

        # First-order filter: dy/dt = -y/tau + sigma * sqrt(2*dt/tau) * noise
        alpha = np.exp(-dt / self.tau)  # decay factor
        noise_scale = self.sigma * np.sqrt(2 * dt / self.tau) * noise
        self.gust = alpha * self.gust + noise_scale

        self.time += dt
        return self.gust.copy()

    def reset(self):
        super().reset()
        self.gust = np.zeros(3)