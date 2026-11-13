"""Rotor model with a stable ground-effect correction."""
import numpy as np


class RotorModel:
    def __init__(self, k_thrust: float, max_omega: float, prop_mass_kg: float = 0.010, prop_radius_m: float = 0.15):
        self.k_thrust = k_thrust
        self.max_omega = max_omega
        self.prop_mass = prop_mass_kg
        self.prop_radius = prop_radius_m
        self.I_prop = (1 / 12) * self.prop_mass * (2 * self.prop_radius) ** 2

    def compute_thrust(self, omega: float, height_above_ground: float = 100.0) -> float:
        """Compute thrust using the square-law model with a bounded ground effect correction."""
        omega = np.clip(omega, 0.0, self.max_omega)
        thrust_oge = self.k_thrust * (omega ** 2)

        R = self.prop_radius
        z = max(height_above_ground, R * 0.1)
        if z > 2.0 * R:
            return thrust_oge

        ratio = 1.0 - (R / (4.0 * z)) ** 2
        if ratio <= 0.0 or not np.isfinite(ratio):
            return thrust_oge
        return thrust_oge / ratio

    def get_angular_momentum(self, omega: float) -> float:
        return self.I_prop * omega

    def get_ground_effect_ratio(self, height_above_ground: float) -> float:
        """Return the T_IGE / T_OGE ratio for display purposes."""
        R = self.prop_radius
        z = max(height_above_ground, R * 0.1)
        if z > 2.0 * R:
            return 1.0
        ratio = 1.0 - (R / (4.0 * z)) ** 2
        if ratio <= 0.0 or not np.isfinite(ratio):
            return 1.0
        return 1.0 / ratio