"""Rotor model with Ground Effect."""
import numpy as np

class RotorModel:
    def __init__(self, k_thrust: float, max_omega: float, prop_mass_kg: float = 0.010, prop_radius_m: float = 0.15):
        self.k_thrust = k_thrust
        self.max_omega = max_omega
        self.prop_mass = prop_mass_kg
        self.prop_radius = prop_radius_m
        self.I_prop = (1/12) * self.prop_mass * (2 * self.prop_radius)**2

    def compute_thrust(self, omega: float, height_above_ground: float = 100.0) -> float:
        """Computes thrust including ground effect (Cheeseman-Bennett)."""
        omega = np.clip(omega, 0.0, self.max_omega)
        thrust_oge = self.k_thrust * (omega ** 2)
        
        # Cheeseman-Bennett Ground Effect
        R = self.prop_radius
        z = max(height_above_ground, R * 0.1)  # Prevent division by zero
        
        # If above 2R, ground effect is negligible
        if z > 2 * R:
            return thrust_oge
        
        ground_effect_ratio = 1.0 / (1.0 - (R / (4.0 * z))**2)
        thrust_ige = thrust_oge * ground_effect_ratio
        return thrust_ige

    def get_angular_momentum(self, omega: float) -> float:
        return self.I_prop * omega
        
    def get_ground_effect_ratio(self, height_above_ground: float) -> float:
        """Returns the T_IGE / T_OGE ratio for display purposes."""
        R = self.prop_radius
        z = max(height_above_ground, R * 0.1)
        if z > 2 * R:
            return 1.0
        return 1.0 / (1.0 - (R / (4.0 * z))**2)
