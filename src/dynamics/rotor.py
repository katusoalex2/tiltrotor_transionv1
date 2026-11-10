"""Rotor and motor model."""

import numpy as np


class RotorModel:
    """Models a single tilt-rotor.
    
    Uses: T = k * omega^2 
    """
    
    def __init__(self, k_thrust: float, max_omega: float):
        """
        Args:
            k_thrust: Thrust coefficient (k). 
                      Calculated as: C_s * rho * D^4
            max_omega: Maximum angular velocity in rad/s.
        """
        self.k_thrust = k_thrust
        self.max_omega = max_omega
        
    def compute_thrust(self, omega: float) -> float:
        """Compute the thrust magnitude in Newtons for a given RPM (rad/s)."""
        # Clamp omega to physical limits
        omega = np.clip(omega, 0.0, self.max_omega)
        return self.k_thrust * (omega ** 2)