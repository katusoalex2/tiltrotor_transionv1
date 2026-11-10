"""Combined Aerodynamic forces for the UAV."""

import numpy as np
from src.aero.vtail import VtailModel

class AeroModel:
    def __init__(self, wing_area: float, vtail: VtailModel, rho: float = 1.225):
        self.wing_area = wing_area
        self.vtail = vtail
        self.rho = rho

    def compute_wing_forces(self, vel_body: np.ndarray) -> tuple[float, float]:
        """Computes Lift (-Z) and Drag (-X) for the main wing."""
        u = vel_body[0]
        w = vel_body[2]
        
        V = np.sqrt(u**2 + w**2)
        V = np.clip(V, 0.0, 50.0) 
        
        if V < 0.1: 
            return 0.0, 0.0

        alpha = np.arctan2(-w, u)
        # Stall protection: clamp alpha to +/- 15 degrees (0.26 rad)
        alpha_clamped = np.clip(alpha, -0.26, 0.26)

        CL = 0.1 + 4.0 * alpha_clamped
        CD = 0.02 + 0.1 * (CL**2)

        q_bar = 0.5 * self.rho * (V ** 2)
        
        lift = q_bar * self.wing_area * CL
        drag = q_bar * self.wing_area * CD
        
        return lift, drag

    def compute_tail_forces(self, vel_body: np.ndarray, delta_left: float, delta_right: float):
        return self.vtail.compute_forces_and_moments(vel_body, delta_left, delta_right)