"""V-Tail aerodynamic model"""

import numpy as np

class VtailModel:
    def __init__(self, area: float, arm_length: float, dihedral_angle: float = 0.6, rho: float = 1.225):
        """
        Args:
            area: Total V-tail surface area (m^2).
            arm_length: Distance from CG to V-tail (m).
            dihedral_angle: Angle of the V (radians). Default ~35 degrees.
            rho: Air density.
        """
        self.area = area
        self.arm_length = arm_length
        self.dihedral = dihedral_angle
        self.rho = rho

    def compute_forces_and_moments(self, vel_body: np.ndarray, delta_left: float, delta_right: float) -> tuple[np.ndarray, np.ndarray]:
        """Computes forces and moments from the V-tail.
        
        Returns:
            force_body: [Fx, Fy, Fz]
            moment_body: [Mx, My, Mz]
        """
        u = vel_body[0]
        V = np.clip(np.abs(u), 0.0, 50.0) # Forward airspeed
        q_bar = 0.5 * self.rho * (V ** 2)
        
        # Mixing logic: 
        # Pitch command (symmetric): (left + right)
        # Yaw command (antisymmetric): (left - right)
        c_lift = 2.0 # Approximate lift slope
        
        # Lift produced by each surface
        lift_left = q_bar * (self.area / 2) * c_lift * delta_left
        lift_right = q_bar * (self.area / 2) * c_lift * delta_right

        # Resolve into Body Axes based on V-tail dihedral angle
        c_dihedral = np.cos(self.dihedral)
        s_dihedral = np.sin(self.dihedral)

        # Vertical forces (Pitch) and Lateral forces (Yaw)
        Fz_left = -lift_left * c_dihedral
        Fz_right = -lift_right * c_dihedral
        Fy_left = lift_left * s_dihedral
        Fy_right = -lift_right * s_dihedral

        force_body = np.array([0.0, Fy_left + Fy_right, Fz_left + Fz_right])

        # Moments (Force x Arm Length)
        # Pitch is created by the vertical force component
        My = (Fz_left + Fz_right) * self.arm_length
        # Yaw is created by the lateral force component
        Mz = (Fy_left + Fy_right) * self.arm_length
        # Roll (Mx) is essentially zero if symmetric, but we keep it for completeness
        Mx = 0.0 

        moment_body = np.array([Mx, My, Mz])
        return force_body, moment_body
    