import numpy as np

class VtailModel:
    def __init__(self, area: float, arm_length: float, dihedral_angle: float = 0.6, rho: float = 1.225):
        self.area = area
        self.arm_length = arm_length
        self.dihedral = dihedral_angle
        self.rho = rho

    def compute_forces_and_moments(self, vel_body: np.ndarray, delta_left: float, delta_right: float):
        u = vel_body[0]
        V = np.clip(np.abs(u), 0.0, 50.0)
        q_bar = 0.5 * self.rho * (V ** 2)
        c_lift = 2.0 
        lift_left = q_bar * (self.area / 2) * c_lift * delta_left
        lift_right = q_bar * (self.area / 2) * c_lift * delta_right
        c_dihedral = np.cos(self.dihedral)
        s_dihedral = np.sin(self.dihedral)
        Fz_left = -lift_left * c_dihedral
        Fz_right = -lift_right * c_dihedral
        Fy_left = lift_left * s_dihedral
        Fy_right = -lift_right * s_dihedral
        force_body = np.array([0.0, Fy_left + Fy_right, Fz_left + Fz_right])
        My = (Fz_left + Fz_right) * self.arm_length
        Mz = (Fy_left + Fy_right) * self.arm_length
        return force_body, np.array([0.0, My, Mz])
