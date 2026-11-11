"""Aerodynamic forces with Wind Triangle and Stall modeling."""
import numpy as np
from src.aero.vtail import VtailModel

class AeroModel:
    def __init__(self, wing_area: float, vtail: VtailModel, rho: float = 1.225):
        self.wing_area = wing_area
        self.vtail = vtail
        self.rho = rho
        self.CL_alpha = 4.0
        self.CL_0 = 0.1
        self.CD_0 = 0.02
        self.AR = 6.0
        self.stall_angle = np.deg2rad(15.0)

    def compute_forces(self, vel_earth: np.ndarray, wind_earth: np.ndarray, att: np.ndarray, total_thrust: float, tilt_angle: float):
        vel_air_earth = vel_earth - wind_earth
        from src.dynamics.frames import rotation_earth_to_body
        R_e2b = rotation_earth_to_body(att[0], att[1], att[2])
        vel_body = R_e2b @ vel_air_earth
        
        u = vel_body[0]
        w = vel_body[2]
        
        V_T = np.sqrt(u**2 + w**2)
        if V_T < 0.5:
            return 0.0, 0.0, V_T, 0.0
            
        alpha = np.arctan2(w, u)
        
        if abs(alpha) < self.stall_angle:
            CL = self.CL_0 + self.CL_alpha * alpha
        else:
            CL = self.CL_0 + self.CL_alpha * np.sign(alpha) * self.stall_angle
            CL *= np.cos(alpha - np.sign(alpha) * self.stall_angle) ** 2
            
        CD = self.CD_0 + (CL**2) / (np.pi * 0.8 * self.AR)
        q_bar = 0.5 * self.rho * (V_T ** 2)
        
        lift = q_bar * self.wing_area * CL
        drag = q_bar * self.wing_area * CD
        return lift, drag, V_T, alpha
        
    def compute_tail_forces(self, vel_earth: np.ndarray, wind_earth: np.ndarray, att: np.ndarray, delta_left: float, delta_right: float):
        vel_air_earth = vel_earth - wind_earth
        return self.vtail.compute_forces_and_moments(vel_air_earth, att, delta_left, delta_right)
