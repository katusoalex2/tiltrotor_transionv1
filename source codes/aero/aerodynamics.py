"""Aerodynamic forces and moments with Wind Triangle and Rotor Wash."""
import numpy as np
from src.aero.vtail import VtailModel

class AeroModel:
    def __init__(self, wing_area: float, vtail: VtailModel, rho: float = 1.225, num_rotors: int = 4, prop_diameter_m: float = 0.3):
        self.wing_area = wing_area
        self.vtail = vtail
        self.rho = rho
        
        self.CL_alpha = 4.0
        self.CL_0 = 0.1
        self.CD_0 = 0.02
        self.AR = 6.0
        self.stall_angle = np.deg2rad(15.0)
        
        # Pitch Moment Coefficients
        self.Cm_0 = -0.05       # Zero-lift pitch moment
        self.Cm_alpha = -0.5    # Pitch moment slope (negative = stable)
        self.chord = 0.35       # Mean Aerodynamic Chord (m)
        self.x_ac = -0.15       # AC offset from CG
        
        self.num_rotors = num_rotors
        self.R_rotor = prop_diameter_m / 2.0
        self.A_rotor = np.pi * (self.R_rotor ** 2)

    def compute_forces_and_moments(self, vel_earth: np.ndarray, wind_earth: np.ndarray, att: np.ndarray, total_thrust: float, tilt_angle: float):
        vel_air_earth = vel_earth - wind_earth
        from src.dynamics.frames import rotation_earth_to_body
        R_e2b = rotation_earth_to_body(att[0], att[1], att[2])
        vel_body = R_e2b @ vel_air_earth
        
        u = vel_body[0]; v = vel_body[1]; w = vel_body[2]
        V_T = np.sqrt(u**2 + v**2 + w**2)
        
        if V_T < 0.5:
            return np.zeros(3), np.zeros(3), V_T, 0.0, 0.0
            
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / V_T, -1.0, 1.0))
        
        thrust_per_rotor = total_thrust / self.num_rotors
        v_i = np.sqrt(max(thrust_per_rotor, 0.0) / (2 * self.rho * self.A_rotor))
        wash_u = v_i * np.sin(tilt_angle)
        wash_w = v_i * np.cos(tilt_angle)
        
        u_eff = u + wash_u
        w_eff = w + wash_w
        V_T_eff = np.sqrt(u_eff**2 + v**2 + w_eff**2)
        alpha_eff = np.arctan2(w_eff, u_eff)
        
        if abs(alpha_eff) < self.stall_angle:
            CL = self.CL_0 + self.CL_alpha * alpha_eff
        else:
            CL = self.CL_0 + self.CL_alpha * np.sign(alpha_eff) * self.stall_angle
            CL *= np.cos(alpha_eff - np.sign(alpha_eff) * self.stall_angle) ** 2
        CD = self.CD_0 + (CL**2) / (np.pi * 0.8 * self.AR)

        q_bar = 0.5 * self.rho * (V_T_eff ** 2)
        lift = q_bar * self.wing_area * CL
        drag = q_bar * self.wing_area * CD
        
        force_body = np.array([-drag, 0.0, -lift])
        
        r_ac = np.array([self.x_ac, 0.0, 0.0])
        moment_cross = np.cross(r_ac, force_body)
        
        M_wing_y = q_bar * self.wing_area * self.chord * (self.Cm_0 + self.Cm_alpha * alpha_eff)
        moment_body = moment_cross + np.array([0.0, M_wing_y, 0.0])
        
        return force_body, moment_body, V_T, alpha, beta

    def compute_tail_forces(self, vel_earth: np.ndarray, wind_earth: np.ndarray, att: np.ndarray, delta_left: float, delta_right: float):
        vel_air_earth = vel_earth - wind_earth
        return self.vtail.compute_forces_and_moments(vel_air_earth, att, delta_left, delta_right)
