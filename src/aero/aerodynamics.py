"""Aerodynamic forces with Wind Triangle, Stall, and Rotor Wash."""
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
        
        # Rotor Wash Parameters (BEMT)
        self.num_rotors = num_rotors
        self.R_rotor = prop_diameter_m / 2.0
        self.A_rotor = np.pi * (self.R_rotor ** 2)

    def compute_forces(self, vel_earth: np.ndarray, wind_earth: np.ndarray, att: np.ndarray, total_thrust: float, tilt_angle: float):
        """Computes Lift, Drag, and True Airspeed incorporating Rotor Wash."""
        
        # 1. Wind Triangle
        vel_air_earth = vel_earth - wind_earth
        from src.dynamics.frames import rotation_earth_to_body
        R_e2b = rotation_earth_to_body(att[0], att[1], att[2])
        vel_body = R_e2b @ vel_air_earth
        
        u = vel_body[0]  # Forward speed (body)
        w = vel_body[2]  # Downward speed (body)
        
        # 2. BEMT Rotor Wash Calculation
        thrust_per_rotor = total_thrust / self.num_rotors
        # Induced velocity in hover (momentum theory)
        v_i = np.sqrt(max(thrust_per_rotor, 0.0) / (2 * self.rho * self.A_rotor))
        
        # Slipstream components in body frame (NED)
        # The rotor wash is aligned with the rotor disk normal (tilted by tilt_angle)
        wash_u = v_i * np.sin(tilt_angle)   # Forward wash
        wash_w = v_i * np.cos(tilt_angle)   # Downward wash (NED: +w is down)
        
        # 3. Effective Airspeed at Wing (Free stream + Rotor Wash)
        u_eff = u + wash_u
        w_eff = w + wash_w
        
        V_T_eff = np.sqrt(u_eff**2 + w_eff**2)
        
        if V_T_eff < 0.5:
            return 0.0, 0.0, 0.0, 0.0
            
        alpha = np.arctan2(w_eff, u_eff)
        
        # 4. Lift and Drag Coefficients (with Stall)
        if abs(alpha) < self.stall_angle:
            CL = self.CL_0 + self.CL_alpha * alpha
        else:
            CL = self.CL_0 + self.CL_alpha * np.sign(alpha) * self.stall_angle
            CL *= np.cos(alpha - np.sign(alpha) * self.stall_angle) ** 2
            
        CD = self.CD_0 + (CL**2) / (np.pi * 0.8 * self.AR)

        # 5. Dynamic Pressure
        q_bar = 0.5 * self.rho * (V_T_eff ** 2)
        
        # 6. Calculate Forces
        lift = q_bar * self.wing_area * CL
        drag = q_bar * self.wing_area * CD
        
        return lift, drag, V_T_eff, alpha
        
    def compute_tail_forces(self, vel_earth: np.ndarray, wind_earth: np.ndarray, att: np.ndarray, delta_left: float, delta_right: float):
        vel_air_earth = vel_earth - wind_earth
        return self.vtail.compute_forces_and_moments(vel_air_earth, att, delta_left, delta_right)
