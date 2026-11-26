"""Rigorous V-Tail aerodynamic model based on Roskam and Etkin."""
import numpy as np

class VtailModel:
    def __init__(self, area: float, arm_length: float, dihedral_deg: float = 35.0, rho: float = 1.225):
        self.area = area
        self.arm_length = arm_length
        self.dihedral = np.deg2rad(dihedral_deg)
        self.rho = rho
        
        # Aerodynamic coefficients (typical for a V-tail airfoil)
        self.CL_alpha_tail = 3.5    # Lift slope per radian
        self.CD_0_tail = 0.015      # Parasitic drag
        self.AR_tail = 4.0          # Aspect ratio of the tail
        self.oswald_efficiency = 0.8
        self.max_lift_coefficient = 1.5
        
    def compute_forces_and_moments(self, vel_air_earth: np.ndarray, att: np.ndarray, delta_left: float, delta_right: float):
        """Computes the V-tail forces and moments in the body frame."""
        
        # 1. Convert Earth Airspeed to Body Airspeed
        from src.dynamics.frames import rotation_earth_to_body
        R_e2b = rotation_earth_to_body(att[0], att[1], att[2])
        vel_body = R_e2b @ vel_air_earth
        
        u = vel_body[0]
        v = vel_body[1]
        w = vel_body[2]
        
        # 2. Calculate Airspeed and angles
        V_T = np.sqrt(u**2 + v**2 + w**2)
        if V_T < 0.5:
            return np.zeros(3), np.zeros(3)
            
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / V_T, -1.0, 1.0))
        
        # 3. Dynamic Pressure
        q_bar = 0.5 * self.rho * (V_T ** 2)
        
        # 4. Effective Angle of Attack for each V-tail surface
        # Left surface (moves symmetrically for pitch, antisymmetrically for yaw)
        alpha_left = alpha + delta_left * np.cos(self.dihedral) + beta * np.sin(self.dihedral)
        # Right surface
        alpha_right = alpha + delta_right * np.cos(self.dihedral) - beta * np.sin(self.dihedral)
        
        # 5. Lift and Drag per surface
        # Lift
        CL_left = self.CL_alpha_tail * alpha_left
        CL_right = self.CL_alpha_tail * alpha_right
        
        # Clamp for stall
        CL_left = np.clip(CL_left, -self.max_lift_coefficient, self.max_lift_coefficient)
        CL_right = np.clip(CL_right, -self.max_lift_coefficient, self.max_lift_coefficient)
        
        L_left = q_bar * (self.area / 2.0) * CL_left
        L_right = q_bar * (self.area / 2.0) * CL_right
        
        # Drag
        CD_left = self.CD_0_tail + (CL_left ** 2) / (np.pi * self.oswald_efficiency * self.AR_tail)
        CD_right = self.CD_0_tail + (CL_right ** 2) / (np.pi * self.oswald_efficiency * self.AR_tail)
        
        D_left = q_bar * (self.area / 2.0) * CD_left
        D_right = q_bar * (self.area / 2.0) * CD_right
        
        force_body = np.array([
            -(D_left + D_right),
            (L_right - L_left) * np.sin(self.dihedral),
            -(L_left + L_right) * np.cos(self.dihedral),
        ])
        # Individual panel moment arms are unknown; use the shared tail arm.
        moment_body = np.cross(np.array([-self.arm_length, 0.0, 0.0]), force_body)
        
        return force_body, moment_body
    