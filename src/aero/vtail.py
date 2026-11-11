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
        CL_left = np.clip(CL_left, -1.5, 1.5)
        CL_right = np.clip(CL_right, -1.5, 1.5)
        
        L_left = q_bar * (self.area / 2.0) * CL_left
        L_right = q_bar * (self.area / 2.0) * CL_right
        
        # Drag
        CD_left = self.CD_0_tail + (CL_left ** 2) / (np.pi * 0.8 * self.AR_tail)
        CD_right = self.CD_0_tail + (CL_right ** 2) / (np.pi * 0.8 * self.AR_tail)
        
        D_left = q_bar * (self.area / 2.0) * CD_left
        D_right = q_bar * (self.area / 2.0) * CD_right
        
        # 6. Decompose into Body Forces (X, Y, Z)
        # The surfaces are tilted at the dihedral angle
        # Force normal to the left surface
        F_normal_left = -L_left * np.cos(self.dihedral) - D_left * np.sin(self.dihedral)
        # Force along the left surface (side)
        F_side_left = -L_left * np.sin(self.dihedral) + D_left * np.cos(self.dihedral)
        
        F_normal_right = -L_right * np.cos(self.dihedral) + D_right * np.sin(self.dihedral)
        F_side_right = L_right * np.sin(self.dihedral) + D_right * np.cos(self.dihedral)
        
        # Total tail forces in Body Frame
        F_x_tail = -(D_left + D_right) # Drag points backward
        F_y_tail = F_side_left - F_side_right # Antisymmetric (Yaw)
        F_z_tail = F_normal_left + F_normal_right # Symmetric (Pitch/Lift)
        
        force_body = np.array([F_x_tail, F_y_tail, F_z_tail])
        
        # 7. Moments (Force x Arm Length)
        # Roll moment (differential force from Y)
        M_x_tail = (F_y_tail * self.arm_length) 
        # Pitch moment (Z force x arm)
        M_y_tail = (-F_z_tail * self.arm_length) 
        # Yaw moment (Y force x arm)
        M_z_tail = (F_y_tail * self.arm_length)
        
        moment_body = np.array([M_x_tail, M_y_tail, M_z_tail])
        
        return force_body, moment_body
    