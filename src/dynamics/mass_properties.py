"""Mass Properties, Aerodynamic Center, and Neutral Point (NED)."""
import numpy as np
import yaml

class MassProperties:
    def __init__(self, config_path="configs/blue_uam.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        self.mass_body = cfg.get('mass', 5.0)
        self.motor_positions = cfg.get('motor_positions', [[0.3, 0.4, 0.0], [0.3, -0.4, 0.0], [-0.3, 0.4, 0.0], [-0.3, -0.4, 0.0]])
        self.mass_motor = 0.15
        self.total_mass = self.mass_body + (4 * self.mass_motor)
        
        # Wing Geometry
        self.wingspan = cfg.get('wingspan', 2.0)
        self.root_chord = cfg.get('root_chord', 0.4)
        self.tip_chord = cfg.get('tip_chord', 0.2)
        self.sweep_deg = cfg.get('sweep', 0.0)
        self.le_root_x = cfg.get('le_root_x', 0.1)
        
        # V-Tail Geometry
        self.vtail_area = cfg.get('vtail_area', 0.1)
        self.vtail_arm = cfg.get('vtail_arm_length', 0.5)
        
        # Aerodynamic Coefficients (derivatives with respect to alpha)
        self.CL_alpha_wing = cfg.get('wing_lift_slope', 4.0)  # per radian
        self.CL_alpha_tail = 3.5  # Typical for V-tail
        self.downwash_gradient = 0.35  # Typical d(epsilon)/d(alpha)
        
        # Computations
        self.cg = self._calculate_cg()
        self.inertia_tensor = self._calculate_inertia()
        self.mac, self.ac_wing_x = self._calculate_mac_and_ac()
        self.ac_z = 0.0
        self.neutral_point_x = self._calculate_neutral_point()

    def _calculate_cg(self) -> np.ndarray:
        body_moment = np.zeros(3)
        motor_moments = np.zeros(3)
        for pos in self.motor_positions:
            m_pos = np.array([pos[0], pos[1], pos[2]])
            motor_moments += self.mass_motor * m_pos
        return (body_moment + motor_moments) / self.total_mass

    def _calculate_inertia(self) -> np.ndarray:
        I = np.zeros((3, 3))
        I += np.diag([0.15, 0.25, 0.40]) 
        for pos in self.motor_positions:
            m_pos = np.array([pos[0], pos[1], pos[2]])
            r_sq = np.dot(m_pos, m_pos)
            I += self.mass_motor * (r_sq * np.eye(3) - np.outer(m_pos, m_pos))
        return I

    def _calculate_mac_and_ac(self):
        """Calculates Mean Aerodynamic Chord and wing Aerodynamic Center."""
        taper_ratio = self.tip_chord / self.root_chord
        mac = (2/3) * self.root_chord * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
        
        sweep_rad = np.deg2rad(self.sweep_deg)
        semispan = self.wingspan / 2.0
        x_mac = self.le_root_x + (semispan / 3.0) * ((1 + 2*taper_ratio) / (1 + taper_ratio)) * np.tan(sweep_rad)
        
        # Wing AC is at 25% of MAC
        ac_wing_x = x_mac + 0.25 * mac
        return mac, ac_wing_x

    def _calculate_neutral_point(self):
        """Calculates the Neutral Point (X_np) using wing and tail contributions."""
        # Calculate wing area (approximate trapezoid)
        wing_area = self.wingspan * ((self.root_chord + self.tip_chord) / 2.0)
        
        # Tail Area Ratio (S_t / S_w)
        area_ratio = self.vtail_area / wing_area
        
        # Tail AC position (Approximate X of V-tail MAC)
        # Assuming V-tail is at the back of the aircraft
        x_ac_tail = self.cg[0] - self.vtail_arm
        
        # Downwash effect factor
        downwash_factor = 1.0 - self.downwash_gradient
        
        # Lift curve slope ratio
        CL_ratio = self.CL_alpha_tail / self.CL_alpha_wing
        
        # Distance between Wing AC and CG
        wing_ac_to_cg = self.ac_wing_x - self.cg[0]
        
        # Neutral Point Equation
        X_np = self.ac_wing_x + (CL_ratio * downwash_factor * area_ratio * (x_ac_tail - self.cg[0]))
        return X_np

    def get_static_margin(self) -> float:
        """Distance between Neutral Point and CG (must be positive for stability)."""
        return self.neutral_point_x - self.cg[0]
    
    def get_static_margin_pct(self) -> float:
        """Static Margin as a percentage of MAC. 
        A healthy aircraft usually has 5% to 15%.
        """
        sm_meters = self.get_static_margin()
        if self.mac > 0:
            return (sm_meters / self.mac) * 100.0
        return 0.0
    
