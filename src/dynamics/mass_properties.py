"""Rigorous Mass Properties, Inertia Tensor, and Stability."""
import numpy as np
import yaml

class MassProperties:
    def __init__(self, config_path="configs/blue_uam.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            
        # Component Masses
        self.mass_fuselage = cfg.get('mass', 4.0) * 0.6
        self.mass_wing = cfg.get('mass', 4.0) * 0.2
        self.mass_vtail = cfg.get('mass', 4.0) * 0.1
        self.mass_motor = cfg.get('mass', 4.0) * 0.025
        self.total_mass = self.mass_fuselage + self.mass_wing + self.mass_vtail + (4 * self.mass_motor)
        
        # Geometry
        self.wingspan = cfg.get('wingspan', 2.0)
        self.root_chord = cfg.get('root_chord', 0.4)
        self.tip_chord = cfg.get('tip_chord', 0.2)
        self.sweep_deg = cfg.get('sweep', 0.0)
        self.le_root_x = 0.1
        self.vtail_area = cfg.get('vtail_area', 0.1)
        self.vtail_arm = cfg.get('vtail_arm_length', 0.5)
        
        self.motor_positions = cfg.get('motor_positions', [[0.3, 0.4, 0.0, True], [0.3, -0.4, 0.0, True], [-0.3, 0.4, 0.0, False], [-0.3, -0.4, 0.0, False]])
        
        # Aerodynamic Centers
        self.mac, self.ac_wing_x = self._calculate_mac_and_ac()
        self.ac_z = 0.0
        self.neutral_point_x = self._calculate_neutral_point()
        
        # Initial Inertia
        self.inertia_tensor = self.calculate_inertia(0.0)

    def _calculate_mac_and_ac(self):
        taper = self.tip_chord / self.root_chord
        mac = (2/3) * self.root_chord * (1 + taper + taper**2) / (1 + taper)
        sweep_rad = np.deg2rad(self.sweep_deg)
        semispan = self.wingspan / 2.0
        x_mac = self.le_root_x + (semispan / 3.0) * ((1 + 2*taper) / (1 + taper)) * np.tan(sweep_rad)
        ac_x = x_mac + 0.25 * mac
        return mac, ac_x

    def _calculate_neutral_point(self):
        wing_area = self.wingspan * ((self.root_chord + self.tip_chord) / 2.0)
        if wing_area == 0: return 0.0
        area_ratio = self.vtail_area / wing_area
        x_ac_tail = -self.vtail_arm
        downwash_factor = 0.65
        CL_ratio = 3.5 / 4.0
        X_np = self.ac_wing_x + (CL_ratio * downwash_factor * area_ratio * (x_ac_tail - 0.0))
        return X_np

    def get_static_margin(self) -> float:
        return self.neutral_point_x - 0.0

    def get_static_margin_pct(self) -> float:
        sm_meters = self.get_static_margin()
        if self.mac > 0:
            return (sm_meters / self.mac) * 100.0
        return 0.0

    def calculate_inertia(self, tilt_angle_deg):
        I = np.zeros((3, 3))
        L_fus = 1.6
        I += (1/12) * self.mass_fuselage * (L_fus**2) * np.diag([0.01, 1.0, 1.0])
        b = self.wingspan
        c = self.root_chord
        I += (1/12) * self.mass_wing * np.diag([b**2, c**2, b**2 + c**2])
        I += self.mass_vtail * (self.vtail_arm**2) * np.diag([1.0, 0.1, 1.0])
        
        tilt_rad = np.deg2rad(tilt_angle_deg)
        pivot_x = 0.1
        for pos in self.motor_positions:
            pos_x, pos_y, pos_z, tilts = pos
            if tilts:
                rel_x = pos_x - pivot_x
                rel_z = pos_z
                new_rel_x = rel_x * np.cos(tilt_rad) + rel_z * np.sin(tilt_rad)
                new_rel_z = -rel_x * np.sin(tilt_rad) + rel_z * np.cos(tilt_rad)
                actual_pos = np.array([pivot_x + new_rel_x, pos_y, new_rel_z])
            else:
                actual_pos = np.array([pos_x, pos_y, pos_z])
            r_sq = np.dot(actual_pos, actual_pos)
            I += self.mass_motor * (r_sq * np.eye(3) - np.outer(actual_pos, actual_pos))
            
        if self.sweep_deg != 0:
            I[0, 2] = -0.05
            I[2, 0] = -0.05
        return I
