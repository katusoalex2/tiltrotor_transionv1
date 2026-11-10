"""Mass Properties, CG, and Aerodynamic Center calculations (NED Frame)."""

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
        
        self.mac_chord = cfg.get('chord', 0.4)
        self.ac_x = 0.1 + (0.25 * self.mac_chord) 
        self.ac_z = 0.0 
        
        self.cg = self._calculate_cg()
        self.inertia_tensor = self._calculate_inertia()

    def _calculate_cg(self) -> np.ndarray:
        body_moment = np.zeros(3)
        motor_moments = np.zeros(3)
        for pos in self.motor_positions:
            m_pos = np.array([pos[0], pos[1], pos[2]])
            motor_moments += self.mass_motor * m_pos
        total_moment = body_moment + motor_moments
        return total_moment / self.total_mass

    def _calculate_inertia(self) -> np.ndarray:
        I = np.zeros((3, 3))
        I += np.diag([0.15, 0.25, 0.40]) 
        for pos in self.motor_positions:
            m_pos = np.array([pos[0], pos[1], pos[2]])
            r_sq = np.dot(m_pos, m_pos)
            I_motor = self.mass_motor * (r_sq * np.eye(3) - np.outer(m_pos, m_pos))
            I += I_motor
        return I

    def get_static_margin(self) -> float:
        return self.ac_x - self.cg[0]

    def print_summary(self):
        print("=" * 50)
        print("AIRCRAFT MASS PROPERTIES (NED FRAME)")
        print("=" * 50)
        print(f"Total Mass: {self.total_mass:.2f} kg")
        print(f"CG Position [X, Y, Z]: {self.cg.round(3)} m")
        print(f"AC Position [X, Y, Z]: [{self.ac_x:.3f}, 0.0, {self.ac_z:.3f}] m")
        print(f"Static Margin: {self.get_static_margin():.3f} m")
        if self.get_static_margin() > 0:
            print("STABILITY: STABLE (AC is behind CG)")
        else:
            print("STABILITY: UNSTABLE (AC is ahead of CG)")
        print()
        print("INERTIA TENSOR (kg?m?):")
        print(np.round(self.inertia_tensor, 4))
        print("=" * 50)

if __name__ == "__main__":
    props = MassProperties()
    props.print_summary()
