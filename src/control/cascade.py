import numpy as np
from src.control.pid import PIDController

class CascadeController:
    def __init__(self, mass: float, k_thrust: float, hover_omega: float):
        self.mass = mass
        self.k_thrust = k_thrust
        self.hover_omega = hover_omega
        self.alt_pid = PIDController(kp=0.1, ki=0.005, kd=0.1, limit=0.45)
        self.pitch_pid = PIDController(kp=0.6, ki=0.05, kd=0.4, limit=0.2)

    def compute(self, t: float, state, ref_altitude: float, ref_speed: float, ref_tilt: float) -> dict:
        current_alt = -state.pos[2]
        current_pitch = state.att[1]
        pitch_from_alt = self.alt_pid.compute(ref_altitude, current_alt, 0.02)
        # Allow pitch up to 25 degrees (0.45 rad)
        desired_pitch = np.clip(pitch_from_alt, -0.25, 0.45)
        vtail_cmd = -self.pitch_pid.compute(desired_pitch, current_pitch, 0.01)
        altitude_error = ref_altitude - current_alt
        total_thrust = (self.mass * 9.81) + (altitude_error * 0.5)
        omega_cmd = np.sqrt(max(total_thrust, 0.0) / self.k_thrust)
        omega_cmd = np.clip(omega_cmd, 0.0, 3000.0)
        return {'omega': omega_cmd, 'tilt': ref_tilt, 'delta_left': vtail_cmd, 'delta_right': vtail_cmd, 'desired_pitch': desired_pitch, 'current_pitch': current_pitch}
