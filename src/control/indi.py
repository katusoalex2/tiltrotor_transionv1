import numpy as np

class INDIController:
    def __init__(self, B_eff: float = -50.0, pitch_gain: float = 4.0, filter_alpha: float = 0.3, limit: float = 0.2):
        self.B_eff = B_eff
        self.pitch_gain = pitch_gain
        self.filter_alpha = filter_alpha
        self.limit = limit
        self.prev_vtail = 0.0
        self.prev_pitch_rate = 0.0
        self.filtered_accel = 0.0

    def compute(self, desired_pitch: float, current_pitch: float, current_pitch_rate: float, dt: float) -> float:
        if dt <= 0: return self.prev_vtail
        raw_accel = (current_pitch_rate - self.prev_pitch_rate) / dt
        self.filtered_accel = self.filter_alpha * raw_accel + (1 - self.filter_alpha) * self.filtered_accel
        
        pitch_error = desired_pitch - current_pitch
        desired_pitch_rate = self.pitch_gain * pitch_error
        desired_rate_dot = (desired_pitch_rate - current_pitch_rate) / 0.1
        
        delta_vtail = (desired_rate_dot - self.filtered_accel) / self.B_eff
        new_vtail = self.prev_vtail + delta_vtail
        new_vtail = np.clip(new_vtail, -self.limit, self.limit)
        
        self.prev_vtail = new_vtail
        self.prev_pitch_rate = current_pitch_rate
        return new_vtail

    def reset(self):
        self.prev_vtail = 0.0
        self.prev_pitch_rate = 0.0
        self.filtered_accel = 0.0
