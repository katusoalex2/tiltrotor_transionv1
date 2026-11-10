"""Gain Scheduler: Interpolates PID gains based on airspeed."""

import numpy as np

class GainScheduler:
    def __init__(self):
        # Define the gains at the two extremes
        # Hover: High gains because V-tail is ineffective without wind
        self.hover_gains = {"kp": 1.2, "ki": 0.1, "kd": 0.3}
        
        # Cruise: Low gains because V-tail is highly effective at 20 m/s
        self.cruise_gains = {"kp": 0.2, "ki": 0.01, "kd": 0.05}
        
        # Speed at which we fully switch to cruise gains (m/s)
        self.cruise_speed = 20.0

    def get_gains(self, current_speed: float) -> dict:
        """Returns interpolated gains based on current airspeed."""
        # 0.0 at hover, 1.0 at cruise speed
        blend = np.clip(abs(current_speed) / self.cruise_speed, 0.0, 1.0)
        
        kp = (1 - blend) * self.hover_gains["kp"] + blend * self.cruise_gains["kp"]
        ki = (1 - blend) * self.hover_gains["ki"] + blend * self.cruise_gains["ki"]
        kd = (1 - blend) * self.hover_gains["kd"] + blend * self.cruise_gains["kd"]
        
        return {"kp": kp, "ki": ki, "kd": kd}