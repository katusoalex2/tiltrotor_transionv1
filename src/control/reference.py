"""Reference trajectory generation for the transition phase."""

import numpy as np


class TransitionReference:
    def __init__(self, start_altitude: float, target_speed: float, transition_time: float):
        """
        Args:
            start_altitude: Target altitude to hold (m).
            target_speed: Target forward speed (m/s) at end of transition.
            transition_time: Time (s) to complete the tilt 0°→90°.
        """
        self.start_altitude = start_altitude
        self.target_speed = target_speed
        self.transition_time = transition_time

    def at(self, t: float) -> dict:
        """Return desired state at time t."""
        # Tilt ramps linearly from 0 to 90 degrees
        progress = min(t / self.transition_time, 1.0)
        tilt = progress * (np.pi / 2)

        # Speed ramps from 0 to target over the transition
        speed = progress * self.target_speed

        # Altitude held constant
        altitude = self.start_altitude

        return {"altitude": altitude, "speed": speed, "tilt": tilt, "progress": progress}