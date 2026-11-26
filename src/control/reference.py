"""Reference trajectory generation for the transition phase."""

import numpy as np


class TransitionReference:
    def __init__(
        self,
        start_altitude: float,
        target_speed: float,
        transition_time: float,
        tilt_hold_time: float = 0.0,
    ):
        """
        Args:
            start_altitude: Target altitude to hold (m).
            target_speed: Target forward speed (m/s) at end of transition.
            transition_time: Seconds to rotate the thrust axis from upward to forward.
            tilt_hold_time: Seconds to hold the upward rotor orientation.
        """
        if transition_time <= 0.0:
            raise ValueError("transition_time must be positive")
        if tilt_hold_time < 0.0:
            raise ValueError("tilt_hold_time must be non-negative")

        self.start_altitude = start_altitude
        self.target_speed = target_speed
        self.transition_time = transition_time
        self.tilt_hold_time = tilt_hold_time

    def at(self, t: float) -> dict:
        """Return desired state at time t."""
        progress = min(t / self.transition_time, 1.0)
        tilt_progress = np.clip(
            (t - self.tilt_hold_time) / self.transition_time,
            0.0,
            1.0,
        )
        tilt = (np.pi / 4) * (1.0 - np.cos(np.pi * tilt_progress))

        # Speed ramps from 0 to target over the transition
        speed = progress * self.target_speed

        # Altitude held constant
        altitude = self.start_altitude

        return {"altitude": altitude, "speed": speed, "tilt": tilt, "progress": progress}
