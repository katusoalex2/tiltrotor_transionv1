"""Time-parameterized reference trajectory generated from an optimized corridor."""

import numpy as np
import os


class ReferenceTrajectory:
    """Generates reference values (tilt, speed, altitude) from a corridor CSV.

    The corridor CSV is expected to have columns:
        time_s, tilt_deg, altitude_m

    Speed is derived from tilt as:
        speed = (tilt / tilt_max) * speed_max
    
    This matches the assumption used in the corridor optimization that
    speed ramps up as rotors tilt forward.
    """

    def __init__(self, csv_path: str, speed_max: float = 12.0):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Corridor file not found: {csv_path}")

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.t = data[:, 0]
        self.tilt_deg = data[:, 1]
        self.altitude_m = data[:, 2]

        self.tilt_rad = np.deg2rad(self.tilt_deg)
        self.tilt_max = self.tilt_rad[-1]
        self.speed_max = speed_max
        self.t_start = self.t[0]
        self.t_end = self.t[-1]

    def at(self, t: float) -> dict:
        """Return the reference state at time t.

        If t is outside the corridor time range, clamps to endpoints.
        """
        t_clamped = np.clip(t, self.t_start, self.t_end)

        tilt = float(np.interp(t_clamped, self.t, self.tilt_rad))
        alt = float(np.interp(t_clamped, self.t, self.altitude_m))

        # Speed ramps proportionally with tilt (0 at hover, speed_max at cruise)
        if self.tilt_max > 1e-6:
            progress = tilt / self.tilt_max
        else:
            progress = 0.0
        speed = progress * self.speed_max

        return {
            "tilt": tilt,
            "altitude": alt,
            "speed": speed,
            "progress": progress,
        }

    def is_complete(self, t: float) -> bool:
        """Has the reference trajectory finished?"""
        return t >= self.t_end


class LinearReference:
    """Simpler linear reference for comparison (backward-compatible)."""

    def __init__(self, start_altitude: float, target_speed: float, transition_time: float):
        self.start_altitude = start_altitude
        self.target_speed = target_speed
        self.transition_time = transition_time

    def at(self, t: float) -> dict:
        progress = min(t / self.transition_time, 1.0)
        return {
            "tilt": progress * (np.pi / 2),
            "altitude": self.start_altitude,
            "speed": progress * self.target_speed,
            "progress": progress,
        }
