"""Time-parameterized reference trajectory generated from an optimized corridor."""

import os
from pathlib import Path

import numpy as np

from src.dynamics.nacelle import deflection_rad_from_tilt_deg, tilt_deg_from_deflection_rad


DEFAULT_CORRIDOR_PATH = Path(__file__).resolve().parents[2] / "UAM Configurations" / "robust_corridor_elevation.csv"


class ReferenceTrajectory:
    """Generates reference values (tilt, speed, altitude) from a corridor CSV.

    The corridor CSV is expected to have columns:
        time_s, nacelle_angle_deg, altitude_m

    The legacy tilt_deg column denotes deflection from vertical. The
    controller's tilt command is also a deflection from vertical in radians.
    
    This matches the assumption used in the corridor optimization that
    speed ramps up as rotors tilt forward.
    """

    def __init__(self, csv_path: str | Path, speed_max: float = 12.0):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Corridor file not found: {csv_path}")

        with open(csv_path, encoding="utf-8-sig") as stream:
            header = stream.readline().strip().split(",")
            data = np.loadtxt(stream, delimiter=",", ndmin=2)
        self.t = data[:, 0]
        self.altitude_m = data[:, 2]

        if header[1] == "nacelle_angle_deg":
            self.tilt_rad = deflection_rad_from_tilt_deg(data[:, 1])
        elif header[1] == "tilt_deg":
            self.tilt_rad = np.deg2rad(data[:, 1])
        else:
            raise ValueError("Corridor CSV needs a nacelle_angle_deg or legacy tilt_deg column")
        self.tilt_deg = tilt_deg_from_deflection_rad(self.tilt_rad)
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
