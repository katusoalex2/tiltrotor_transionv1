"""Guidance logic for commanding altitude, speed, and heading tracking."""

import numpy as np


class GuidanceController:
    """Simple flight-path guidance law for the tilt-rotor transition model."""

    def __init__(self, altitude_kp: float = 0.8, speed_kp: float = 0.6, heading_kp: float = 0.4,
                 max_pitch_rad: float = np.deg2rad(25.0)):
        self.altitude_kp = float(altitude_kp)
        self.speed_kp = float(speed_kp)
        self.heading_kp = float(heading_kp)
        self.max_pitch_rad = float(max_pitch_rad)

    def compute(self, state, target_altitude: float, target_speed: float, target_heading: float = 0.0,
                dt: float = 0.01):
        current_altitude = float(-state.pos[2])
        current_speed = float(state.vel[0])
        current_heading = float(state.att[2])

        altitude_error = target_altitude - current_altitude
        desired_pitch = np.clip(self.altitude_kp * altitude_error, -self.max_pitch_rad, self.max_pitch_rad)

        speed_error = target_speed - current_speed
        desired_tilt = np.clip(self.speed_kp * speed_error, -np.pi / 2, np.pi / 2)

        heading_error = np.arctan2(np.sin(target_heading - current_heading), np.cos(target_heading - current_heading))
        desired_yaw_rate = self.heading_kp * heading_error

        command = {
            "altitude": float(target_altitude),
            "speed": float(target_speed),
            "heading": float(target_heading),
            "desired_pitch": float(desired_pitch),
            "desired_tilt": float(desired_tilt),
            "desired_yaw_rate": float(desired_yaw_rate),
            "altitude_error": float(altitude_error),
            "speed_error": float(speed_error),
            "heading_error": float(heading_error),
        }
        return command

    def update(self, state, target_altitude: float, target_speed: float, target_heading: float = 0.0,
               dt: float = 0.01):
        return self.compute(state, target_altitude, target_speed, target_heading, dt)
