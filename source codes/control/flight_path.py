"""Flight-path regulator for steering the tilt-rotor toward a target path."""

import numpy as np


class FlightPathRegulator:
    """A simple regulator that converts path tracking errors into tilt and pitch commands."""

    def __init__(self, altitude_kp: float = 0.7, speed_kp: float = 0.8, pitch_kp: float = 0.9,
                 max_pitch_rad: float = np.deg2rad(25.0), max_tilt_rad: float = np.pi / 2):
        self.altitude_kp = float(altitude_kp)
        self.speed_kp = float(speed_kp)
        self.pitch_kp = float(pitch_kp)
        self.max_pitch_rad = float(max_pitch_rad)
        self.max_tilt_rad = float(max_tilt_rad)

    def compute(self, state, target_altitude: float, target_speed: float, target_heading: float = 0.0,
                dt: float = 0.01):
        current_altitude = float(-state.pos[2])
        current_speed = float(state.vel[0])
        current_pitch = float(state.att[1])

        altitude_error = target_altitude - current_altitude
        pitch_command = self.pitch_kp * (self.altitude_kp * altitude_error)
        pitch_command = np.clip(pitch_command, -self.max_pitch_rad, self.max_pitch_rad)

        speed_error = target_speed - current_speed
        tilt_command = np.clip(self.speed_kp * speed_error, -self.max_tilt_rad, self.max_tilt_rad)

        heading_error = np.arctan2(np.sin(target_heading - state.att[2]), np.cos(target_heading - state.att[2]))
        yaw_command = 0.25 * heading_error

        return {
            "desired_pitch": float(pitch_command),
            "desired_tilt": float(tilt_command),
            "desired_yaw": float(yaw_command),
            "pitch_error": float(altitude_error),
            "speed_error": float(speed_error),
            "heading_error": float(heading_error),
            "current_pitch": float(current_pitch),
        }

    def update(self, state, target_altitude: float, target_speed: float, target_heading: float = 0.0,
               dt: float = 0.01):
        return self.compute(state, target_altitude, target_speed, target_heading, dt)
