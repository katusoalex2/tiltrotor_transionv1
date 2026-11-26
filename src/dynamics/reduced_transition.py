"""Reduced altitude model for the legacy corridor schedule studies."""

import casadi as ca
import numpy as np

from src.configuration.aircraft_parameters import AircraftParameters


def altitude_acceleration(tilt, aircraft: AircraftParameters):
    """Fixed-hover-RPM approximation, not trim or a dynamic speed solution."""
    count = len(aircraft.motor_positions)
    tilting = sum(position[3] for position in aircraft.motor_positions)
    fixed = count - tilting
    hover_omega = np.sqrt(aircraft.mass * aircraft.gravity / (count * aircraft.k_thrust))
    omega = min(hover_omega, aircraft.max_omega)
    thrust = aircraft.k_thrust * omega**2
    rotor_vertical = thrust * (fixed + tilting * ca.cos(tilt))
    speed = 12.0 * tilt / (np.pi / 2.0)
    lift = 0.5 * aircraft.air_density * aircraft.wing_area * 0.5 * speed**2 * ca.sin(tilt)
    return (rotor_vertical + lift - aircraft.mass * aircraft.gravity) / aircraft.mass
