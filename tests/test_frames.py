"""Unit tests for rigorous coordinate frames."""

import numpy as np
from src.dynamics.frames import (
    rotation_body_to_earth,
    rotation_earth_to_body,
    rotation_body_to_wind,
    rotation_body_to_nacelle,
    body_rates_to_euler_rates,
    euler_rates_to_body_rates,
)


def test_dcm_is_orthogonal():
    R = rotation_body_to_earth(0.3, -0.2, 1.1)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_earth_to_body_is_transpose():
    R_b2e = rotation_body_to_earth(0.1, 0.2, 0.3)
    R_e2b = rotation_earth_to_body(0.1, 0.2, 0.3)
    assert np.allclose(R_e2b, R_b2e.T)


def test_zero_rotation_is_identity():
    R = rotation_body_to_earth(0.0, 0.0, 0.0)
    assert np.allclose(R, np.eye(3))


def test_pitch_90_rotates_body_x_to_earth_neg_z():
    R = rotation_body_to_earth(0.0, np.pi / 2, 0.0)
    body_x = np.array([1.0, 0.0, 0.0])
    earth_vector = R @ body_x
    assert np.allclose(earth_vector, [0.0, 0.0, -1.0], atol=1e-9)


def test_yaw_90_rotates_body_x_to_earth_north_to_east():
    R = rotation_body_to_earth(0.0, 0.0, np.pi / 2)
    body_x = np.array([1.0, 0.0, 0.0])
    earth_vector = R @ body_x
    assert np.allclose(earth_vector, [0.0, 1.0, 0.0], atol=1e-9)


def test_wind_frame_zero_alpha_beta():
    R_w = rotation_body_to_wind(0.0, 0.0)
    assert np.allclose(R_w, np.eye(3))


def test_wind_frame_90_alpha():
    """90 deg AoA: body X points to wind -Z (nose up = opposite of wind down)."""
    R_w = rotation_body_to_wind(np.pi / 2, 0.0)
    body_x = np.array([1.0, 0.0, 0.0])
    wind_vector = R_w @ body_x
    assert np.allclose(wind_vector, [0.0, 0.0, -1.0], atol=1e-9)


def test_nacelle_tilt_zero():
    R = rotation_body_to_nacelle(0.0)
    assert np.allclose(R, np.eye(3))


def test_nacelle_tilt_90_vects_thrust_forward():
    R = rotation_body_to_nacelle(np.pi / 2)
    nacelle_up = np.array([0.0, 0.0, -1.0])
    body_thrust_dir = R @ nacelle_up
    assert np.allclose(body_thrust_dir, [-1.0, 0.0, 0.0], atol=1e-9)


def test_euler_rates_roundtrip():
    """body ? euler ? body must return the original omega."""
    omega = np.array([0.1, 0.2, 0.3])
    roll, pitch = 0.15, 0.25
    euler_rates = body_rates_to_euler_rates(roll, pitch, omega)
    omega_recovered = euler_rates_to_body_rates(roll, pitch, euler_rates)
    assert np.allclose(omega, omega_recovered, atol=1e-9)
