"""3D Rotation Matrices for transforming between Body and World frames.

Implements the Body -> Inertial matrix from .
World Frame is NED (X=East, Y=North, Z=Down).
Body Frame is Forward-Right-Down.
"""

import numpy as np


def rotation_body_to_world(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Returns the 3x3 rotation matrix R_b2w.
    
    Use this to convert a vector in the Body frame to the World frame.
    (Note: This is the transpose of the Inertial-to-Body matrix shown on ).
    """
    c_phi = np.cos(roll)
    s_phi = np.sin(roll)
    c_theta = np.cos(pitch)
    s_theta = np.sin(pitch)
    c_psi = np.cos(yaw)
    s_psi = np.sin(yaw)

    # (Body to Inertial)
    R = np.array([
        [c_theta * c_psi, s_phi * s_theta * c_psi - c_phi * s_psi, c_phi * s_theta * c_psi + s_phi * s_psi],
        [c_theta * s_psi, s_phi * s_theta * s_psi + c_phi * c_psi, c_phi * s_theta * s_psi - s_phi * c_psi],
        [-s_theta,        s_phi * c_theta,                         c_phi * c_theta]
    ])
    return R


def kinematic_euler_rates(roll: float, pitch: float, omega: np.ndarray) -> np.ndarray:
    """Converts body angular rates (p, q, r) to Euler angle rates (roll_dot, pitch_dot, yaw_dot).

    P = roll_dot, Q = pitch_dot, R = yaw_dot. 
    This is a simplification (valid for small angles). We will use the rigorous
    kinematic matrix to avoid gimbal lock issues later.
    """
    p, q, r = omega[0], omega[1], omega[2]
    
    c_theta = np.cos(pitch)
    s_theta = np.sin(pitch)
    t_theta = np.tan(pitch)
    
    # If pitch is exactly 90 degrees, this matrix blows up. 
    # We clamp it to avoid division by zero in early phases.
    if abs(c_theta) < 1e-6:
        c_theta = 1e-6 if c_theta >= 0 else -1e-6
    
    # Standard aerospace Euler kinematic equations
    phi_dot = p + q * s_theta * t_theta + r * c_theta * t_theta
    theta_dot = q * c_theta - r * s_theta
    psi_dot = (q * s_theta + r * c_theta) / c_theta
    
    return np.array([phi_dot, theta_dot, psi_dot])
