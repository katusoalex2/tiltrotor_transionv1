"""Rigorous Coordinate System Definitions for the Tilt-Rotor UAV."""

"""
this module defines the coordinate systems used in the tilt-rotor UAV dynamics model. The following frames are defined:
- Body Frame: Forward-Right-Down (FRD)
- Earth Frame: North-East-Down (NED) 

Axis           COordinate System
 X             Forward(North[+], South[-])
 Y             Right(East[+], West[-])
 Z             Down(Down[+], Up[-])

 - Wind Frame: Longitudinal-Lateral-Vertical (LLV)
- Nacelle Frame: Tilt Frame
"""

import numpy as np


def rotation_body_to_earth(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Returns the Direction Cosine Matrix (DCM) from Body to Earth (NED).
    Standard 3-2-1 (Z-Y-X) sequence.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp,   sr*cp,            cr*cp]
    ])


def rotation_earth_to_body(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Returns the DCM from Earth to Body (transpose of body-to-earth)."""
    return rotation_body_to_earth(roll, pitch, yaw).T


def rotation_body_to_wind(alpha: float, beta: float) -> np.ndarray:
    """Returns the DCM from Body to Wind Frame (Stevens & Lewis, Eq 2.4-4).
    
    At alpha=90, beta=0, body X points to wind -Z (physically correct:
    when nose is straight up, wind Z (down) is opposite to body X).
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)

    return np.array([
        [ca*cb,  sb,   sa*cb],
        [-ca*sb, cb,  -sa*sb],
        [-sa,    0.0,  ca]
    ])


def rotation_body_to_nacelle(tilt_angle: float) -> np.ndarray:
    """Returns the DCM from Body to Nacelle (Tilt) Frame.
    
    Rotates around Y-body axis by tilt_angle.
    """
    ct, st = np.cos(tilt_angle), np.sin(tilt_angle)
    return np.array([
        [ct,  0.0, st],
        [0.0, 1.0, 0.0],
        [-st, 0.0, ct]
    ])


def body_rates_to_euler_rates(roll: float, pitch: float, omega: np.ndarray) -> np.ndarray:
    """Converts [p, q, r] body rates to [roll_dot, pitch_dot, yaw_dot].
    Stevens & Lewis Eq 1.4-4.
    """
    p, q, r = omega
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)

    if abs(cp) < 1e-6:
        cp = 1e-6 if cp >= 0 else -1e-6
    tp = sp / cp

    phi_dot = p + (q * sr + r * cr) * tp
    theta_dot = q * cr - r * sr
    psi_dot = (q * sr + r * cr) / cp
    return np.array([phi_dot, theta_dot, psi_dot])


def euler_rates_to_body_rates(roll: float, pitch: float, euler_rates: np.ndarray) -> np.ndarray:
    """Converts [roll_dot, pitch_dot, yaw_dot] to [p, q, r] body rates.
    Inverse of body_rates_to_euler_rates. Stevens & Lewis Eq 1.4-5.
    """
    phi_dot, theta_dot, psi_dot = euler_rates
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)

    p = phi_dot - psi_dot * sp
    q = theta_dot * cr + psi_dot * sr * cp
    r = -theta_dot * sr + psi_dot * cr * cp
    return np.array([p, q, r])
