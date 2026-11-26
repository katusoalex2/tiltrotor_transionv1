"""Shared attached-flow and post-stall wing lift/drag coefficient model."""

import casadi as ca
import numpy as np


def low_speed_aerodynamic_factor(speed_mps, full_load_speed_mps=0.5):
    """Blend the optional low-speed load onset while keeping the existing cutoff."""
    symbolic = isinstance(speed_mps, (ca.SX, ca.MX, ca.DM))
    if full_load_speed_mps <= 0.5:
        return ca.if_else(speed_mps >= 0.5, 1.0, 0.0) if symbolic else float(speed_mps >= 0.5)
    ratio = (speed_mps - 0.5) / (full_load_speed_mps - 0.5)
    blend = ca.fmin(1.0, ca.fmax(0.0, ratio)) if symbolic else np.clip(ratio, 0.0, 1.0)
    return blend**2 * (3.0 - 2.0 * blend)


def wing_lift_drag_coefficients(
    alpha_rad,
    *,
    cl_zero: float,
    lift_slope_per_rad: float,
    stall_angle_rad: float,
    cd_zero: float,
    aspect_ratio: float,
    oswald_efficiency: float,
):
    """Return CL and CD using a linear pre-stall curve and cosine-squared decay."""
    symbolic = isinstance(alpha_rad, (ca.SX, ca.MX, ca.DM))
    if symbolic:
        abs_alpha = ca.fabs(alpha_rad)
        sign_alpha = ca.sign(alpha_rad)
        cosine = ca.cos
        choose = ca.if_else
    else:
        alpha_rad = np.asarray(alpha_rad, dtype=float)
        abs_alpha = np.abs(alpha_rad)
        sign_alpha = np.sign(alpha_rad)
        cosine = np.cos
        choose = np.where

    cl_linear = cl_zero + lift_slope_per_rad * alpha_rad
    cl_at_stall = cl_zero + lift_slope_per_rad * sign_alpha * stall_angle_rad
    cl_post_stall = cl_at_stall * cosine(
        alpha_rad - sign_alpha * stall_angle_rad
    ) ** 2
    cl = choose(abs_alpha < stall_angle_rad, cl_linear, cl_post_stall)
    cd = cd_zero + cl**2 / (
        np.pi * oswald_efficiency * aspect_ratio
    )
    return cl, cd
