"""Convert between thrust-axis elevation and model deflection from vertical.

Project angle is measured up from forward. A vertical rotor is 90 degrees
and a forward-facing rotor is zero degrees. The rigid-body integrator and
controller use deflection from vertical for their rotations and schedules.
"""

import numpy as np


HOVER_TILT_DEG = 90.0
CRUISE_TILT_DEG = 0.0
TILT_CONVENTION = "90 deg upward (hover), 0 deg forward (cruise)"


def tilt_deg_from_deflection_deg(deflection_deg):
    """Convert model deflection in degrees to thrust-axis elevation."""
    return HOVER_TILT_DEG - deflection_deg


def tilt_deg_from_deflection_rad(deflection_rad):
    """Convert model deflection in radians to thrust-axis elevation."""
    return tilt_deg_from_deflection_deg(np.rad2deg(deflection_rad))


def deflection_rad_from_tilt_deg(tilt_deg):
    """Convert thrust-axis elevation to the model's rotation from vertical."""
    return np.deg2rad(HOVER_TILT_DEG - tilt_deg)
