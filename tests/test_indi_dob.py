"""Test the DOB-augmented INDI controller."""

import numpy as np
from src.control.indi_dob import INDIControllerWithDOB


def test_dob_compensates_constant_disturbance():
    """Simulate a constant disturbance; DOB should drive it toward zero."""
    indi = INDIControllerWithDOB(B_eff=-50.0, pitch_gain=4.0, limit=0.5)

    # Plant with constant 0.5 rad/s^2 disturbance on acceleration
    B_eff_true = -50.0
    disturbance = 0.5
    pitch = 0.0
    pitch_rate = 0.0
    target_pitch = 0.1
    dt = 0.01

    errors = []
    for i in range(500):
        vtail = indi.compute(target_pitch, pitch, pitch_rate, dt)
        pitch_accel = B_eff_true * vtail + disturbance
        pitch_rate += pitch_accel * dt
        pitch += pitch_rate * dt
        if i > 100:
            errors.append(abs(pitch - target_pitch))

    # With DOB, final tracking error should be small (DOB cancels the disturbance)
    assert np.mean(errors[-100:]) < 0.05, f"Mean error: {np.mean(errors[-100:])}"


def test_dob_estimates_disturbance():
    """DOB's disturbance estimate should approach the true disturbance."""
    indi = INDIControllerWithDOB(B_eff=-50.0, pitch_gain=4.0, limit=0.5)

    B_eff_true = -50.0
    disturbance = 1.0
    pitch = 0.0
    pitch_rate = 0.0
    dt = 0.01

    for _ in range(1000):
        vtail = indi.compute(0.0, pitch, pitch_rate, dt)
        pitch_accel = B_eff_true * vtail + disturbance
        pitch_rate += pitch_accel * dt
        pitch += pitch_rate * dt

    # DOB estimate should be close to 1.0 (the true disturbance)
    assert abs(indi.disturbance_estimate - disturbance) < 0.3, \
        f"DOB estimate: {indi.disturbance_estimate}, true: {disturbance}"