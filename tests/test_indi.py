import numpy as np
from src.control.indi import INDIController


def test_indi_zero_error_holds_command():
    indi = INDIController()
    for _ in range(100):
        cmd = indi.compute(desired_pitch=0.0, current_pitch=0.0, current_pitch_rate=0.0, dt=0.01)
    assert abs(cmd) < 0.01


def test_indi_positive_error_commands_negative_vtail():
    indi = INDIController()
    cmd = indi.compute(desired_pitch=0.1, current_pitch=0.0, current_pitch_rate=0.0, dt=0.01)
    assert cmd < 0.0


def test_indi_converges_to_zero_error():
    indi = INDIController(B_eff=-50.0, pitch_gain=4.0, limit=0.5)
    B_eff_true = -50.0
    pitch = 0.0
    pitch_rate = 0.0
    target_pitch = 0.1
    dt = 0.01
    for _ in range(500):
        vtail = indi.compute(target_pitch, pitch, pitch_rate, dt)
        pitch_accel = B_eff_true * vtail
        pitch_rate += pitch_accel * dt
        pitch += pitch_rate * dt
    assert abs(pitch - target_pitch) < 0.02


def test_indi_respects_limit():
    indi = INDIController(B_eff=-50.0, pitch_gain=10.0, limit=0.2)
    for _ in range(100):
        cmd = indi.compute(desired_pitch=1.0, current_pitch=0.0, current_pitch_rate=0.0, dt=0.01)
        assert abs(cmd) <= 0.2 + 1e-9


def test_indi_handles_model_mismatch():
    indi = INDIController(B_eff=-50.0, pitch_gain=4.0, limit=0.5)
    B_eff_true = -35.0
    pitch = 0.0
    pitch_rate = 0.0
    target_pitch = 0.1
    dt = 0.01
    for _ in range(500):
        vtail = indi.compute(target_pitch, pitch, pitch_rate, dt)
        pitch_accel = B_eff_true * vtail
        pitch_rate += pitch_accel * dt
        pitch += pitch_rate * dt
    assert abs(pitch - target_pitch) < 0.05
