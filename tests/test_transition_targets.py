import numpy as np
import pytest

from src.simulation.transition_targets import assess_transition_targets


def samples():
    alpha = np.deg2rad(8.0)
    return {
        "t": np.array([1.0, 4.0, 20.99, 21.0]),
        "alt": np.array([100.0, 99.5, 98.766, 100.0]),
        "ref_alt": np.full(4, 100.0),
        "pitch": np.array([0.0, 2.0, 2.8, 2.8]),
        "tilt": np.array([0.0, 10.0, 90.0, 90.0]),
        "V_T": np.array([0.0, 14.99, 15.0, 15.0]),
        "u_body": np.array([0.0, 14.99, 15.0 * np.cos(alpha), 15.0]),
        "w_body": np.array([0.0, 0.0, 15.0 * np.sin(alpha), 0.0]),
    }


def evaluate(data):
    return {
        row["Target"]: row["Result"]
        for row in assess_transition_targets(data, transition_start_s=1.0, transition_end_s=21.0)
    }


def test_target_limits_and_settling_are_inclusive():
    assert set(evaluate(samples()).values()) == {"Met"}


def test_incomplete_transition_never_passes_and_violations_are_reported():
    data = {key: value[:-1] for key, value in samples().items()}
    data["pitch"][1] = 3.0
    data["tilt"][-1] = 40.0
    data["V_T"][:] = 0.0
    results = evaluate(data)

    assert results["Transition completion"] == "Not met"
    assert results["Peak altitude loss: 0.5–1.234 m"] == "Unassessed (incomplete)"
    assert results["Pitch after 3 s: +2.0–+2.8°"] == "Not met"
    assert results["Angle of attack at ≥15 m/s: 5–12°"] == "Unassessed"


def test_target_assessment_rejects_mismatched_data():
    data = samples()
    data["pitch"] = np.array([1.0])
    with pytest.raises(ValueError, match="matching one-dimensional"):
        evaluate(data)
