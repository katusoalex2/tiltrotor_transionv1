import numpy as np
import pytest

from src.control.reference import TransitionReference


def test_cosine_tilt_reference_holds_then_transitions_smoothly():
    ref = TransitionReference(
        start_altitude=100.0,
        target_speed=20.0,
        transition_time=10.0,
        tilt_hold_time=2.0,
    )

    assert ref.at(1.0)["tilt"] == 0.0
    assert np.isclose(ref.at(2.0)["tilt"], 0.0)
    assert np.isclose(ref.at(7.0)["tilt"], np.pi / 4)
    assert np.isclose(ref.at(12.0)["tilt"], np.pi / 2)
    assert np.isclose(ref.at(15.0)["tilt"], np.pi / 2)

    epsilon = 1e-5
    start_rate = (ref.at(2.0 + epsilon)["tilt"] - ref.at(2.0)["tilt"]) / epsilon
    end_rate = (ref.at(12.0)["tilt"] - ref.at(12.0 - epsilon)["tilt"]) / epsilon
    assert abs(start_rate) < 1e-5
    assert abs(end_rate) < 1e-5


@pytest.mark.parametrize(
    ("transition_time", "tilt_hold_time"),
    [(0.0, 0.0), (-1.0, 0.0), (1.0, -0.1)],
)
def test_transition_reference_rejects_invalid_timing(transition_time, tilt_hold_time):
    with pytest.raises(ValueError):
        TransitionReference(100.0, 20.0, transition_time, tilt_hold_time)
