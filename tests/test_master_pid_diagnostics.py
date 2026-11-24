import numpy as np
import pytest

from scripts import plot_master_pid_diagnostics as diagnostics


def test_gust_is_passed_as_wind_without_changing_ground_velocity(monkeypatch):
    observed = {}
    original_step = diagnostics.TiltrotorDynamics.step

    def capture_step(self, state, *args, **kwargs):
        if not observed:
            observed["velocity"] = state.vel.copy()
            observed["wind"] = args[0].copy()
        return original_step(self, state, *args, **kwargs)

    monkeypatch.setattr(diagnostics.TiltrotorDynamics, "step", capture_step)
    log, stop_reason = diagnostics.simulate(duration=0.01)

    np.testing.assert_array_equal(observed["velocity"], np.zeros(3))
    assert np.linalg.norm(observed["wind"]) > 0
    assert stop_reason is None
    np.testing.assert_allclose(log["time"], [0.01])


def test_simulation_stops_when_pitch_limit_is_exceeded():
    log, stop_reason = diagnostics.simulate(duration=5.0, pitch_limit_deg=1.0)

    assert stop_reason is not None
    assert "pitch exceeded" in stop_reason
    assert log["time"][-1] < 5.0
    assert abs(log["pitch"][-1]) >= 1.0
    assert all(len(values) == len(log["time"]) for values in log.values())
    assert np.all(np.isfinite(log["altitude"]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration": 0.0},
        {"dt": 0.0},
        {"pitch_limit_deg": 90.0},
        {"gust_sigma": -1.0},
    ],
)
def test_simulation_rejects_invalid_diagnostic_inputs(kwargs):
    with pytest.raises(ValueError):
        diagnostics.simulate(**kwargs)
