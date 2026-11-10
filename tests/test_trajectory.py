"""Unit tests for ReferenceTrajectory and LinearReference."""

import os
import tempfile
import numpy as np
import pytest

from src.control.trajectory import ReferenceTrajectory, LinearReference


@pytest.fixture
def sample_corridor():
    """Create a temporary corridor CSV for testing."""
    data = np.array([
        [0.0, 0.0, 100.0],
        [5.0, 45.0, 95.0],
        [10.0, 90.0, 90.0],
    ])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("time_s,tilt_deg,altitude_m\n")
        for row in data:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")
        path = f.name
    yield path
    os.remove(path)


def test_trajectory_starts_at_zero_tilt(sample_corridor):
    traj = ReferenceTrajectory(sample_corridor, speed_max=12.0)
    r = traj.at(0.0)
    assert np.isclose(r["tilt"], 0.0, atol=1e-6)
    assert np.isclose(r["altitude"], 100.0)
    assert np.isclose(r["speed"], 0.0)


def test_trajectory_ends_at_cruise(sample_corridor):
    traj = ReferenceTrajectory(sample_corridor, speed_max=12.0)
    r = traj.at(10.0)
    assert np.isclose(np.rad2deg(r["tilt"]), 90.0, atol=0.1)
    assert np.isclose(r["altitude"], 90.0)
    assert np.isclose(r["speed"], 12.0)


def test_trajectory_interpolates(sample_corridor):
    traj = ReferenceTrajectory(sample_corridor, speed_max=12.0)
    r = traj.at(5.0)
    assert np.isclose(np.rad2deg(r["tilt"]), 45.0, atol=0.1)
    assert np.isclose(r["altitude"], 95.0)


def test_trajectory_clamps_outside_range(sample_corridor):
    traj = ReferenceTrajectory(sample_corridor, speed_max=12.0)
    r_before = traj.at(-5.0)
    r_after = traj.at(999.0)
    assert np.isclose(r_before["tilt"], 0.0)
    assert np.isclose(np.rad2deg(r_after["tilt"]), 90.0, atol=0.1)


def test_linear_reference_matches_old_behavior():
    ref = LinearReference(start_altitude=100.0, target_speed=20.0, transition_time=10.0)
    r = ref.at(5.0)
    assert np.isclose(np.rad2deg(r["tilt"]), 45.0, atol=0.1)
    assert np.isclose(r["speed"], 10.0)
    assert np.isclose(r["altitude"], 100.0)