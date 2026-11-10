##"""Tests for AircraftState: the pure-data container."""

import numpy as np

from src.dynamics.state import AircraftState


def test_default_state_is_all_zeros():
    s = AircraftState()
    assert np.array_equal(s.pos, np.zeros(3))
    assert np.array_equal(s.vel, np.zeros(3))
    assert np.array_equal(s.att, np.zeros(3))
    assert np.array_equal(s.omega, np.zeros(3))
    assert s.tilt == 0.0


def test_instances_do_not_share_arrays():
    """Classic dataclass trap: if default_factory is missing,
    every instance shares the same arrays, and mutating one affects all."""
    s1 = AircraftState()
    s2 = AircraftState()
    s1.pos[0] = 999.0
    assert s2.pos[0] == 0.0, "Instances are sharing arrays — fix default_factory!"


def test_copy_is_a_deep_copy():
    s1 = AircraftState()
    s1.pos = np.array([1.0, 2.0, 3.0])
    s1.vel = np.array([4.0, 5.0, 6.0])
    s1.tilt = 0.5

    s2 = s1.copy()

    # Independently mutating the copy must not affect the original.
    s2.pos[0] = 777.0
    s2.vel[0] = 888.0
    s2.tilt = 1.0

    assert s1.pos[0] == 1.0
    assert s1.vel[0] == 4.0
    assert s1.tilt == 0.5


def test_copy_preserves_values():
    s1 = AircraftState()
    s1.pos = np.array([1.0, 2.0, 3.0])
    s2 = s1.copy()
    assert np.array_equal(s1.pos, s2.pos)