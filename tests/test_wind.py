"""Unit tests for the wind models."""

import numpy as np
import pytest

from src.wind.gust import WindField, ConstantGust, DrydenGust


def test_constant_gust_returns_fixed_vector():
    wind = ConstantGust(u=3.0, v=-2.0, w=1.0)
    for _ in range(100):
        w = wind.step(dt=0.01)
        assert np.array_equal(w, np.array([3.0, -2.0, 1.0]))


def test_constant_gust_time_advances():
    wind = ConstantGust(u=1.0, v=0.0, w=0.0)
    for _ in range(100):
        wind.step(dt=0.01)
    assert np.isclose(wind.time, 1.0)


def test_dryden_gust_returns_reasonable_magnitude():
    """The gust RMS should be in the ballpark of sigma."""
    wind = DrydenGust(sigma=(2.0, 2.0, 2.0), airspeed=15.0)
    samples = []
    for _ in range(5000):
        samples.append(wind.step(dt=0.01))
    samples = np.array(samples)

    # RMS should be roughly close to sigma (within 50% for 50s of data)
    rms = np.sqrt(np.mean(samples**2, axis=0))
    for i in range(3):
        assert 0.3 * 2.0 < rms[i] < 3.0 * 2.0, f"RMS axis {i} = {rms[i]}"


def test_dryden_gust_has_correlation():
    """Successive samples should be correlated (smooth), not white noise."""
    wind = DrydenGust(sigma=(2.0, 2.0, 2.0), airspeed=15.0)
    samples = np.array([wind.step(dt=0.01) for _ in range(1000)])

    # Autocorrelation at lag 1 should be high (smooth)
    x = samples[:, 0] - samples[:, 0].mean()
    autocorr_1 = np.sum(x[:-1] * x[1:]) / np.sum(x**2)
    assert autocorr_1 > 0.8, f"Autocorrelation at lag 1 = {autocorr_1}, expected > 0.8"


def test_dryden_reproducible_with_seed():
    """Same seed should give same result."""
    wind_a = DrydenGust(seed=123)
    wind_b = DrydenGust(seed=123)
    samples_a = np.array([wind_a.step(dt=0.01) for _ in range(100)])
    samples_b = np.array([wind_b.step(dt=0.01) for _ in range(100)])
    assert np.allclose(samples_a, samples_b)


def test_dryden_reset():
    """Reset should bring gust back to zero."""
    wind = DrydenGust(seed=42)
    for _ in range(100):
        wind.step(dt=0.01)
    assert not np.allclose(wind.gust, 0.0)

    wind.reset()
    assert np.allclose(wind.gust, 0.0)
    assert wind.time == 0.0


def test_wind_field_is_abstract():
    """Base class should not be instantiable without step()."""
    base = WindField()
    with pytest.raises(NotImplementedError):
        base.step(0.01)