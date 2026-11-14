import numpy as np
import pytest

from src.control.envelope_protection import EnvelopeProtection
from src.control.transition_observer import TransitionStateObserver
from src.control.architecture import ControlLawArchitecture
from src.dynamics.state import AircraftState
from src.simulation.failures import ActuatorFailure
from src.simulation.monte_carlo import MonteCarloValidator
from src.wind.gust import DiscreteGust, DrydenGust


def test_discrete_gust_is_smooth_reproducible_and_resets():
    gust = DiscreteGust(2.0, 4.0, (8.0, -2.0, 1.0))

    assert np.array_equal(gust.get_wind(1.0), np.zeros(3))
    assert np.allclose(gust.get_wind(2.0), np.zeros(3))
    assert np.allclose(gust.get_wind(4.0), np.array([8.0, -2.0, 1.0]))
    assert np.allclose(gust.get_wind(6.0), np.zeros(3))
    gust.step(3.0)
    gust.reset()
    assert gust.time == 0.0


def test_dryden_gust_uses_seed_and_configured_time_constant():
    wind_a = DrydenGust(sigma=(1.0, 2.0, 0.0), seed=12, tau=0.5)
    wind_b = DrydenGust(sigma=(1.0, 2.0, 0.0), seed=12, tau=0.5)
    samples_a = np.array([wind_a.step(0.01) for _ in range(100)])
    samples_b = np.array([wind_b.step(0.01) for _ in range(100)])

    assert np.allclose(samples_a, samples_b)
    assert np.all(samples_a[:, 2] == 0.0)
    wind_a.reset()
    assert np.allclose(wind_a.step(0.01), samples_a[0])


def test_transition_observer_compensates_for_wind_and_filters_state():
    observer = TransitionStateObserver(time_constant_s=0.2, cruise_speed_mps=20.0)
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([15.0, 0.0, 0.0]),
        att=np.zeros(3),
        omega=np.zeros(3),
        tilt=np.deg2rad(45.0),
    )

    estimate = observer.update(state, wind_earth=np.array([5.0, 0.0, 0.0]), dt=0.01)

    assert estimate.airspeed_mps == pytest.approx(10.0)
    assert estimate.altitude_m == pytest.approx(100.0)
    assert estimate.phase == "transition"
    state.vel[0] = 30.0
    filtered = observer.update(state, wind_earth=np.zeros(3), dt=0.01)
    assert 10.0 < filtered.airspeed_mps < 30.0


def test_envelope_protection_holds_tilt_below_transition_speed():
    envelope = {
        "tilt_deg": np.array([0.0, 45.0, 90.0]),
        "minimum_speed_mps": np.array([0.0, 5.0, 10.0]),
        "maximum_speed_mps": np.array([20.0, 30.0, 40.0]),
        "feasible": np.array([True, True, True]),
    }
    observer = TransitionStateObserver()
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([2.0, 0.0, 0.0]),
        att=np.zeros(3),
        omega=np.zeros(3),
        tilt=np.deg2rad(10.0),
    )
    estimate = observer.update(state, dt=0.01)

    result = EnvelopeProtection(envelope, speed_margin_mps=0.5).apply(
        target_speed_mps=12.0,
        target_tilt_rad=np.deg2rad(45.0),
        estimate=estimate,
    )

    assert result.protected
    assert "airspeed_below_transition_envelope" in result.reasons
    assert result.tilt_rad == pytest.approx(state.tilt)
    assert result.speed_mps == pytest.approx(12.0)


def test_envelope_protection_limits_angle_of_attack_and_speed_reference():
    envelope = {
        "tilt_deg": np.array([0.0, 45.0, 90.0]),
        "minimum_speed_mps": np.array([0.0, 5.0, 10.0]),
        "maximum_speed_mps": np.array([20.0, 30.0, 40.0]),
        "feasible": np.array([True, True, True]),
    }
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([10.0, 0.0, 0.0]),
        att=np.array([0.0, np.deg2rad(20.0), 0.0]),
        omega=np.zeros(3),
        tilt=np.deg2rad(45.0),
    )
    estimate = TransitionStateObserver().update(state, dt=0.01)

    result = EnvelopeProtection(envelope, max_alpha_rad=np.deg2rad(10.0)).apply(
        target_speed_mps=35.0,
        target_tilt_rad=np.deg2rad(60.0),
        estimate=estimate,
    )

    assert "angle_of_attack_limit" in result.reasons
    assert "speed_reference_limited" in result.reasons
    assert result.tilt_rad == pytest.approx(state.tilt)
    assert result.speed_mps < 35.0


def test_envelope_protection_scales_tilt_change_with_control_authority():
    envelope = {
        "tilt_deg": np.array([0.0, 45.0, 90.0]),
        "minimum_speed_mps": np.array([0.0, 5.0, 10.0]),
        "maximum_speed_mps": np.array([20.0, 30.0, 40.0]),
        "feasible": np.array([True, True, True]),
    }
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([20.0, 0.0, 0.0]),
        att=np.zeros(3),
        omega=np.zeros(3),
        tilt=np.deg2rad(10.0),
    )
    estimate = TransitionStateObserver().update(state, dt=0.01)

    result = EnvelopeProtection(envelope).apply(
        target_speed_mps=20.0,
        target_tilt_rad=np.deg2rad(45.0),
        estimate=estimate,
        available_control_authority=0.5,
    )

    assert "control_authority_limited" in result.reasons
    assert result.tilt_rad == pytest.approx(np.deg2rad(27.5))


def test_control_architecture_uses_observer_and_envelope_protection():
    envelope = {
        "tilt_deg": np.array([0.0, 45.0, 90.0]),
        "minimum_speed_mps": np.array([0.0, 5.0, 10.0]),
        "maximum_speed_mps": np.array([20.0, 30.0, 40.0]),
        "feasible": np.array([True, True, True]),
    }
    positions = (
        (0.3, 0.4, 0.0, True),
        (0.3, -0.4, 0.0, True),
        (-0.3, 0.4, 0.0, False),
        (-0.3, -0.4, 0.0, False),
    )
    controller = ControlLawArchitecture(
        mass=5.0,
        k_thrust=1e-5,
        max_omega=3000.0,
        motor_positions=positions,
        envelope_protection=EnvelopeProtection(envelope),
    )
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([20.0, 0.0, 0.0]),
        att=np.zeros(3),
        omega=np.zeros(3),
        tilt=np.deg2rad(10.0),
    )

    command = controller.compute(
        state,
        {"altitude": 100.0, "speed": 20.0, "tilt": np.deg2rad(45.0)},
        dt=0.01,
        available_control_authority=0.5,
    )

    assert command["envelope_protection_active"]
    assert command["transition_airspeed_mps"] == pytest.approx(20.0)
    assert command["requested_tilt"] == pytest.approx(np.deg2rad(45.0))
    assert command["protected_tilt"] == pytest.approx(np.deg2rad(27.5))
    assert command["tilt"] == pytest.approx(np.deg2rad(27.5))


def test_actuator_failure_applies_only_after_start_time():
    failure = ActuatorFailure("omega_fl", start_time_s=2.0, remaining_effectiveness=0.25)
    commands = {"omega_fl": 100.0, "omega_fr": 100.0}

    assert failure.apply(commands, 1.0)["omega_fl"] == 100.0
    assert failure.apply(commands, 2.0)["omega_fl"] == 25.0
    assert commands["omega_fl"] == 100.0


def test_monte_carlo_is_seeded_and_reports_safety_failures():
    validator = MonteCarloValidator(
        trials=12,
        seed=42,
        failure_probability=0.5,
        failure_time_range_s=(2.0, 4.0),
    )

    def simulate(scenario):
        return {
            "completed": scenario.trial_index != 0,
            "max_abs_pitch_rad": 0.1 if scenario.trial_index else 1.2,
            "max_abs_alpha_rad": 0.1,
            "min_altitude_m": 10.0,
        }

    first = validator.run(simulate)
    second = validator.run(simulate)

    assert first.success_rate == second.success_rate
    assert [trial.scenario for trial in first.trials] == [
        trial.scenario for trial in second.trials
    ]
    assert len(first.trials) == 12
    assert first.trials[0].failure_reasons == ("simulation_incomplete", "max_abs_pitch_rad_max_limit")
    assert "max_abs_pitch_rad" in first.metric_percentiles
    gust_a = first.trials[0].scenario.make_gust(airspeed_mps=12.0)
    gust_b = first.trials[0].scenario.make_gust(airspeed_mps=12.0)
    assert np.allclose(gust_a.step(0.01), gust_b.step(0.01))


def test_monte_carlo_rejects_missing_safety_metrics():
    validator = MonteCarloValidator(trials=1)
    with pytest.raises(ValueError, match="missing safety fields"):
        validator.run(lambda _: {"completed": True})
