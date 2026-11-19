import numpy as np
import pytest

from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.dynamics.rotor import RotorModel
from src.dynamics.state import AircraftState
from src.dynamics.tiltrotor import TiltrotorDynamics
import src.dynamics.transition as transition_module
from src.dynamics.transition import (
    assess_motor_electrical_point,
    assess_rotor_electrical_point,
    control_effectiveness_map,
    constrained_transition_corridor,
    solve_longitudinal_trim,
    transition_stability_map,
    transition_speed_envelope,
)


def make_dynamics():
    return TiltrotorDynamics(
        mass=5.0,
        rotor=RotorModel(k_thrust=1e-5, max_omega=3000.0),
        inertia=np.array([0.1, 0.2, 0.3]),
        aero=AeroModel(
            wing_area=0.6,
            vtail=VtailModel(area=0.1, arm_length=0.5),
        ),
    )


def test_transition_speed_envelope_has_tilt_dependent_speed_limits():
    envelope = transition_speed_envelope(
        mass=5.0,
        wing_area=0.6,
        max_thrust_per_rotor=20.0,
        rotor_tilts=[True, True, False, False],
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
    )

    assert envelope["feasible"].shape == (46,)
    assert envelope["minimum_speed_mps"][0] == 0.0
    assert envelope["maximum_speed_mps"][0] == 0.0
    assert envelope["minimum_speed_mps"][-1] > 0.0
    assert envelope["maximum_speed_mps"][-1] > envelope["minimum_speed_mps"][-1]


def test_transition_speed_envelope_rejects_missing_rotors():
    with pytest.raises(ValueError, match="at least one rotor"):
        transition_speed_envelope(
            mass=5.0,
            wing_area=0.6,
            max_thrust_per_rotor=20.0,
            rotor_tilts=[],
            cl_max=1.1,
            cd_0=0.02,
            aspect_ratio=6.0,
        )


def test_longitudinal_trim_balances_hover_forces_and_pitch_moment():
    dynamics = make_dynamics()

    trim = solve_longitudinal_trim(dynamics, airspeed_mps=0.0, tilt_rad=0.0)

    assert trim["feasible"]
    assert trim["max_normalized_residual"] < 0.02
    assert np.isclose(trim["motor_rpm"], np.sqrt(5.0 * 9.81 / (4.0e-5)) * 60 / (2 * np.pi), rtol=1e-3)


def test_longitudinal_trim_uses_distinct_front_and_rear_pair_speeds():
    dynamics = make_dynamics()
    tilt = np.deg2rad(15.0)
    trim = solve_longitudinal_trim(dynamics, airspeed_mps=2.0, tilt_rad=tilt)

    assert trim["feasible"]
    assert abs(trim["front_motor_rpm"] - trim["rear_motor_rpm"]) > 1000.0
    front_omega = trim["front_motor_rpm"] * 2 * np.pi / 60
    rear_omega = trim["rear_motor_rpm"] * 2 * np.pi / 60
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([2.0, 0.0, 0.0]),
        att=np.array([0.0, np.deg2rad(trim["alpha_deg"]), 0.0]),
        tilt=tilt,
    )
    tail = np.deg2rad(trim["tail_deflection_deg"])
    dynamics.step(state, np.zeros(3), front_omega, front_omega, rear_omega, rear_omega,
                  tilt, tail, tail, 0.01)
    np.testing.assert_allclose(
        dynamics.last_diagnostics["acceleration_ned"][[0, 2]],
        trim["acceleration_ned"][[0, 2]],
        atol=1e-8,
    )
    assert abs(dynamics.last_diagnostics["moment_body"][1]) < 0.02 * 5.0 * 9.81 * dynamics.aero.chord


def test_control_effectiveness_map_measures_motor_and_tail_moments():
    dynamics = make_dynamics()
    trim = solve_longitudinal_trim(dynamics, airspeed_mps=10.0, tilt_rad=0.0)
    labels, effectiveness = control_effectiveness_map(
        dynamics,
        airspeed_mps=10.0,
        tilt_rad=0.0,
        alpha_rad=np.deg2rad(trim["alpha_deg"]),
        collective_omega=trim["motor_rpm"] * 2.0 * np.pi / 60.0,
        tail_deflection_rad=np.deg2rad(trim["tail_deflection_deg"]),
    )

    assert len(labels) == 5
    assert effectiveness.shape == (3, 5)
    assert abs(effectiveness[1, 1]) > 0.0
    assert abs(effectiveness[1, 3]) > 0.0


def test_transition_stability_map_reports_trim_quality_and_local_derivatives():
    result = transition_stability_map(
        make_dynamics(),
        airspeeds_mps=[0.0, 10.0],
        tilt_angles_rad=[0.0, np.deg2rad(45.0)],
    )

    assert result["cm_alpha_per_rad"].shape == (2, 2)
    assert result["control_effectiveness_nm"].shape == (2, 2, 3, 5)
    assert result["control_effectiveness_labels"]
    assert np.isnan(result["cm_alpha_per_rad"][0]).all()
    assert not result["assessment_mask"][0].any()

    assert np.isfinite(result["cm_alpha_per_rad"][1, 0])
    assert result["cm_alpha_per_rad"][1, 0] < 0.0
    assert np.isclose(result["cl_beta_per_rad"][1, 0], 0.0)
    assert result["cn_beta_per_rad"][1, 0] > 0.0
    assert not result["trim_feasible"][1, 0]
    assert not result["assessment_mask"][1, 0]
    assert result["trim_motor_rpm"].shape == (2, 2)
    assert result["trim_pitch_moment_residual"].shape == (2, 2)


def test_transition_stability_map_rejects_unsorted_conditions():
    with pytest.raises(ValueError, match="airspeeds must be"):
        transition_stability_map(
            make_dynamics(),
            airspeeds_mps=[10.0, 5.0],
            tilt_angles_rad=[0.0],
        )


def test_motor_electrical_screen_requires_explicit_data_and_applies_current_limits():
    parameters = {
        "motor_kv_rpm_per_volt": 920.0,
        "motor_resistance_ohm": 0.05,
        "motor_no_load_current_a": 1.0,
        "motor_continuous_current_a": 50.0,
        "propeller_torque_coefficient_nm_s2_rad2": 1.0e-5,
        "battery_nominal_voltage_v": 22.2,
        "battery_internal_resistance_ohm": 0.05,
        "battery_capacity_ah": 5.0,
        "battery_c_rating": 30.0,
        "battery_continuous_current_a": 150.0,
        "esc_efficiency": 0.95,
        "esc_continuous_current_a": 60.0,
    }

    estimate = assess_motor_electrical_point(1000.0, 4, parameters)
    assert estimate["assessed"]
    assert estimate["feasible"]
    assert estimate["motor_current_a"] > 0.0
    assert estimate["battery_current_a"] > 0.0

    parameters["motor_continuous_current_a"] = 5.0
    limited = assess_motor_electrical_point(1000.0, 4, parameters)
    assert limited["assessed"]
    assert not limited["feasible"]

    missing = assess_motor_electrical_point(1000.0, 4, {})
    assert not missing["assessed"]
    assert "propeller_torque_coefficient_nm_s2_rad2" in missing["missing_parameters"]


def test_electrical_screen_uses_both_pair_speeds_for_battery_demand():
    parameters = {
        "motor_kv_rpm_per_volt": 920.0,
        "motor_resistance_ohm": 0.05,
        "motor_no_load_current_a": 1.0,
        "motor_continuous_current_a": 50.0,
        "propeller_torque_coefficient_nm_s2_rad2": 1.0e-5,
        "battery_nominal_voltage_v": 22.2,
        "battery_internal_resistance_ohm": 0.05,
        "battery_capacity_ah": 5.0,
        "battery_c_rating": 30.0,
        "battery_continuous_current_a": 150.0,
        "esc_efficiency": 0.95,
        "esc_continuous_current_a": 60.0,
    }
    mixed = assess_rotor_electrical_point([500.0, 500.0, 1500.0, 1500.0], parameters)
    average = assess_motor_electrical_point(1000.0, 4, parameters)
    fast = assess_motor_electrical_point(1500.0, 4, parameters)

    assert mixed["assessed"]
    assert mixed["battery_current_a"] > average["battery_current_a"]
    assert mixed["battery_current_a"] < fast["battery_current_a"]
    assert mixed["motor_current_a"] == pytest.approx(fast["motor_current_a"])


def test_constrained_corridor_applies_gust_and_tilt_rate_and_fails_closed():
    result = constrained_transition_corridor(
        make_dynamics(),
        max_thrust_per_rotor=20.0,
        rotor_tilts=[True, True, False, False],
        airspeeds_mps=[0.0, 10.0],
        tilt_angles_deg=np.arange(0.0, 91.0, 15.0),
        max_tilt_rate_deg_s=10.0,
        transition_duration_s=2.0,
        gust_speed_margin_mps=100.0,
        electrical_parameters={},
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
    )

    actuator_mask = result["actuator_rate_mask"]
    assert actuator_mask[:, :2].all()
    assert not actuator_mask[:, 2:].any()
    assert not result["gust_mask"].any()
    assert not result["qualified_mask"].any()
    expected_known_mask = (
        result["gust_mask"]
        & result["trim_mask"]
        & result["rotor_thrust_mask"]
        & result["pitch_moment_mask"]
        & result["static_stability_mask"]
        & actuator_mask
    )
    assert np.array_equal(result["known_constraints_mask"], expected_known_mask)
    assert "propeller_torque_coefficient_nm_s2_rad2" in (
        result["electrical_missing_parameters"]
    )


def test_corridor_screens_slew_and_electrical_demand_for_each_pair(monkeypatch):
    shape = (2, 2)
    stability = {
        "trim_feasible": np.ones(shape, dtype=bool),
        "trim_pitch_moment_residual": np.zeros(shape),
        "assessment_mask": np.ones(shape, dtype=bool),
        "cm_alpha_per_rad": -np.ones(shape),
        "cl_beta_per_rad": -np.ones(shape),
        "cn_beta_per_rad": np.ones(shape),
        "trim_front_motor_rpm": np.array([[1000.0, 1000.0], [1000.0, 500.0]]),
        "trim_rear_motor_rpm": np.array([[1000.0, 1000.0], [1000.0, 1500.0]]),
    }
    monkeypatch.setattr(transition_module, "transition_stability_map", lambda *a, **k: stability)
    monkeypatch.setattr(transition_module, "transition_speed_envelope", lambda **k: {
        "tilt_deg": np.array([0.0, 90.0]),
        "minimum_speed_mps": np.array([0.0, 0.0]),
        "maximum_speed_mps": np.array([30.0, 30.0]),
        "feasible": np.array([True, True]),
    })
    electrical_inputs = []
    def record_electrical(rpms, parameters):
        electrical_inputs.append(np.array(rpms))
        return {"assessed": True, "feasible": True, "missing_parameters": ()}
    monkeypatch.setattr(transition_module, "assess_rotor_electrical_point", record_electrical)
    result = constrained_transition_corridor(
        make_dynamics(),
        max_thrust_per_rotor=20.0,
        rotor_tilts=[True, True, False, False],
        airspeeds_mps=[0.0, 10.0],
        tilt_angles_deg=[0.0, 90.0],
        max_tilt_rate_deg_s=10.0,
        transition_duration_s=20.0,
        electrical_parameters={"motor_slew_rate_rpm_s": 10.0},
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
    )

    assert result["motor_slew_assessed_mask"][1, 1]
    assert not result["motor_slew_feasible_mask"][1, 1]
    assert not result["qualified_mask"][1, 1]
    assert any(np.array_equal(rpms, [500.0, 500.0, 1500.0, 1500.0]) for rpms in electrical_inputs)


def test_corridor_trim_respects_explicit_per_rotor_thrust_limit():
    dynamics = make_dynamics()
    result = constrained_transition_corridor(
        dynamics,
        max_thrust_per_rotor=1.0,
        rotor_tilts=[True, True, False, False],
        airspeeds_mps=[0.0],
        tilt_angles_deg=[0.0, 90.0],
        max_tilt_rate_deg_s=10.0,
        transition_duration_s=20.0,
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
    )
    stability = result["stability_map"]
    maximum_omega = np.sqrt(1.0 / dynamics.rotor.k_thrust)
    maximum_rpm = maximum_omega * 60.0 / (2 * np.pi)
    assert np.all(stability["trim_front_motor_rpm"] <= maximum_rpm + 1e-6)
    assert np.all(stability["trim_rear_motor_rpm"] <= maximum_rpm + 1e-6)
    assert not result["trim_mask"][0, 0]
    assert not result["qualified_mask"].any()
