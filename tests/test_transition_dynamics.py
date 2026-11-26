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
    rear_thrust_sensitivity_sweep,
)


def test_disturbed_longitudinal_loads_preserve_reversed_airflow_quadrant():
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.configuration.aircraft_parameters import load_aircraft_parameters
    from src.dynamics.state import AircraftState
    from src.dynamics.transition_trajectory import longitudinal_loads

    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft, low_speed_full_load_mps=2.5)
    wind = np.array([2.0, 0.0, 0.0])
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]),
                          vel=np.array([-2.0, 0.0, -1.0]), att=np.array([0, 0.15, 0]))
    tilt, front, rear, tail = 0.25, 600.0, 500.0, 0.03
    dynamics.step(state, wind, front, front, rear, rear, tilt, tail, tail, 0.01)
    numeric = dynamics.last_diagnostics
    symbolic = longitudinal_loads(aircraft, 2.5, exact_kinematics=True)
    longitudinal = np.asarray(symbolic([-4.0, 1.0, 0.0, 0.15, 0.0], [tilt, front, rear, tail])).ravel()
    np.testing.assert_allclose(
        longitudinal[:3],
        [numeric["acceleration_ned"][0], -numeric["acceleration_ned"][2], numeric["moment_body"][1]],
        atol=1e-10,
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


def test_longitudinal_optimizer_lift_grows_with_speed_and_angle_of_attack():
    from scripts.optimize_longitudinal_transition import (
        MAX_ALPHA_RAD,
        longitudinal_dynamics,
    )

    zero_alpha = longitudinal_dynamics(12.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stall_limited = longitudinal_dynamics(
        12.0, 0.0, 0.0, 0.0, 0.0, MAX_ALPHA_RAD,
    )

    assert float(zero_alpha["lift"]) == pytest.approx(5.292)
    assert float(stall_limited["lift"]) > float(zero_alpha["lift"])
    assert float(stall_limited["lift_coefficient"]) == pytest.approx(
        0.1 + 4.0 * MAX_ALPHA_RAD
    )
    assert float(stall_limited["wing_lift_vertical_n"]) == pytest.approx(
        float(stall_limited["lift"])
    )


def test_longitudinal_optimizer_uses_shared_post_stall_polar():
    from scripts.optimize_longitudinal_transition import longitudinal_dynamics
    from src.configuration.aircraft_parameters import load_aircraft_parameters
    from src.aero.wing_polar import wing_lift_drag_coefficients

    aircraft = load_aircraft_parameters()
    alpha = np.deg2rad(30.0)
    point = longitudinal_dynamics(12.0, 0.0, 0.0, 0.0, 0.0, alpha)
    expected_cl, expected_cd = wing_lift_drag_coefficients(
        alpha,
        cl_zero=0.1,
        lift_slope_per_rad=4.0,
        stall_angle_rad=np.deg2rad(15.0),
        cd_zero=0.02,
        aspect_ratio=aircraft.wing_aspect_ratio,
        oswald_efficiency=aircraft.oswald_efficiency,
    )

    assert float(point["lift_coefficient"]) == pytest.approx(float(expected_cl))
    assert float(point["drag"]) == pytest.approx(
        0.5 * aircraft.air_density * 12.0**2 * aircraft.wing_area * float(expected_cd)
    )


def test_longitudinal_optimizer_resolves_tilted_thrust_and_vertical_support():
    from scripts.optimize_longitudinal_transition import longitudinal_dynamics

    tilt = np.deg2rad(45.0)
    point = longitudinal_dynamics(0.0, 0.0, tilt, 20.0, 15.0, 0.0)

    assert float(point["force_x_body"]) == pytest.approx(20.0 * np.sin(tilt))
    assert float(point["front_horizontal_thrust_n"]) == pytest.approx(
        20.0 * np.sin(tilt)
    )
    assert float(point["front_vertical_thrust_n"]) == pytest.approx(
        20.0 * np.cos(tilt)
    )
    assert float(point["fixed_vertical_thrust_n"]) == pytest.approx(15.0)
    assert float(point["vertical_force_balance_residual_n"]) == pytest.approx(
        0.0, abs=1e-10
    )


def test_longitudinal_optimizer_vertical_components_match_required_force():
    from scripts.optimize_longitudinal_transition import longitudinal_dynamics

    point = longitudinal_dynamics(
        8.0, 1.5, np.deg2rad(40.0), 25.0, 20.0, np.deg2rad(8.0),
    )
    component_sum = (
        float(point["wing_lift_vertical_n"])
        + float(point["wing_drag_vertical_n"])
        + float(point["tail_vertical_force_n"])
        + float(point["front_vertical_thrust_n"])
        + float(point["fixed_vertical_thrust_n"])
    )

    assert component_sum == pytest.approx(float(point["required_vertical_force_n"]))


def test_longitudinal_optimizer_rejects_pitch_untrimmable_reference_transition():
    from scripts.optimize_longitudinal_transition import solve_transition

    with pytest.raises(RuntimeError, match="pitch-moment balance"):
        solve_transition()


def test_optimizer_schedule_pretilts_before_forward_speed_increase():
    from scripts.optimize_longitudinal_transition import (
        DT_S,
        MAX_TILT_RATE_RAD_S,
        PRE_ACCELERATION_SEGMENTS,
        SEGMENTS,
        SCHEDULE_MIN_FORWARD_TILT_RAD,
        build_transition_schedule_reference,
    )

    time = np.arange(SEGMENTS + 1) * DT_S
    speed_reference, tilt_reference = build_transition_schedule_reference(time)

    np.testing.assert_allclose(
        speed_reference[: PRE_ACCELERATION_SEGMENTS + 1],
        0.0,
        atol=1e-12,
    )
    assert tilt_reference[PRE_ACCELERATION_SEGMENTS] == pytest.approx(
        SCHEDULE_MIN_FORWARD_TILT_RAD
    )
    assert np.all(
        tilt_reference[PRE_ACCELERATION_SEGMENTS:]
        >= SCHEDULE_MIN_FORWARD_TILT_RAD
    )
    assert np.max(np.diff(tilt_reference) / DT_S) <= MAX_TILT_RATE_RAD_S


def test_optimizer_pitch_moment_matches_force_model_without_rotor_wash():
    from scripts.optimize_longitudinal_transition import longitudinal_dynamics
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.configuration.aircraft_parameters import load_aircraft_parameters

    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft)
    speed = 10.0
    alpha = np.deg2rad(8.0)
    tail = np.deg2rad(-3.0)
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([speed, 0.0, 0.0]),
        att=np.array([0.0, alpha, 0.0]),
        tilt=0.0,
    )
    dynamics.step(state, np.zeros(3), 0.0, 0.0, 0.0, 0.0, 0.0, tail, tail, 0.01)
    point = longitudinal_dynamics(speed, 0.0, 0.0, 0.0, 0.0, alpha, tail)

    assert float(point["pitch_moment_n_m"]) == pytest.approx(
        dynamics.last_diagnostics["moment_body"][1], abs=1e-8
    )


def test_optimizer_rotor_pitch_moment_matches_configured_rotor_geometry():
    from scripts.optimize_longitudinal_transition import longitudinal_dynamics
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.configuration.aircraft_parameters import load_aircraft_parameters

    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft)
    tilt = np.deg2rad(35.0)
    front_pair_thrust = 20.0
    rear_pair_thrust = 15.0
    front_omega = np.sqrt(
        front_pair_thrust / (2.0 * aircraft.k_thrust)
    )
    rear_omega = np.sqrt(rear_pair_thrust / (2.0 * aircraft.k_thrust))
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.zeros(3),
        att=np.zeros(3),
        tilt=tilt,
    )
    dynamics.step(
        state,
        np.zeros(3),
        front_omega,
        front_omega,
        rear_omega,
        rear_omega,
        tilt,
        0.0,
        0.0,
        0.01,
    )
    point = longitudinal_dynamics(
        0.0, 0.0, tilt, front_pair_thrust, rear_pair_thrust
    )

    expected_pitch_moment = front_pair_thrust * (0.2 + 0.1 * np.cos(tilt)) - 0.3 * rear_pair_thrust
    assert float(point["rotor_pitch_moment_n_m"]) == pytest.approx(expected_pitch_moment, abs=1e-8)
    assert float(point["rotor_pitch_moment_n_m"]) == pytest.approx(
        dynamics.last_diagnostics["moment_body"][1], abs=1e-8
    )


def test_rotor_speed_output_rejects_thrust_above_configured_rpm_limit():
    from scripts.optimize_longitudinal_transition import (
        AIRCRAFT,
        add_rotor_speed_outputs,
    )

    above_front_limit = 2.0 * AIRCRAFT.max_thrust_per_rotor * 1.01
    with pytest.raises(RuntimeError, match="exceeds the configured"):
        add_rotor_speed_outputs({
            "tilting_pair_thrust_n": np.array([above_front_limit]),
            "fixed_pair_thrust_n": np.array([0.0]),
        })


def test_rotor_speed_outputs_use_assumed_loaded_speed_fraction():
    from scripts.optimize_longitudinal_transition import (
        AIRCRAFT,
        add_rotor_speed_outputs,
    )

    factor = 0.8
    front_limit = AIRCRAFT.estimated_motor_rpm * factor
    front_pair_thrust = (
        2.0 * AIRCRAFT.k_thrust
        * (front_limit * 2.0 * np.pi / 60.0) ** 2
    )
    result = add_rotor_speed_outputs({
        "tilting_pair_thrust_n": np.array([front_pair_thrust]),
        "fixed_pair_thrust_n": np.array([0.0]),
    }, factor)

    assert result["front_rotor_rpm"][0] == pytest.approx(front_limit)
    assert result["front_rpm_limit"][0] == pytest.approx(front_limit)
    assert result["front_rpm_margin"][0] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("factor", [0.0, -0.1, 1.01, float("nan")])
def test_rotor_speed_fraction_rejects_invalid_values(factor):
    from scripts.optimize_longitudinal_transition import solve_transition

    with pytest.raises(ValueError, match="speed limit fraction"):
        solve_transition(factor)


def test_rotor_speed_sweep_marks_insufficient_hover_thrust_infeasible():
    from scripts.rotor_speed_limit_sensitivity import run_sensitivity_sweep

    result = run_sensitivity_sweep((0.5,))

    assert result[0]["status"] == "infeasible_hover_thrust"
    assert not result[0]["feasible"]
    assert "below aircraft weight" in result[0]["failure_reason"]


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


def test_transition_speed_envelope_applies_rear_reserve_to_vertical_support():
    kwargs = dict(
        mass=5.0,
        wing_area=0.6,
        max_thrust_per_rotor=20.0,
        rotor_tilts=[True, True, False, False],
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
    )
    unrestricted = transition_speed_envelope(
        **kwargs, rear_thrust_reserve_fraction=0.0,
    )
    reserved = transition_speed_envelope(
        **kwargs, rear_thrust_reserve_fraction=0.1,
    )

    assert reserved["front_pair_max_thrust_n"][-1] == pytest.approx(40.0)
    assert reserved["rear_pair_max_thrust_n"][-1] == pytest.approx(36.0)
    assert reserved["max_vertical_rotor_thrust_n"][-1] == pytest.approx(36.0)
    assert reserved["wing_lift_required_n"][-1] > unrestricted["wing_lift_required_n"][-1]
    assert reserved["minimum_speed_mps"][-1] > unrestricted["minimum_speed_mps"][-1]
    expected_speed = np.sqrt(
        2.0 * (5.0 * 9.81 - 36.0) / (1.225 * 0.6 * 1.1)
    )
    assert reserved["minimum_speed_mps"][-1] == pytest.approx(expected_speed)


def test_wing_lift_and_thrust_sensitivity_sweep_returns_envelopes():
    from scripts.wing_lift_sensitivity import wing_lift_thrust_sensitivity
    from src.configuration.aircraft_parameters import load_aircraft_parameters

    aircraft = load_aircraft_parameters()
    result = wing_lift_thrust_sensitivity(
        aircraft,
        lift_scales=(0.8, 1.2),
        thrust_scales=(0.4, 1.0),
        samples=5,
    )

    assert result["minimum_speed_mps"].shape == (2, 2, 5)
    assert result["feasible"].shape == (2, 2, 5)
    assert result["tilt_deg"][-1] == pytest.approx(90.0)
    assert result["minimum_speed_mps"][0, 0, -1] > 0.0
    assert result["minimum_speed_mps"][1, 0, -1] < (
        result["minimum_speed_mps"][0, 0, -1]
    )
    assert result["minimum_speed_mps"][0, 1, -1] == pytest.approx(0.0)


def test_transition_speed_envelope_reads_reserve_from_aircraft_config(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        transition_module,
        "load_aircraft_parameters",
        lambda: SimpleNamespace(rear_thrust_reserve_fraction=0.25),
    )
    envelope = transition_speed_envelope(
        mass=5.0,
        wing_area=0.6,
        max_thrust_per_rotor=20.0,
        rotor_tilts=[True, True, False, False],
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
        samples=2,
    )

    assert envelope["rear_thrust_reserve_fraction"][-1] == pytest.approx(0.25)
    assert envelope["rear_pair_max_thrust_n"][-1] == pytest.approx(30.0)


def test_rear_thrust_sensitivity_sweep_crosses_full_tilt_support_threshold():
    from src.configuration.aircraft_parameters import load_aircraft_parameters

    aircraft = load_aircraft_parameters()
    result = rear_thrust_sensitivity_sweep(aircraft)
    thrust = result["max_thrust_per_rotor_n"]
    speed = result["minimum_speed_mps"][:, -1]
    margin = result["vertical_margin_at_zero_speed_n"][:, -1]
    threshold = result["rear_only_hover_threshold_per_rotor_n"]
    threshold_index = np.argmin(np.abs(thrust - threshold))

    assert thrust[threshold_index] == pytest.approx(threshold)
    assert margin[threshold_index] == pytest.approx(0.0, abs=1e-10)
    assert speed[threshold_index] == pytest.approx(0.0)
    assert margin[threshold_index - 1] < 0.0
    assert speed[threshold_index - 1] > 0.0
    assert margin[threshold_index + 1] > 0.0
    assert speed[threshold_index + 1] == pytest.approx(0.0)
    assert np.any(np.isclose(thrust, aircraft.max_thrust_per_rotor))


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


def test_coordinate_derived_vtail_geometry_trims_10mps_45deg_without_extra_travel():
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.configuration.aircraft_parameters import load_aircraft_parameters

    aircraft = load_aircraft_parameters()
    trim = solve_longitudinal_trim(
        build_aircraft_dynamics(aircraft),
        airspeed_mps=10.0,
        tilt_rad=np.deg2rad(45.0),
        tail_limit_rad=aircraft.vtail_max_deflection_rad,
    )

    assert trim["feasible"]
    assert abs(trim["tail_deflection_deg"]) <= aircraft.vtail_max_deflection_deg
    assert trim["max_normalized_residual"] < 0.02


def test_longitudinal_trim_uses_residual_tolerance_even_if_solver_stops_early(monkeypatch):
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.configuration.aircraft_parameters import load_aircraft_parameters

    least_squares = transition_module.least_squares

    def report_early_stop(*args, **kwargs):
        result = least_squares(*args, **kwargs)
        result.success = False
        result.message = "forced early-stop diagnostic"
        return result

    monkeypatch.setattr(transition_module, "least_squares", report_early_stop)
    dynamics = build_aircraft_dynamics(load_aircraft_parameters())
    trim = solve_longitudinal_trim(
        dynamics,
        airspeed_mps=10.0,
        tilt_rad=np.deg2rad(45.0),
    )

    assert not trim["solver_converged"]
    assert trim["max_normalized_residual"] <= 0.02
    assert trim["feasible"]


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
    assert not result["cl_beta_assessed"].any()
    assert not result["static_stability_assessed_mask"].any()
    assert not result["static_stability_mask"].any()
    assert np.all(
        result["static_stability_status"][np.isfinite(result["cm_alpha_per_rad"])]
        == "unassessed"
    )
    np.testing.assert_array_equal(result["trim_feasible"][1], [True, False])
    np.testing.assert_array_equal(result["assessment_mask"][1], [True, False])
    assert result["trim_motor_rpm"].shape == (2, 2)
    assert result["trim_alpha_deg"].shape == (2, 2)
    assert result["trim_pitch_moment_residual"].shape == (2, 2)
    assert result["trim_longitudinal_residual"].shape == (2, 2)
    assert result["trim_vertical_residual"].shape == (2, 2)
    assert result["trim_tail_deflection_deg"].shape == (2, 2)
    assert result["tail_authority_margin_deg"].shape == (2, 2)
    assert np.all(
        result["tail_authority_margin_deg"]
        >= -1e-8
    )


def test_static_stability_status_preserves_known_failures_and_unknown_roll():
    assessment = np.ones(3, dtype=bool)
    cm_alpha = np.array([-1.0, 0.1, -1.0])
    cl_beta = np.array([0.0, 0.0, 0.1])
    cn_beta = np.ones(3)
    cl_beta_assessed = np.array([False, False, True])

    known_pass, assessed, full_pass, status = (
        transition_module._static_stability_masks(
            assessment,
            cm_alpha,
            cl_beta,
            cn_beta,
            cl_beta_assessed,
        )
    )

    np.testing.assert_array_equal(known_pass, [True, False, False])
    np.testing.assert_array_equal(assessed, [False, False, True])
    np.testing.assert_array_equal(full_pass, [False, False, False])
    np.testing.assert_array_equal(status, ["unassessed", "fail", "fail"])


def test_point_screen_aggregates_pass_fail_and_incomplete_reasons():
    statuses, reasons = transition_module._point_screen_statuses(
        analytic_mask=np.ones(4, dtype=bool),
        gust_mask=np.ones(4, dtype=bool),
        trim_mask=np.array([True, False, True, True]),
        tail_authority_mask=np.ones(4, dtype=bool),
        rotor_rpm_mask=np.ones(4, dtype=bool),
        rotor_thrust_mask=np.ones(4, dtype=bool),
        static_stability_assessed_mask=np.array([True, False, False, True]),
        static_stability_mask=np.array([True, False, False, True]),
        static_stability_status=np.array(
            ["pass", "unassessed", "unassessed", "pass"],
            dtype=object,
        ),
        electrical_assessed_mask=np.array([True, False, False, True]),
        electrical_feasible_mask=np.array([True, False, False, False]),
    )

    np.testing.assert_array_equal(statuses, ["pass", "fail", "incomplete", "fail"])
    assert reasons[0] == ""
    assert reasons[1] == "point_trim_infeasible"
    assert reasons[2] == "static_stability_unassessed;electrical_unassessed"
    assert reasons[3] == "electrical_fail"


def test_longitudinal_trim_uses_configured_tail_travel_when_unspecified(monkeypatch):
    from types import SimpleNamespace

    limit_deg = 4.0
    monkeypatch.setattr(
        transition_module,
        "load_aircraft_parameters",
        lambda: SimpleNamespace(vtail_max_deflection_rad=np.deg2rad(limit_deg)),
    )
    trim = solve_longitudinal_trim(
        make_dynamics(), airspeed_mps=8.0, tilt_rad=0.0,
    )

    assert trim["tail_deflection_limit_deg"] == pytest.approx(limit_deg)
    assert abs(trim["tail_deflection_deg"]) <= limit_deg + 1e-8
    assert trim["tail_authority_margin_deg"] == pytest.approx(
        limit_deg - abs(trim["tail_deflection_deg"])
    )


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
        & result["tail_authority_mask"]
        & result["rotor_thrust_mask"]
        & result["pitch_moment_mask"]
        & result["known_static_stability_mask"]
        & actuator_mask
    )
    assert np.array_equal(result["known_constraints_mask"], expected_known_mask)
    assert "propeller_torque_coefficient_nm_s2_rad2" in (
        result["electrical_missing_parameters"]
    )


def test_unassessed_cl_beta_does_not_fail_known_constraints_or_qualify_corridor(
    monkeypatch,
):
    shape = (1, 2)
    stability = {
        "trim_feasible": np.ones(shape, dtype=bool),
        "trim_pitch_moment_residual": np.zeros(shape),
        "trim_tail_deflection_deg": np.zeros(shape),
        "tail_authority_margin_deg": np.ones(shape),
        "assessment_mask": np.ones(shape, dtype=bool),
        "cm_alpha_per_rad": -np.ones(shape),
        "cl_beta_per_rad": np.zeros(shape),
        "cn_beta_per_rad": np.ones(shape),
        "cl_beta_assessed": np.zeros(shape, dtype=bool),
        "known_static_stability_mask": np.ones(shape, dtype=bool),
        "static_stability_assessed_mask": np.zeros(shape, dtype=bool),
        "static_stability_mask": np.zeros(shape, dtype=bool),
        "static_stability_status": np.full(shape, "unassessed", dtype=object),
        "trim_front_motor_rpm": np.full(shape, 1000.0),
        "trim_rear_motor_rpm": np.full(shape, 1000.0),
    }
    monkeypatch.setattr(
        transition_module,
        "transition_stability_map",
        lambda *args, **kwargs: stability,
    )
    monkeypatch.setattr(transition_module, "transition_speed_envelope", lambda **kwargs: {
        "tilt_deg": np.array([0.0, 90.0]),
        "minimum_speed_mps": np.array([0.0, 0.0]),
        "maximum_speed_mps": np.array([20.0, 20.0]),
        "max_vertical_rotor_thrust_n": np.array([80.0, 80.0]),
        "feasible": np.array([True, True]),
    })

    result = constrained_transition_corridor(
        make_dynamics(),
        max_thrust_per_rotor=20.0,
        rotor_tilts=[True, True, False, False],
        airspeeds_mps=[10.0],
        tilt_angles_deg=[0.0, 90.0],
        max_tilt_rate_deg_s=10.0,
        transition_duration_s=20.0,
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
    )

    assert result["known_static_stability_mask"].all()
    assert not result["static_stability_assessed_mask"].any()
    assert np.all(result["static_stability_status"] == "unassessed")
    assert result["known_constraints_mask"].all()
    assert not result["qualified_mask"].any()
    assert np.all(result["point_screen_status"] == "incomplete")
    assert all(
        "static_stability_unassessed" in reason
        for reason in result["point_screen_reasons"].flat
    )


def test_corridor_screens_slew_and_electrical_demand_for_each_pair(monkeypatch):
    shape = (2, 2)
    stability = {
        "trim_feasible": np.ones(shape, dtype=bool),
        "trim_pitch_moment_residual": np.zeros(shape),
        "trim_tail_deflection_deg": np.zeros(shape),
        "tail_authority_margin_deg": np.ones(shape),
        "assessment_mask": np.ones(shape, dtype=bool),
        "cm_alpha_per_rad": -np.ones(shape),
        "cl_beta_per_rad": -np.ones(shape),
        "cn_beta_per_rad": np.ones(shape),
        "cl_beta_assessed": np.ones(shape, dtype=bool),
        "known_static_stability_mask": np.ones(shape, dtype=bool),
        "static_stability_assessed_mask": np.ones(shape, dtype=bool),
        "static_stability_mask": np.ones(shape, dtype=bool),
        "static_stability_status": np.full(shape, "pass", dtype=object),
        "trim_front_motor_rpm": np.array([[1000.0, 1000.0], [1000.0, 500.0]]),
        "trim_rear_motor_rpm": np.array([[1000.0, 1000.0], [1000.0, 1500.0]]),
    }
    monkeypatch.setattr(transition_module, "transition_stability_map", lambda *a, **k: stability)
    monkeypatch.setattr(transition_module, "transition_speed_envelope", lambda **k: {
        "tilt_deg": np.array([0.0, 90.0]),
        "minimum_speed_mps": np.array([0.0, 0.0]),
        "maximum_speed_mps": np.array([30.0, 30.0]),
        "max_vertical_rotor_thrust_n": np.array([80.0, 40.0]),
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
    assert result["point_screen_status"][1, 1] == "pass"
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


def test_corridor_enforces_rear_thrust_reserve_separately(monkeypatch):
    dynamics = make_dynamics()
    shape = (2, 2)
    max_rear_thrust_n = 20.0 * (1.0 - 0.1)
    rear_at_limit_rpm = np.sqrt(max_rear_thrust_n / dynamics.rotor.k_thrust) * 60 / (2 * np.pi)
    rear_over_limit_rpm = np.sqrt(19.0 / dynamics.rotor.k_thrust) * 60 / (2 * np.pi)
    stability = {
        "trim_feasible": np.ones(shape, dtype=bool),
        "trim_pitch_moment_residual": np.zeros(shape),
        "trim_tail_deflection_deg": np.zeros(shape),
        "tail_authority_margin_deg": np.ones(shape),
        "assessment_mask": np.ones(shape, dtype=bool),
        "cm_alpha_per_rad": -np.ones(shape),
        "cl_beta_per_rad": -np.ones(shape),
        "cn_beta_per_rad": np.ones(shape),
        "cl_beta_assessed": np.ones(shape, dtype=bool),
        "known_static_stability_mask": np.ones(shape, dtype=bool),
        "static_stability_assessed_mask": np.ones(shape, dtype=bool),
        "static_stability_mask": np.ones(shape, dtype=bool),
        "static_stability_status": np.full(shape, "pass", dtype=object),
        "trim_front_motor_rpm": np.full(shape, 1000.0),
        "trim_rear_motor_rpm": np.array([
            [rear_at_limit_rpm, rear_over_limit_rpm],
            [rear_at_limit_rpm, rear_at_limit_rpm],
        ]),
    }
    monkeypatch.setattr(transition_module, "transition_stability_map", lambda *a, **k: stability)
    result = constrained_transition_corridor(
        dynamics,
        max_thrust_per_rotor=20.0,
        rotor_tilts=[True, True, False, False],
        airspeeds_mps=[0.0, 10.0],
        tilt_angles_deg=[0.0, 90.0],
        max_tilt_rate_deg_s=10.0,
        transition_duration_s=20.0,
        cl_max=1.1,
        cd_0=0.02,
        aspect_ratio=6.0,
    )

    assert result["front_thrust_mask"].all()
    assert not result["rear_thrust_mask"][0, 1]
    assert result["rear_thrust_mask"][1, 1]
    assert not result["rotor_thrust_mask"][0, 1]
    assert result["rear_thrust_margin_per_rotor_n"][0, 1] == pytest.approx(-1.0)
    assert result["trim_rear_rpm_margin"][0, 1] == pytest.approx(
        rear_at_limit_rpm - rear_over_limit_rpm
    )
    assert result["rear_rpm_mask"][0, 0]
    assert not result["rear_rpm_mask"][0, 1]
    assert not result["vertical_support_mask"][0, 1]
    assert result["vertical_support_mask"][1, 1]
