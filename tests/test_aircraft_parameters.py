from pathlib import Path

import numpy as np
import pytest
import yaml

from src.configuration.aircraft_parameters import (
    DEFAULT_AIRCRAFT_PATH,
    load_aircraft_parameters,
)
from src.dynamics.transition import transition_speed_envelope


def write_config(tmp_path: Path, **changes) -> Path:
    config = yaml.safe_load(DEFAULT_AIRCRAFT_PATH.read_text(encoding="utf-8"))
    config.update(changes)
    path = tmp_path / "aircraft.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_motor_speed_tracks_series_cells_and_kv(tmp_path):
    aircraft = load_aircraft_parameters(write_config(
        tmp_path, battery_cells=4, cell_voltage_v=3.7,
        motor_kv_rpm_per_volt=1000, motor_speed_factor=0.8, k_thrust=1e-5,
    ))
    assert aircraft.battery_voltage_v == pytest.approx(14.8)
    assert aircraft.estimated_motor_rpm == pytest.approx(11840)
    assert aircraft.max_omega == pytest.approx(11840 * 2 * np.pi / 60)
    assert aircraft.max_thrust_per_rotor == pytest.approx(1e-5 * aircraft.max_omega**2)


def test_electrical_inputs_are_optional_and_map_to_existing_screen(tmp_path):
    from src.dynamics.transition import assess_rotor_electrical_point

    optional_fields = (
        "motor_resistance_ohm",
        "motor_no_load_current_a",
        "motor_continuous_current_a",
        "propeller_torque_coefficient_nm_s2_rad2",
        "battery_internal_resistance_ohm",
        "battery_capacity_ah",
        "battery_c_rating",
        "battery_continuous_current_a",
        "esc_efficiency",
        "esc_continuous_current_a",
    )
    defaults = load_aircraft_parameters(write_config(
        tmp_path, **{name: None for name in optional_fields},
    ))
    assert defaults.electrical_parameters["battery_nominal_voltage_v"] == pytest.approx(
        defaults.battery_cells * defaults.cell_voltage_v
    )
    assert defaults.electrical_parameters["motor_resistance_ohm"] is None
    missing = assess_rotor_electrical_point([1000.0] * 4, defaults.electrical_parameters)
    assert not missing["assessed"]
    assert "motor_resistance_ohm" in missing["missing_parameters"]

    aircraft = load_aircraft_parameters(write_config(
        tmp_path,
        motor_resistance_ohm=0.05,
        motor_no_load_current_a=1.0,
        motor_continuous_current_a=50.0,
        propeller_torque_coefficient_nm_s2_rad2=1.0e-5,
        battery_internal_resistance_ohm=0.05,
        battery_capacity_ah=5.0,
        battery_c_rating=30.0,
        battery_continuous_current_a=150.0,
        esc_efficiency=0.95,
        esc_continuous_current_a=60.0,
    ))
    assessed = assess_rotor_electrical_point(
        [1000.0] * 4, aircraft.electrical_parameters,
    )
    assert assessed["assessed"]
    assert assessed["feasible"]


@pytest.mark.parametrize("changes, message", [
    ({"motor_resistance_ohm": -0.01}, "motor_resistance_ohm"),
    ({"motor_continuous_current_a": 0.0}, "motor_continuous_current_a"),
    ({"battery_capacity_ah": "five"}, "battery_capacity_ah"),
    ({"esc_efficiency": 1.1}, "esc_efficiency"),
])
def test_invalid_optional_electrical_inputs_are_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_aircraft_parameters(write_config(tmp_path, **changes))


def test_wing_area_and_aspect_ratio_are_derived_from_planform(tmp_path):
    baseline = load_aircraft_parameters()
    assert baseline.wing_mean_chord_m == pytest.approx(0.3111111111)
    assert baseline.wing_aerodynamic_center_x_m == pytest.approx(0.0, abs=1e-10)

    aircraft = load_aircraft_parameters(write_config(
        tmp_path, wingspan=4.0, root_chord=0.5, tip_chord=0.3,
        wing_root_le_x_m=0.2, wing_tip_le_x_m=0.15,
    ))
    assert aircraft.wing_area == pytest.approx(1.6)
    assert aircraft.wing_aspect_ratio == pytest.approx(10.0)
    assert aircraft.wing_mean_chord_m == pytest.approx(0.4083333333)
    assert aircraft.wing_aerodynamic_center_x_m == pytest.approx(0.075)

    from src.configuration.aircraft_models import build_aircraft_dynamics

    dynamics = build_aircraft_dynamics(aircraft)
    assert dynamics.aero.wing_area == pytest.approx(1.6)
    assert dynamics.aero.AR == pytest.approx(10.0)
    assert dynamics.aero.chord == pytest.approx(aircraft.wing_mean_chord_m)
    assert dynamics.aero.x_ac == pytest.approx(aircraft.wing_aerodynamic_center_x_m)


def test_vtail_control_travel_is_loaded_from_aircraft_configuration(tmp_path):
    aircraft = load_aircraft_parameters(write_config(
        tmp_path, vtail_max_deflection_deg=8.0,
    ))
    assert aircraft.vtail_max_deflection_rad == pytest.approx(np.deg2rad(8.0))


def test_vtail_dihedral_and_arm_are_derived_from_dimensioned_geometry(tmp_path):
    baseline = load_aircraft_parameters()
    assert baseline.vtail_tip_chord_m == pytest.approx(0.1)
    assert baseline.vtail_dihedral_deg == pytest.approx(26.5650511771)
    assert baseline.vtail_arm_length == pytest.approx(0.5944444444)

    from src.configuration.aircraft_models import build_aircraft_dynamics

    dynamics = build_aircraft_dynamics(baseline)
    assert dynamics.aero.vtail.dihedral == pytest.approx(
        np.deg2rad(baseline.vtail_dihedral_deg)
    )
    assert dynamics.aero.vtail.arm_length == pytest.approx(
        baseline.vtail_arm_length
    )

    changed = load_aircraft_parameters(write_config(
        tmp_path, vtail_tip_z_m=-0.3, vtail_tip_le_x_m=-0.4,
    ))
    assert changed.vtail_dihedral_deg == pytest.approx(45.0)
    assert changed.vtail_arm_length < baseline.vtail_arm_length


@pytest.mark.parametrize("changes, message", [
    ({"battery_cells": 0}, "battery_cells"),
    ({"motor_speed_factor": 1.1}, "motor_speed_factor"),
    ({"rear_thrust_reserve_fraction": 1.0}, "rear_thrust_reserve_fraction"),
    ({"rear_thrust_reserve_fraction": -0.1}, "rear_thrust_reserve_fraction"),
    ({"mass": float("nan")}, "mass"),
    ({"motor_positions": [[0, 0, 0, 1]]}, "motor position"),
    ({"vtail_max_deflection_deg": 91.0}, "vtail_max_deflection_deg"),
    ({"vtail_arm_length": 0.6}, "unknown"),
    ({"vtail_dihedral_deg": 30.0}, "unknown"),
    ({"vtail_tip_chord_fraction": 0.0}, "vtail_tip_chord_fraction"),
    ({"vtail_aerodynamic_center_fraction_chord": 1.1}, "vtail_aerodynamic_center_fraction_chord"),
    ({"wing_aerodynamic_center_fraction_mac": 1.1}, "wing_aerodynamic_center_fraction_mac"),
    ({"wing_area": 1.5}, "unknown"),
    ({"extra_parameter": 1}, "unknown"),
])
def test_invalid_aircraft_inputs_are_rejected(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_aircraft_parameters(write_config(tmp_path, **changes))


def test_changes_to_aircraft_file_change_envelope(tmp_path):
    def boundary(cells):
        aircraft = load_aircraft_parameters(write_config(tmp_path, battery_cells=cells))
        envelope = transition_speed_envelope(
            aircraft.mass, aircraft.wing_area, aircraft.max_thrust_per_rotor,
            [position[3] for position in aircraft.motor_positions],
            rho=aircraft.air_density, cl_max=1.0, cd_0=aircraft.wing_drag_coeff,
            aspect_ratio=aircraft.wing_aspect_ratio, gravity=aircraft.gravity,
            rear_thrust_reserve_fraction=aircraft.rear_thrust_reserve_fraction,
        )
        return envelope["minimum_speed_mps"][-1]

    assert boundary(4) > boundary(6)


def test_configured_gravity_changes_envelope():
    kwargs = dict(
        mass=5.0, wing_area=1.5, max_thrust_per_rotor=5.0,
        rotor_tilts=[True, True, False, False], cl_max=1.0,
        cd_0=0.02, aspect_ratio=6.0,
    )
    earth = transition_speed_envelope(**kwargs, gravity=9.81)
    reduced = transition_speed_envelope(**kwargs, gravity=8.0)
    assert earth["minimum_speed_mps"][-1] > reduced["minimum_speed_mps"][-1]


def test_force_analysis_uses_configured_mass_rotors_and_battery(tmp_path, monkeypatch):
    from scripts import plot_force_model_curves as analysis

    positions = [
        [0.8, -0.5, 0.0, True], [0.8, 0.5, 0.0, True],
        [-0.3, -0.4, 0.0, False], [-0.3, 0.4, 0.0, False],
    ]
    aircraft = load_aircraft_parameters(write_config(
        tmp_path, mass=4.0, battery_cells=4, motor_positions=positions,
    ))
    monkeypatch.setattr(analysis, "load_aircraft_parameters", lambda: aircraft)
    dynamics, omega, loaded = analysis.build_dynamics()
    assert dynamics.mass == 4.0
    assert dynamics.rotor.max_omega == pytest.approx(aircraft.max_omega)
    assert loaded.motor_positions[0][0] == 0.8
    result = analysis.evaluate_point(dynamics, omega, 0.0, 0.0, loaded)
    assert result["total_up_n"] == pytest.approx(4.0 * aircraft.gravity)
    original = load_aircraft_parameters()
    original_result = analysis.evaluate_point(dynamics, omega, 0.0, 0.0, original)
    assert result["pitch_moment_nm"] != pytest.approx(original_result["pitch_moment_nm"])


def test_configured_four_rotor_model_keeps_rear_thrust_at_full_tilt(tmp_path):
    from src.configuration.aircraft_models import build_aircraft_dynamics

    aircraft = load_aircraft_parameters(write_config(
        tmp_path, motor_positions=[
            [0.7, -0.4, 0.0, True], [0.7, 0.4, 0.0, True],
            [-0.5, -0.4, 0.0, False], [-0.5, 0.4, 0.0, False],
        ],
    ))
    dynamics = build_aircraft_dynamics(aircraft)
    omega = aircraft.max_omega * 0.5
    force, moment, _ = dynamics._rotor_forces_and_moments(
        np.full(4, omega), np.pi / 2, 100.0,
    )
    thrust = aircraft.k_thrust * omega**2
    assert force[0] == pytest.approx(2 * thrust)
    assert force[2] == pytest.approx(-2 * thrust)
    assert moment[1] == pytest.approx(0.2 * thrust)


def test_baseline_four_rotor_hover_balances_weight_and_moments():
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.dynamics.nacelle import deflection_rad_from_tilt_deg
    from src.dynamics.state import AircraftState

    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft)
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))
    omega = np.sqrt(aircraft.mass * aircraft.gravity / (4.0 * aircraft.k_thrust))
    hover = deflection_rad_from_tilt_deg(90.0)
    updated = dynamics.step(state, np.zeros(3), *([omega] * 4), hover, 0.0, 0.0, 0.01)

    np.testing.assert_allclose(dynamics.last_diagnostics["force_body"], [0, 0, -49.05])
    np.testing.assert_allclose(updated.vel, 0, atol=1e-12)
    np.testing.assert_allclose(updated.omega, 0, atol=1e-12)


def test_baseline_full_tilt_cruise_balances_forces_and_moments():
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.dynamics.nacelle import deflection_rad_from_tilt_deg
    from src.dynamics.state import AircraftState

    aircraft = load_aircraft_parameters()
    dynamics = build_aircraft_dynamics(aircraft)
    tilt = deflection_rad_from_tilt_deg(0.0)
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]), vel=np.array([30.0, 0.0, 0.0]), tilt=tilt,
    )
    # Independently balanced at zero pitch: front thrust equals drag,
    # rear thrust equals weight minus net lift, and pitch moments sum to zero.
    front_pair_thrust, rear_pair_thrust = 11.63197801311935, 18.276670142506823
    front_omega, rear_omega = np.sqrt(
        np.array([front_pair_thrust, rear_pair_thrust]) / (2.0 * aircraft.k_thrust)
    )
    tail_deflection = np.deg2rad(-6.718572724531241)
    assert max(front_omega, rear_omega) < aircraft.max_omega
    assert abs(tail_deflection) < aircraft.vtail_max_deflection_rad

    updated = dynamics.step(
        state, np.zeros(3), front_omega, front_omega, rear_omega, rear_omega,
        tilt, tail_deflection, tail_deflection, 0.01,
    )

    np.testing.assert_allclose(dynamics.last_diagnostics["acceleration_ned"], 0, atol=1e-10)
    np.testing.assert_allclose(dynamics.last_diagnostics["moment_body"], 0, atol=1e-10)
    np.testing.assert_allclose(updated.vel, state.vel, atol=1e-12)
    np.testing.assert_allclose(updated.att, state.att, atol=1e-12)


def test_upgraded_closed_loop_reads_configuration(tmp_path, monkeypatch):
    from scripts import closed_loop_transition
    aircraft = load_aircraft_parameters(write_config(tmp_path, battery_cells=2))
    monkeypatch.setattr(closed_loop_transition, "load_aircraft_parameters", lambda: aircraft)
    result = closed_loop_transition.run_transition(duration=0.02, dt=0.01)
    assert np.all(result["omega"] <= aircraft.max_omega)
    assert result["omega"][0] == pytest.approx(aircraft.max_omega)


def test_configured_gravity_and_aerodynamic_parameters_reach_models(tmp_path):
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.dynamics.state import AircraftState

    aircraft = load_aircraft_parameters(write_config(
        tmp_path, gravity=8.0, wing_pitch_moment_slope=-0.7,
        vtail_lift_slope=2.8, oswald_efficiency=0.7,
    ))
    dynamics = build_aircraft_dynamics(aircraft)
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))
    dynamics.step(state, np.zeros(3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01)
    assert dynamics.last_diagnostics["acceleration_ned"][2] == pytest.approx(8.0)
    assert dynamics.aero.Cm_alpha == -0.7
    assert dynamics.aero.oswald_efficiency == 0.7
    assert dynamics.aero.vtail.CL_alpha_tail == 2.8


def test_reduced_corridor_respects_fixed_rotors_and_battery_limit(tmp_path):
    from src.dynamics.reduced_transition import altitude_acceleration

    weak = load_aircraft_parameters(write_config(tmp_path, battery_cells=2))
    hover = float(altitude_acceleration(0.0, weak))
    assert hover == pytest.approx(
        (4 * weak.max_thrust_per_rotor - weak.mass * weak.gravity) / weak.mass
    )
    wing_lift = 0.5 * weak.air_density * weak.wing_area * 0.5 * 12.0**2
    cruise = float(altitude_acceleration(np.pi / 2, weak))
    assert cruise == pytest.approx(
        (2 * weak.max_thrust_per_rotor + wing_lift - weak.mass * weak.gravity) / weak.mass
    )


def test_geometry_uses_configured_span_motor_positions_and_pivot(tmp_path, monkeypatch):
    import src.dynamics.cad_geometry as geometry

    aircraft = load_aircraft_parameters(write_config(
        tmp_path, wingspan=4.0, wing_tip_le_x_m=0.25,
        tilt_pivot_xz_m=[0.0, 0.0],
        motor_positions=[
            [1.0, -0.5, 0.0, True], [1.0, 0.5, 0.0, True],
            [-0.5, -0.5, 0.0, False], [-0.5, 0.5, 0.0, False],
        ],
    ))
    monkeypatch.setattr(geometry, "load_aircraft_parameters", lambda: aircraft)
    cad = geometry.CadGeometry()
    assert cad.offsets["wing_right"]["tip_le"] == [0.25, 2.0, 0.0]
    np.testing.assert_allclose(cad.calculate_tilted_motors(0)[0]["pos"], [0.0, -0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(cad.calculate_tilted_motors(90)[0]["thrust_dir"], [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(cad.calculate_tilted_motors(0)[0]["thrust_dir"], [1, 0, 0], atol=1e-12)
    wing = cad.offsets["wing_right"]
    wing_properties = cad.compute_wing_properties(
        wing["root_le"], wing["root_te"], wing["tip_le"], wing["tip_te"],
    )
    assert wing_properties["ac_x"] == pytest.approx(aircraft.wing_aerodynamic_center_x_m)


def test_reduced_corridor_does_not_return_failed_solver_iterate(monkeypatch):
    import casadi
    from scripts.robust_corridor import solve_corridor

    def fail(_self):
        raise RuntimeError("solver failed")

    monkeypatch.setattr(casadi.Opti, "solve", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        solve_corridor(False)


def test_independent_simulation_uses_configured_four_rotors(tmp_path):
    from scripts.sil_crosscheck import IndependentDynamics
    from src.configuration.aircraft_models import build_aircraft_dynamics
    from src.dynamics.state import AircraftState

    aircraft = load_aircraft_parameters(write_config(tmp_path, gravity=8.0))
    independent = IndependentDynamics(aircraft)
    independent.pos = np.array([0.0, 0.0, -100.0])
    dynamics = build_aircraft_dynamics(aircraft)
    dynamics.aero = None
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))
    omega, tilt, dt = 800.0, np.pi / 4, 0.001
    state = dynamics.step(state, np.zeros(3), omega, omega, omega, omega, tilt, 0.0, 0.0, dt)
    independent.step(omega, tilt, 0.0, dt)
    np.testing.assert_allclose(independent.vel, state.vel, atol=1e-12)
    np.testing.assert_allclose(independent.omega, state.omega, atol=1e-12)


def test_sil_crosscheck_runs_both_configured_simulations():
    from scripts.sil_crosscheck import run_independent_sim, run_main_sim
    from src.configuration.aircraft_models import build_collective_controller
    from src.control.trajectory import LinearReference

    aircraft = load_aircraft_parameters()
    reference = LinearReference(100, 12, 10)
    for run in (run_main_sim, run_independent_sim):
        log = run(0.05, 0.01, build_collective_controller(aircraft), reference)
        assert len(log["t"]) == 5
        assert np.isfinite(log["alt"]).all()


def test_finalize_stops_when_test_suite_fails(monkeypatch):
    from types import SimpleNamespace
    from scripts import finalize

    monkeypatch.setattr(finalize.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="tests failed", stderr=""))
    with pytest.raises(RuntimeError, match="Test suite failed"):
        finalize.main()


@pytest.mark.parametrize("module_name, function_name, args", [
    ("scripts.compare_indi_dob", "run_one", (0.0, 1, "pid")),
    ("scripts.monte_carlo_gusts", "run_one", (0.0, 1, "pid")),
    ("scripts.statistical_comparison", "run_one", (0.0, 1, "pid")),
])
def test_wind_studies_run_with_configured_aircraft(module_name, function_name, args):
    from importlib import import_module

    result = getattr(import_module(module_name), function_name)(*args, dt=1.0)
    assert np.isfinite(result["max_alt_loss"])
