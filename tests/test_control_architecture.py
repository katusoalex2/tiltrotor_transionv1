import numpy as np

from src.allocation.allocator import ControlAllocator
from src.control.architecture import ControlLawArchitecture
from src.control.pid import PIDController
from src.control.scheduler import GainScheduler
from src.aero.aerodynamics import AeroModel
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel


MOTOR_POSITIONS = (
    (0.3, 0.4, 0.0, True),
    (0.3, -0.4, 0.0, True),
    (-0.3, 0.4, 0.0, False),
    (-0.3, -0.4, 0.0, False),
)


def make_controller():
    return ControlLawArchitecture(
        mass=5.0,
        k_thrust=1e-5,
        max_omega=3000.0,
        motor_positions=MOTOR_POSITIONS,
        vtail_model=VtailModel(area=0.1, arm_length=0.5),
    )


def test_allocation_matrix_matches_rotor_geometry_and_tail_derivatives():
    allocator = ControlAllocator(motor_positions=MOTOR_POSITIONS)
    tail_effectiveness = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    matrix = allocator.allocation_matrix(tilt_cmd=0.0, tail_effectiveness=tail_effectiveness)

    np.testing.assert_allclose(matrix[0, :4], np.ones(4))
    np.testing.assert_allclose(matrix[1:3, :4], [
        [0.4, -0.4, 0.4, -0.4],
        [0.3, 0.3, -0.3, -0.3],
    ])
    np.testing.assert_allclose(matrix[1:, 4:], tail_effectiveness)


def test_allocator_tracks_hover_collective_and_bounds_actuators():
    allocator = ControlAllocator(k_thrust=1e-5, max_omega=3000.0, motor_positions=MOTOR_POSITIONS)

    command = allocator.allocate(total_thrust=5.0 * 9.81)

    rotor_omegas = np.array([
        command["omega_fl"],
        command["omega_fr"],
        command["omega_rl"],
        command["omega_rr"],
    ])
    assert np.all((rotor_omegas >= 0.0) & (rotor_omegas <= 3000.0))
    assert np.isclose(1e-5 * np.sum(rotor_omegas**2), 5.0 * 9.81, rtol=1e-5)
    assert np.isclose(command["delta_left"], 0.0, atol=1e-8)
    assert np.isclose(command["delta_right"], 0.0, atol=1e-8)
    assert not command["allocation_saturated"]


def test_allocator_reports_unachievable_hover_yaw_request():
    allocator = ControlAllocator(motor_positions=MOTOR_POSITIONS)

    command = allocator.allocate(total_thrust=5.0 * 9.81, yaw_moment=1.0)

    assert command["allocation_saturated"]
    assert np.isclose(command["allocation_residual"][3], 1.0)


def test_allocator_pitch_priority_trades_collective_for_pitch_authority():
    allocator = ControlAllocator(
        k_thrust=1e-5,
        max_omega=3000.0,
        motor_positions=MOTOR_POSITIONS,
    )
    request = {
        "total_thrust": 25.0,
        "pitch_moment": 2.0,
        "tilt_cmd": 0.0,
        "tail_effectiveness": np.zeros((3, 2)),
        "moment_bias": np.array([0.0, -6.0, 0.0]),
    }

    balanced = allocator.allocate(**request)
    pitch_priority = allocator.allocate(
        **request,
        axis_weights=np.array([1.0, 1.0, 3.0, 1.0]),
    )

    assert abs(pitch_priority["allocation_residual"][2]) < abs(
        balanced["allocation_residual"][2]
    )
    assert abs(pitch_priority["allocation_residual"][0]) > abs(
        balanced["allocation_residual"][0]
    )


def test_gain_scheduler_blends_using_speed_and_tilt():
    scheduler = GainScheduler(
        hover_gains={"kp": 2.0, "ki": 1.0, "kd": 0.5},
        cruise_gains={"kp": 0.5, "ki": 0.2, "kd": 0.1},
    )

    assert scheduler.transition_blend(0.0, 0.0) == 0.0
    assert scheduler.get_gains(0.0, 0.0)["kp"] == 2.0
    assert scheduler.transition_blend(20.0, np.pi / 2.0) == 1.0
    assert scheduler.get_gains(10.0, np.pi / 4.0)["kp"] == 1.25


def test_control_architecture_tracks_path_heading_and_limits_outputs():
    controller = make_controller()
    state = AircraftState(
        pos=np.array([0.0, 0.0, -20.0]),
        vel=np.array([5.0, 0.0, 0.0]),
    )
    reference = {
        "north": 0.0,
        "east": 100.0,
        "altitude": 20.0,
        "speed": 5.0,
        "tilt": np.pi / 4.0,
    }

    command = controller.compute(state, reference, dt=0.01)

    assert np.isclose(command["desired_heading"], np.pi / 2.0)
    assert command["desired_roll"] > 0.0
    assert 0.0 <= command["tilt"] <= np.pi / 2.0
    assert abs(command["delta_left"]) <= 0.2
    assert abs(command["delta_right"]) <= 0.2
    assert np.all(np.isfinite(command["allocation_residual"]))
    assert 0.0 <= command["gain_blend"] <= 1.0


def test_control_architecture_wraps_heading_error_across_pi():
    controller = make_controller()
    state = AircraftState(
        pos=np.array([0.0, 0.0, -20.0]),
        att=np.array([0.0, 0.0, -np.deg2rad(179.0)]),
    )

    command = controller.compute(
        state,
        {"altitude": 20.0, "speed": 0.0, "heading": np.deg2rad(179.0)},
        dt=0.01,
    )

    assert abs(command["guidance_heading_error"]) < np.deg2rad(3.0)
    assert abs(command["desired_r"]) < 0.1


def test_outer_loop_converts_euler_rate_targets_to_body_rates():
    controller = ControlLawArchitecture(
        mass=5.0,
        k_thrust=1e-5,
        max_omega=3000.0,
        motor_positions=MOTOR_POSITIONS,
        attitude_gains={"kp": 1.0, "ki": 0.0, "kd": 0.0},
    )
    roll, pitch = np.deg2rad([10.0, 20.0])
    state = AircraftState(
        pos=np.array([0.0, 0.0, -20.0]),
        att=np.array([roll, pitch, 0.0]),
    )

    command = controller.compute(
        state,
        {"altitude": 20.0, "speed": 0.0, "heading": 0.0},
        dt=0.01,
    )

    euler_roll_rate = -roll
    euler_pitch_rate = -pitch
    expected_body_rates = np.array([
        euler_roll_rate,
        euler_pitch_rate * np.cos(roll),
        -euler_pitch_rate * np.sin(roll),
    ])
    np.testing.assert_allclose(
        [command["desired_p"], command["desired_q"], command["desired_r"]],
        expected_body_rates,
    )


def test_control_architecture_allocates_aerodynamic_pitch_bias():
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.6, vtail=vtail, prop_diameter_m=0.3048)
    controller = ControlLawArchitecture(
        mass=5.0,
        k_thrust=1e-5,
        max_omega=3000.0,
        motor_positions=MOTOR_POSITIONS,
        vtail_model=vtail,
        aero_model=aero,
    )
    dynamics = TiltrotorDynamics(
        mass=5.0,
        rotor=RotorModel(k_thrust=1e-5, max_omega=3000.0),
        aero=aero,
    )
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([2.0, 0.0, 0.0]),
    )

    command = controller.compute(state, {"altitude": 100.0, "speed": 2.0}, dt=0.01)
    dynamics.step(
        state,
        np.zeros(3),
        command["omega_fl"],
        command["omega_fr"],
        command["omega_rl"],
        command["omega_rr"],
        command["tilt"],
        command["delta_left"],
        command["delta_right"],
        0.01,
    )

    assert abs(dynamics.last_diagnostics["moment_body"][1]) < 0.1


def test_guidance_reduces_collective_for_aerodynamic_lift():
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.6, vtail=vtail, prop_diameter_m=0.3048)
    controller = ControlLawArchitecture(
        mass=5.0,
        k_thrust=1e-5,
        max_omega=3000.0,
        motor_positions=MOTOR_POSITIONS,
        vtail_model=vtail,
        aero_model=aero,
    )
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        vel=np.array([8.0, 0.0, 0.0]),
        att=np.array([0.0, np.deg2rad(10.0), 0.0]),
    )

    command = controller.compute(
        state,
        {"altitude": 100.0, "speed": 8.0},
        dt=0.01,
    )

    assert command["requested_thrust"] < 5.0 * 9.81


def test_tilt_governor_backs_off_when_pitch_allocation_is_unachievable():
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.6, vtail=vtail, prop_diameter_m=0.3048)
    controller = ControlLawArchitecture(
        mass=5.0,
        k_thrust=1e-5,
        max_omega=3000.0,
        motor_positions=MOTOR_POSITIONS,
        vtail_model=vtail,
        aero_model=aero,
    )
    state = AircraftState(
        pos=np.array([17.9, 0.35, -105.1]),
        vel=np.array([7.85, 0.34, -4.53]),
        att=np.deg2rad(np.array([-2.98, -9.31, 3.61])),
    )
    state.tilt = np.deg2rad(31.0)
    requested_tilt = np.deg2rad(31.3)

    command = controller.compute(
        state,
        {"altitude": 100.0, "speed": 4.2, "tilt": np.deg2rad(31.3)},
        dt=0.01,
        actuator_tilt_rad=requested_tilt,
    )

    assert command["tilt_limited"]
    assert command["tilt"] < state.tilt
    expected_actuator_tilt = state.tilt - (requested_tilt - state.tilt)
    np.testing.assert_allclose(
        command["allocation_matrix"],
        controller.allocator.allocation_matrix(expected_actuator_tilt),
    )


def test_pid_integral_anti_windup_respects_output_limit():
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, limit=1.0)

    output = pid.compute(setpoint=10.0, measurement=0.0, dt=1.0)

    assert output == 1.0
    assert pid.integral == 1.0
