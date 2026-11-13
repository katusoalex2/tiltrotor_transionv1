import numpy as np
import pytest
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import GRAVITY, TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel

def test_rotor_thrust_squared_law():
    rotor = RotorModel(k_thrust=1.2e-5, max_omega=1000.0)
    assert np.isclose(rotor.compute_thrust(100.0), 0.12)

def test_hover_vertical_thrust():
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1,0.1,0.2]), aero=aero)
    omega_hover = np.sqrt(49.05 / 1e-5)
    state = AircraftState()
    new_state = dyn.step(state, omega_cmd=omega_hover, tilt_cmd=0.0, delta_left=0.0, delta_right=0.0, dt=0.01)
    assert np.isclose(new_state.vel[2], 0.0, atol=1e-5)

def test_transition_forward_thrust():
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1,0.1,0.2]), aero=aero)
    state = AircraftState()
    new_state = dyn.step(state, omega_cmd=500.0, tilt_cmd=np.pi/2, delta_left=0.0, delta_right=0.0, dt=0.01)
    assert new_state.vel[0] > 0.0
    assert new_state.vel[2] > 0.0

def test_zero_dt_raises():
    rotor = RotorModel(1e-5, 1000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1,0.1,0.2]), aero=aero)
    with pytest.raises(ValueError):
        dyn.step(AircraftState(), omega_cmd=100.0, tilt_cmd=0.0, delta_left=0.0, delta_right=0.0, dt=0.0)

def test_zero_mass_raises():
    rotor = RotorModel(1e-5, 1000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.5, vtail=vtail)
    with pytest.raises(ValueError):
        TiltrotorDynamics(mass=0.0, rotor=rotor, inertia=np.array([0.1,0.1,0.2]), aero=aero)


def test_translational_kinematics_integrates_ned_position_from_velocity():
    dynamics = TiltrotorDynamics(
        mass=2.0,
        rotor=RotorModel(k_thrust=0.0, max_omega=1000.0),
        inertia=np.ones(3),
    )
    dt = 0.1

    updated = dynamics.step(
        AircraftState(),
        omega_cmd=0.0,
        tilt_cmd=0.0,
        dt=dt,
    )

    assert np.allclose(updated.vel, [0.0, 0.0, GRAVITY * dt])
    assert np.allclose(updated.pos, [0.0, 0.0, 0.5 * GRAVITY * dt**2])


def test_rotational_dynamics_include_eulers_gyroscopic_coupling():
    dynamics = TiltrotorDynamics(
        mass=2.0,
        rotor=RotorModel(k_thrust=0.0, max_omega=1000.0),
        inertia=np.array([2.0, 3.0, 4.0]),
    )
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        omega=np.array([1.0, 2.0, 3.0]),
    )
    dt = 0.01

    updated = dynamics.step(state, omega_cmd=0.0, tilt_cmd=0.0, dt=dt)

    assert np.allclose(updated.omega, [1.0 - 3.0 * dt, 2.0 + 2.0 * dt, 3.0 - 0.5 * dt])


def test_rotational_kinematics_integrate_body_yaw_rate():
    dynamics = TiltrotorDynamics(
        mass=2.0,
        rotor=RotorModel(k_thrust=0.0, max_omega=1000.0),
        inertia=np.ones(3),
    )
    state = AircraftState(
        pos=np.array([0.0, 0.0, -100.0]),
        omega=np.array([0.0, 0.0, 1.0]),
    )
    dt = 0.01

    updated = dynamics.step(state, omega_cmd=0.0, tilt_cmd=0.0, dt=dt)

    assert np.allclose(updated.att, [0.0, 0.0, dt])


def test_four_rotor_thrust_contributes_moment_through_motor_arm():
    class MassProperties:
        motor_positions = [
            [1.0, 0.5, 0.0, False],
            [1.0, -0.5, 0.0, False],
            [-1.0, 0.5, 0.0, False],
            [-1.0, -0.5, 0.0, False],
        ]

        def calculate_inertia(self, tilt_angle_deg):
            return np.eye(3)

    dynamics = TiltrotorDynamics(
        mass=1.0,
        rotor=RotorModel(k_thrust=1.0, max_omega=10.0),
        mass_props=MassProperties(),
    )
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))

    updated = dynamics.step(
        state,
        np.zeros(3),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
    )

    assert np.isclose(updated.omega[0], 0.05)
    assert np.isclose(updated.omega[1], 0.1)


def test_vtail_surface_inputs_contribute_body_force_and_moment():
    class ControlledAero:
        def compute_forces_and_moments(self, vel, wind, att, total_thrust, tilt):
            return np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0

        def compute_tail_forces(self, vel, wind, att, delta_left, delta_right):
            return np.array([0.0, 2.0, 0.0]), np.array([3.0, 4.0, 5.0])

    dynamics = TiltrotorDynamics(
        mass=2.0,
        rotor=RotorModel(k_thrust=0.0, max_omega=1000.0),
        inertia=np.array([2.0, 3.0, 4.0]),
        aero=ControlledAero(),
    )
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))
    dt = 0.1

    updated = dynamics.step(
        state,
        omega_cmd=0.0,
        tilt_cmd=0.0,
        delta_left=0.2,
        delta_right=-0.2,
        dt=dt,
    )

    assert np.isclose(updated.vel[1], dt)
    assert np.allclose(updated.omega, [1.5 * dt, (4.0 / 3.0) * dt, 1.25 * dt])
    assert np.allclose(dynamics.last_diagnostics["force_body"], [0.0, 2.0, 0.0])
    assert np.allclose(dynamics.last_diagnostics["moment_body"], [3.0, 4.0, 5.0])
    assert np.allclose(dynamics.last_diagnostics["angular_acceleration_body"], [1.5, 4.0 / 3.0, 1.25])
