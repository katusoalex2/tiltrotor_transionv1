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
