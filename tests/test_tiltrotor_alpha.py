import numpy as np

from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.dynamics.rotor import RotorModel
from src.dynamics.state import AircraftState
from src.dynamics.tiltrotor import TiltrotorDynamics


class IdentityMassProperties:
    def calculate_inertia(self, tilt_angle_deg):
        return np.eye(3)


def test_step_stores_angle_of_attack_from_aerodynamics():
    dynamics = TiltrotorDynamics(
        mass=5.0,
        rotor=RotorModel(k_thrust=1e-5, max_omega=3000.0),
        mass_props=IdentityMassProperties(),
        aero=AeroModel(wing_area=0.5, vtail=VtailModel(area=0.1, arm_length=0.5)),
    )
    state = AircraftState(vel=np.array([10.0, 0.0, 2.0]))

    updated = dynamics.step(
        state,
        np.zeros(3),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.01,
    )

    assert np.isclose(updated.alpha, np.arctan2(2.0, 10.0))
