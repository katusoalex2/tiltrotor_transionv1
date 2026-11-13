import numpy as np
import pytest

from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.dynamics.transition import (
    control_effectiveness_map,
    solve_longitudinal_trim,
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
