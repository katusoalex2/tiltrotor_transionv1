import numpy as np

from src.aero.handoff import compute_lift_breakdown


def test_fixed_rear_rotors_still_support_aircraft_at_ninety_degrees():
    result = compute_lift_breakdown(
        None, np.pi / 2, 0.0, 0.0, 20.0, 0.5, 5.0,
        rotor_thrusts=[4.0, 6.0, 3.0, 7.0],
        rotor_tilts=[True, True, False, False],
    )

    assert np.isclose(result["rotor_thrust"], 10.0)
