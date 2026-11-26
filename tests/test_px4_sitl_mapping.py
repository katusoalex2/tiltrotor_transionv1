import json
from pathlib import Path

import pytest

from src.configuration.aircraft_profile import load_aircraft_profile
from src.configuration.aircraft_profile import (
    dashboard_defaults,
    profile_with_dashboard_values,
)
from src.configuration.px4_sitl_mapping import (
    build_px4_sitl_mapping,
    serialize_px4_sitl_mapping,
)


PROFILE_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "UAM Configurations" / "illustrative_uam_simulation_profile.json"
)


def load_template():
    return load_aircraft_profile(PROFILE_TEMPLATE.read_bytes(), PROFILE_TEMPLATE.name)


def test_sitl_mapping_preserves_profile_geometry_without_assigning_hardware_outputs():
    mapping = build_px4_sitl_mapping(
        load_template(),
        px4_version="unknown",
        flight_controller="unknown",
    )

    rotors = mapping["actuator_geometry"]["rotors"]
    assert mapping["scope"] == "SITL_MAPPING_ONLY"
    assert mapping["flashable"] is False
    assert mapping["px4_parameter_file_generated"] is False
    assert mapping["source_hardware_outputs_enabled"] is False
    assert len(rotors) == 4
    assert [rotor["tilt_enabled"] for rotor in rotors] == [True, True, False, False]
    assert rotors[0]["position_profile_m"] == [0.3, -0.4, 0.0]
    assert all(rotor["physical_output_assignment"] is None for rotor in rotors)
    assert rotors[0]["rotation_direction"] == "UNVERIFIED"
    assert any("exact PX4 release" in item for item in mapping["unresolved_before_hardware_configuration"])
    assert any("coordinate convention" in item for item in mapping["unresolved_before_hardware_configuration"])


def test_sitl_mapping_exports_editable_direction_and_tilt_choices_but_stays_nonflashable():
    profile = load_template()
    directions = {
        "front_left": "CW",
        "front_right": "CCW",
        "rear_left": "CCW",
        "rear_right": "CW",
    }

    serialized = serialize_px4_sitl_mapping(
        profile,
        px4_version="v1.16.x",
        flight_controller="Pixhawk placeholder",
        motor_directions=directions,
        tilt_servo_arrangement="INDEPENDENT_SERVOS",
    )
    mapping = json.loads(serialized)

    assert mapping["target"]["px4_version"] == "v1.16.x"
    assert mapping["actuator_geometry"]["tilt"]["servo_arrangement"] == "INDEPENDENT_SERVOS"
    assert [rotor["rotation_direction"] for rotor in mapping["actuator_geometry"]["rotors"]] == [
        "CW",
        "CCW",
        "CCW",
        "CW",
    ]
    assert mapping["flashable"] is False
    assert mapping["px4_parameter_file_generated"] is False


def test_sitl_mapping_rejects_unrecognized_motor_directions():
    with pytest.raises(ValueError, match="motor direction"):
        build_px4_sitl_mapping(
            load_template(),
            px4_version="unknown",
            flight_controller="unknown",
            motor_directions={"front_left": "reverse"},
        )


def test_sitl_mapping_uses_dashboard_rotor_edits_and_supports_profile_rotor_counts():
    profile = load_template()
    profile["aircraft"]["geometry"]["rotors"].append(
        {
            "id": "center_rear",
            "position_m": [-0.5, 0.0, 0.0],
            "tilt_enabled": False,
        }
    )
    values = dashboard_defaults(profile)
    values["motor_positions"] = [
        [0.1, 0.2, 0.3, False],
        [0.4, 0.5, 0.6, True],
        [0.7, 0.8, 0.9, False],
        [1.0, 1.1, 1.2, True],
        [1.3, 1.4, 1.5, True],
    ]
    active_profile = profile_with_dashboard_values(profile, values)

    mapping = build_px4_sitl_mapping(
        active_profile,
        px4_version="unknown",
        flight_controller="unknown",
    )
    mapped_rotors = mapping["actuator_geometry"]["rotors"]

    assert len(mapped_rotors) == 5
    assert mapped_rotors[0]["position_profile_m"] == [0.1, 0.2, 0.3]
    assert [rotor["tilt_enabled"] for rotor in mapped_rotors] == [
        False,
        True,
        False,
        True,
        True,
    ]


def test_sitl_mapping_rejects_profiles_with_hardware_outputs_enabled():
    profile = load_template()
    profile["hardware"]["actuator_outputs_enabled"] = True

    with pytest.raises(ValueError, match="actuator outputs to be disabled"):
        build_px4_sitl_mapping(
            profile,
            px4_version="unknown",
            flight_controller="unknown",
        )
