import json
from pathlib import Path

import pytest

from src.configuration.aircraft_profile import (
    dashboard_defaults,
    load_aircraft_profile,
    profile_with_dashboard_values,
    serialize_aircraft_profile,
    simulation_compatibility_issues,
)


PROFILE_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "UAM Configurations" / "illustrative_uam_simulation_profile.json"
)


def load_template():
    return load_aircraft_profile(PROFILE_TEMPLATE.read_bytes(), PROFILE_TEMPLATE.name)


def test_template_loads_with_simulator_defaults_and_supported_geometry():
    profile = load_template()
    defaults = dashboard_defaults(profile)

    assert profile["hardware"]["actuator_outputs_enabled"] is False
    assert defaults["fuselage_mass_kg"] == 3.0
    assert defaults["fuselage_length_m"] == 1.6
    assert defaults["motor_tilt_pivot_x_m"] == 0.1
    assert defaults["motor_positions"][0] == [0.3, -0.4, 0.0, True]
    assert simulation_compatibility_issues(profile) == ()


def test_profile_round_trip_preserves_custom_fields_and_dashboard_edits():
    profile = load_template()
    profile["extensions"]["vendor_parameters"] = {
        "sensor_bus": "CAN-FD",
        "nested": {"custom_gain": 1.25},
    }
    values = dashboard_defaults(profile)
    values["fuselage_mass_kg"] = 4.2
    values["fuselage_length_m"] = 2.1
    values["vtail_tip_x_m"] = -0.45
    exported = profile_with_dashboard_values(profile, values)

    serialized = serialize_aircraft_profile(exported)
    imported = load_aircraft_profile(serialized, "aircraft.json")

    assert imported["extensions"]["vendor_parameters"]["nested"]["custom_gain"] == 1.25
    assert dashboard_defaults(imported)["fuselage_mass_kg"] == 4.2
    assert dashboard_defaults(imported)["fuselage_length_m"] == 2.1
    assert dashboard_defaults(imported)["vtail_tip_x_m"] == -0.45


def test_profile_rejects_enabled_hardware_outputs_and_non_finite_data():
    profile = load_template()
    profile["hardware"]["actuator_outputs_enabled"] = True
    with pytest.raises(ValueError, match="actuator outputs disabled"):
        serialize_aircraft_profile(profile)

    profile = load_template()
    profile["extensions"]["bad_value"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        serialize_aircraft_profile(profile)


def test_profile_reports_aircraft_architectures_not_supported_by_current_simulator():
    profile = load_template()
    profile["aircraft"]["geometry"]["rotors"].append(
        {"id": "extra", "position_m": [0.0, 0.0, 0.0], "tilt_enabled": False}
    )

    issues = simulation_compatibility_issues(profile)

    assert any("exactly four rotors" in issue for issue in issues)


def test_profile_with_missing_simulation_inputs_is_not_marked_compatible():
    profile = load_template()
    del profile["aircraft"]["mass_properties"]["component_masses_kg"]["motor"]

    issues = simulation_compatibility_issues(profile)

    assert any("mass_properties.component_masses_kg.motor" in issue for issue in issues)


def test_profile_rejects_unknown_schema_and_invalid_dashboard_ranges():
    profile = load_template()
    profile["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        serialize_aircraft_profile(profile)

    profile = load_template()
    profile["simulation"] = {"dashboard_overrides": {"max_pitch_deg": 70.0}}
    with pytest.raises(ValueError, match="max_pitch_deg"):
        serialize_aircraft_profile(profile)


def test_json_import_preserves_unknown_top_level_extensions():
    profile = load_template()
    profile["third_party_system"] = {"custom_parameter": 42}
    imported = load_aircraft_profile(
        json.dumps(profile).encode("utf-8"),
        "aircraft.json",
    )

    assert imported["third_party_system"] == {"custom_parameter": 42}
