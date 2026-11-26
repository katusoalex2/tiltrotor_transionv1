import io
import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from src.configuration.aircraft_profile import (
    dashboard_defaults,
    load_aircraft_profile,
)
from src.configuration.aircraft_profile_excel import (
    load_aircraft_profile_excel,
    serialize_aircraft_profile_excel,
)


PROFILE_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "UAM Configurations" / "illustrative_uam_simulation_profile.json"
)


def load_template():
    return load_aircraft_profile(PROFILE_TEMPLATE.read_bytes(), PROFILE_TEMPLATE.name)


def workbook_with_edit(contents, pointer, value_json):
    workbook = load_workbook(io.BytesIO(contents))
    for worksheet in workbook.worksheets:
        for row in worksheet.iter_rows(min_row=2):
            if row[0].value == pointer:
                row[1].value = value_json
                stream = io.BytesIO()
                workbook.save(stream)
                workbook.close()
                return stream.getvalue()
    workbook.close()
    raise AssertionError(f"workbook did not contain {pointer}")


def test_excel_profile_round_trip_preserves_unknown_and_nested_values():
    profile = load_template()
    profile["extensions"]["vendor_parameters"] = {
        "key/with~escapes": [None, True, 1.25, {"nested": "value"}],
    }
    profile["third_party_system"] = {"custom_parameter": "not a formula: =1+1"}

    imported = load_aircraft_profile_excel(serialize_aircraft_profile_excel(profile))

    assert imported == profile


def test_excel_edits_update_dashboard_values():
    contents = serialize_aircraft_profile_excel(load_template())
    contents = workbook_with_edit(
        contents,
        "/aircraft/mass_properties/component_masses_kg/fuselage",
        "4.25",
    )

    imported = load_aircraft_profile_excel(contents)

    assert dashboard_defaults(imported)["fuselage_mass_kg"] == 4.25


def test_excel_profile_rejects_formula_cells():
    contents = serialize_aircraft_profile_excel(load_template())
    contents = workbook_with_edit(
        contents,
        "/aircraft/mass_properties/component_masses_kg/fuselage",
        "=1+1",
    )

    with pytest.raises(ValueError, match="contains a formula"):
        load_aircraft_profile_excel(contents)


def test_excel_profile_rejects_invalid_json_and_enabled_hardware_outputs():
    contents = serialize_aircraft_profile_excel(load_template())
    invalid_json = workbook_with_edit(
        contents,
        "/aircraft/mass_properties/component_masses_kg/fuselage",
        "not-json",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        load_aircraft_profile_excel(invalid_json)

    enabled_outputs = workbook_with_edit(
        contents,
        "/hardware/actuator_outputs_enabled",
        json.dumps(True),
    )
    with pytest.raises(ValueError, match="actuator outputs disabled"):
        load_aircraft_profile_excel(enabled_outputs)


def test_excel_profile_rejects_non_workbooks():
    with pytest.raises(ValueError, match="valid .xlsx workbook"):
        load_aircraft_profile_excel(b"not an Excel file")
