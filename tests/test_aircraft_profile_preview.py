from pathlib import Path

import numpy as np

from src.configuration.aircraft_profile import dashboard_defaults, load_aircraft_profile
from src.configuration.aircraft_profile_preview import build_aircraft_profile_figure


PROFILE_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "configs" / "uam_profile_template.yaml"
)


def load_preview_values():
    profile = load_aircraft_profile(
        PROFILE_TEMPLATE.read_bytes(),
        PROFILE_TEMPLATE.name,
    )
    return dashboard_defaults(profile)


def test_preview_tracks_profile_dimensions_and_any_configured_rotor_count():
    values = load_preview_values()
    values["fuselage_length_m"] = 2.4
    values["wing_tip_y_m"] = 1.8
    values["motor_positions"].append([0.0, 0.0, 0.0, False])

    figure = build_aircraft_profile_figure("Test UAM", values, rotor_tilt_deg=0.0)

    traces = {trace.name: trace for trace in figure.data}
    fuselage = traces["Fuselage envelope"]
    assert np.isclose(max(fuselage.x) - min(fuselage.x), 2.4)
    assert np.isclose(max(abs(value) for value in traces["Left wing"].y), 1.8)
    assert len([name for name in traces if name.endswith(" center")]) == 5
    assert "Test UAM - profile geometry preview" == figure.layout.title.text


def test_preview_tilt_moves_enabled_rotors_and_changes_thrust_axis():
    values = load_preview_values()
    level = build_aircraft_profile_figure("Test UAM", values, rotor_tilt_deg=0.0)
    tilted = build_aircraft_profile_figure("Test UAM", values, rotor_tilt_deg=60.0)

    level_traces = {trace.name: trace for trace in level.data}
    tilted_traces = {trace.name: trace for trace in tilted.data}

    assert not np.isclose(
        level_traces["Rotor 1 center"].z[0],
        tilted_traces["Rotor 1 center"].z[0],
    )
    assert np.isclose(
        tilted_traces["Rotor 1 thrust axis"].x[1]
        - tilted_traces["Rotor 1 thrust axis"].x[0],
        np.sin(np.deg2rad(60.0)) * (12.0 * 0.0254),
    )
    assert np.allclose(
        level_traces["Rotor 3 center"].z,
        tilted_traces["Rotor 3 center"].z,
    )
