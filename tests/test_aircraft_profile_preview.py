from pathlib import Path

import numpy as np

from src.configuration.aircraft_profile import dashboard_defaults, load_aircraft_profile
from src.configuration.aircraft_profile_preview import (
    build_aircraft_profile_figure,
    build_transition_reference_animation,
)
from src.control.mission import MissionProfile


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


def test_reference_animation_reaches_ninety_degrees_without_tilting_fixed_rotors():
    figure = build_transition_reference_animation("Test UAM", load_preview_values())
    first, last = figure.frames[0], figure.frames[-1]
    first_traces = {trace.name: trace for trace in first.data}
    last_traces = {trace.name: trace for trace in last.data}

    assert first.name == "0.00"
    assert last.name == "21.00"
    assert "REFERENCE ONLY" in last.layout.title.text
    front_angles = []
    for frame in figure.frames:
        traces = {trace.name: trace for trace in frame.data}
        axis = traces["Rotor 1 thrust axis"]
        front_angles.append(np.arctan2(axis.x[1] - axis.x[0], axis.z[1] - axis.z[0]))
        for coordinate in ("x", "y", "z"):
            np.testing.assert_allclose(
                getattr(traces["Rotor 3 thrust axis"], coordinate),
                getattr(first_traces["Rotor 3 thrust axis"], coordinate),
            )
    mission = MissionProfile()
    expected = [mission.at(float(frame.name))["tilt"] for frame in figure.frames]
    np.testing.assert_allclose(front_angles, expected, atol=1e-12)
    assert np.all(np.diff(front_angles) >= 0.0)
    assert np.isclose(front_angles[0], 0.0)
    assert np.isclose(front_angles[-1], np.pi / 2)
    front_axis = last_traces["Rotor 1 thrust axis"]
    assert front_axis.x[1] > front_axis.x[0]
    assert np.isclose(front_axis.z[1], front_axis.z[0], atol=1e-12)
