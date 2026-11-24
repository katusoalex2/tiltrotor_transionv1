"""Schematic 3D rendering for editable aircraft-profile geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go

from src.control.mission import MissionProfile


def build_aircraft_profile_figure(
    aircraft_name: str,
    values: Mapping[str, Any],
    rotor_tilt_deg: float,
) -> go.Figure:
    """Build a profile-driven 3D aircraft preview in an X-forward, Z-up frame."""
    fuselage_length = _positive_number(values, "fuselage_length_m")
    fuselage_width = _positive_number(values, "fuselage_width_m")
    fuselage_height = _positive_number(values, "fuselage_height_m")
    wing_root_x = _number(values, "wing_root_le_x_m")
    wing_root_y = _number(values, "wing_root_y_m")
    wing_tip_x = _number(values, "wing_tip_le_x_m")
    wing_tip_y = _positive_number(values, "wing_tip_y_m")
    wing_root_chord = _positive_number(values, "wing_root_chord_m")
    wing_tip_chord = _positive_number(values, "wing_tip_chord_m")
    vtail_root_x = _number(values, "vtail_root_x_m")
    vtail_tip_x = _number(values, "vtail_tip_x_m")
    vtail_tip_y = _positive_number(values, "vtail_tip_y_m")
    vtail_tip_z_down = _number(values, "vtail_tip_z_m")
    vtail_area = _positive_number(values, "vtail_area_m2")
    propeller_diameter_in = _positive_number(values, "prop_diameter_in")
    pivot_x = _number(values, "motor_tilt_pivot_x_m")
    pivot_z_down = _number(values, "motor_tilt_pivot_z_m")
    tilt_deg = _number({"tilt": rotor_tilt_deg}, "tilt")
    if not 0.0 <= tilt_deg <= 90.0:
        raise ValueError("rotor tilt angle must be between 0 and 90 degrees")

    motor_positions = values.get("motor_positions")
    if not isinstance(motor_positions, list) or not motor_positions:
        raise ValueError("aircraft profile preview requires at least one rotor")
    for index, position in enumerate(motor_positions):
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 4
            or not all(_is_finite_number(coordinate) for coordinate in position[:3])
            or not isinstance(position[3], bool)
        ):
            raise ValueError(f"motor_positions[{index}] must contain x/y/z and a tilt flag")

    figure = go.Figure()
    scene_points: list[np.ndarray] = []

    fuselage_vertices, fuselage_faces = _fuselage_mesh(
        fuselage_length,
        fuselage_width,
        fuselage_height,
    )
    figure.add_trace(
        go.Mesh3d(
            x=fuselage_vertices[:, 0],
            y=fuselage_vertices[:, 1],
            z=fuselage_vertices[:, 2],
            i=fuselage_faces[:, 0],
            j=fuselage_faces[:, 1],
            k=fuselage_faces[:, 2],
            color="#8b929a",
            opacity=0.95,
            flatshading=True,
            name="Fuselage envelope",
            hovertemplate="Fuselage envelope<extra></extra>",
        )
    )
    scene_points.append(fuselage_vertices)

    for side, sign, color in (
        ("left", -1.0, "#2878b5"),
        ("right", 1.0, "#2878b5"),
    ):
        root_y = sign * abs(wing_root_y)
        tip_y = sign * wing_tip_y
        corners = np.array(
            [
                [wing_root_x, root_y, 0.0],
                [wing_root_x - wing_root_chord, root_y, 0.0],
                [wing_tip_x - wing_tip_chord, tip_y, 0.0],
                [wing_tip_x, tip_y, 0.0],
            ],
            dtype=float,
        )
        vertices, faces = _panel_mesh(
            corners,
            root_thickness=0.12 * wing_root_chord,
            tip_thickness=0.12 * wing_tip_chord,
        )
        figure.add_trace(
            _mesh_trace(
                vertices,
                faces,
                name=f"{side.capitalize()} wing",
                color=color,
            )
        )
        scene_points.append(vertices)

    tail_span = math.hypot(vtail_tip_y, vtail_tip_z_down)
    tail_root_chord = vtail_area / (1.5 * tail_span)
    tail_tip_chord = tail_root_chord * 0.5
    tail_tip_z_up = -vtail_tip_z_down
    for side, sign in (("left", -1.0), ("right", 1.0)):
        corners = np.array(
            [
                [vtail_root_x, 0.0, 0.0],
                [vtail_root_x - tail_root_chord, 0.0, 0.0],
                [
                    vtail_tip_x - tail_tip_chord,
                    sign * vtail_tip_y,
                    tail_tip_z_up,
                ],
                [vtail_tip_x, sign * vtail_tip_y, tail_tip_z_up],
            ],
            dtype=float,
        )
        vertices, faces = _panel_mesh(
            corners,
            root_thickness=0.08 * tail_root_chord,
            tip_thickness=0.08 * tail_tip_chord,
        )
        figure.add_trace(
            _mesh_trace(
                vertices,
                faces,
                name=f"{side.capitalize()} V-tail",
                color="#a74a4a",
            )
        )
        scene_points.append(vertices)

    tilt = math.radians(tilt_deg)
    propeller_radius = propeller_diameter_in * 0.0254 / 2.0
    disk_angles = np.linspace(0.0, 2.0 * math.pi, 49)
    arrow_length = max(0.25, 2.0 * propeller_radius)
    rotor_colors = ("#1677c8", "#e08a24")

    for index, (x, y, z_down, tilt_enabled) in enumerate(motor_positions, start=1):
        rotor_tilt = tilt if tilt_enabled else 0.0
        rotor_cosine = math.cos(rotor_tilt)
        rotor_sine = math.sin(rotor_tilt)
        relative_x = float(x) - pivot_x
        relative_z = float(z_down) - pivot_z_down
        rotor_x = pivot_x + relative_x * rotor_cosine + relative_z * rotor_sine
        rotor_z_down = (
            pivot_z_down - relative_x * rotor_sine + relative_z * rotor_cosine
            if tilt_enabled
            else float(z_down)
        )
        center = np.array([rotor_x, float(y), -rotor_z_down], dtype=float)
        color = rotor_colors[0] if tilt_enabled else rotor_colors[1]

        disk_x = center[0] + propeller_radius * np.cos(disk_angles) * rotor_cosine
        disk_y = center[1] + propeller_radius * np.sin(disk_angles)
        disk_z = center[2] + propeller_radius * np.cos(disk_angles) * rotor_sine
        figure.add_trace(
            go.Scatter3d(
                x=disk_x,
                y=disk_y,
                z=disk_z,
                mode="lines",
                line={"color": color, "width": 4},
                name=f"Rotor {index} propeller disk",
                showlegend=False,
                hovertemplate=f"Rotor {index} propeller disk<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers",
                marker={"color": color, "size": 6},
                name=f"Rotor {index} center",
                showlegend=False,
                hovertemplate=f"Rotor {index} center<extra></extra>",
            )
        )
        thrust_end = center + arrow_length * np.array(
            [rotor_sine, 0.0, rotor_cosine],
            dtype=float,
        )
        figure.add_trace(
            go.Scatter3d(
                x=[center[0], thrust_end[0]],
                y=[center[1], thrust_end[1]],
                z=[center[2], thrust_end[2]],
                mode="lines",
                line={"color": color, "width": 3, "dash": "dash"},
                name=f"Rotor {index} thrust axis",
                showlegend=False,
                hovertemplate=f"Rotor {index} thrust axis<extra></extra>",
            )
        )
        scene_points.extend((np.array([center]), np.array([thrust_end])))
        scene_points.append(np.column_stack((disk_x, disk_y, disk_z)))

    all_points = np.concatenate(scene_points, axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    margin = max(0.15, float(np.max(upper - lower)) * 0.08)
    axis_titles = ("X forward (m)", "Y right (m)", "Z up (m)")
    scene = {
        axis: {
            "title": title,
            "range": [float(lower[index] - margin), float(upper[index] + margin)],
            "backgroundcolor": "rgba(240, 240, 240, 0.45)",
            "gridcolor": "rgba(100, 100, 100, 0.25)",
            "zerolinecolor": "rgba(70, 70, 70, 0.5)",
        }
        for index, (axis, title) in enumerate(
            zip(("xaxis", "yaxis", "zaxis"), axis_titles)
        )
    }
    figure.update_layout(
        title={"text": f"{aircraft_name} - profile geometry preview", "x": 0.02},
        scene={
            **scene,
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 44},
        height=620,
    )
    return figure


def build_transition_reference_animation(
    aircraft_name: str,
    values: Mapping[str, Any],
) -> go.Figure:
    """Animate the timed mission reference, not a simulated aircraft trajectory."""
    mission = MissionProfile()
    times = np.linspace(0.0, mission.total_time, 43)
    figure = build_aircraft_profile_figure(aircraft_name, values, 0.0)
    rotor_indices = [
        index for index, trace in enumerate(figure.data)
        if trace.name and trace.name.startswith("Rotor ")
    ]
    axis_names = ("xaxis", "yaxis", "zaxis")
    bounds = {
        axis: list(getattr(figure.layout.scene, axis).range)
        for axis in axis_names
    }
    frames = []
    steps = []
    for time in times:
        angle = math.degrees(float(mission.at(float(time))["tilt"]))
        preview = build_aircraft_profile_figure(aircraft_name, values, angle)
        for axis in axis_names:
            axis_range = getattr(preview.layout.scene, axis).range
            bounds[axis][0] = min(bounds[axis][0], axis_range[0])
            bounds[axis][1] = max(bounds[axis][1], axis_range[1])
        name = f"{time:.2f}"
        frames.append(go.Frame(
            name=name,
            traces=rotor_indices,
            data=[preview.data[index] for index in rotor_indices],
            layout=go.Layout(
                title_text=f"REFERENCE ONLY — {time:.1f} s, tilt {angle:.1f}°"
            ),
        ))
        steps.append({
            "label": f"{time:.1f}",
            "method": "animate",
            "args": [[name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        })

    figure.frames = frames
    figure.update_layout(
        title={"text": "REFERENCE ONLY — 0.0 s, tilt 0.0°", "x": 0.02},
        scene={axis: {"range": bounds[axis]} for axis in axis_names},
        annotations=[{
            "text": "REFERENCE ONLY — NOT SIMULATED FLIGHT",
            "x": 0.5, "y": 1.0, "xref": "paper", "yref": "paper",
            "showarrow": False,
        }],
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.0, "y": 0.0,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [
                    None, {"fromcurrent": True, "frame": {"duration": 500, "redraw": True},
                           "transition": {"duration": 0}}
                ]},
                {"label": "Pause", "method": "animate", "args": [
                    [None], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                             "transition": {"duration": 0}}
                ]},
            ],
        }],
        sliders=[{"active": 0, "currentvalue": {"prefix": "Reference time (s): "},
                  "steps": steps, "x": 0.2, "len": 0.8, "y": 0.0}],
        margin={"l": 0, "r": 0, "b": 80, "t": 70},
    )
    return figure


def _fuselage_mesh(
    length: float,
    width: float,
    height: float,
    radial_segments: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    station_scales = np.array([0.08, 0.4, 0.75, 0.95, 1.0, 0.95, 0.75, 0.4, 0.08])
    stations = np.linspace(-length / 2.0, length / 2.0, len(station_scales))
    angles = np.linspace(0.0, 2.0 * math.pi, radial_segments, endpoint=False)
    vertices = np.array(
        [
            [
                x,
                width * 0.5 * scale * math.cos(angle),
                height * 0.5 * scale * math.sin(angle),
            ]
            for x, scale in zip(stations, station_scales)
            for angle in angles
        ],
        dtype=float,
    )
    faces: list[tuple[int, int, int]] = []
    for station in range(len(stations) - 1):
        for segment in range(radial_segments):
            next_segment = (segment + 1) % radial_segments
            first = station * radial_segments + segment
            second = station * radial_segments + next_segment
            third = (station + 1) * radial_segments + next_segment
            fourth = (station + 1) * radial_segments + segment
            faces.extend(((first, second, third), (first, third, fourth)))

    nose_center = len(vertices)
    tail_center = nose_center + 1
    vertices = np.vstack(
        (vertices, [stations[-1], 0.0, 0.0], [stations[0], 0.0, 0.0])
    )
    for segment in range(radial_segments):
        next_segment = (segment + 1) % radial_segments
        faces.append((nose_center, (len(stations) - 1) * radial_segments + segment,
                      (len(stations) - 1) * radial_segments + next_segment))
        faces.append((tail_center, next_segment, segment))
    return vertices, np.asarray(faces, dtype=int)


def _panel_mesh(
    corners: np.ndarray,
    root_thickness: float,
    tip_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    thickness = np.array(
        [root_thickness, root_thickness, tip_thickness, tip_thickness],
        dtype=float,
    ) * 0.5
    top = corners.copy()
    bottom = corners.copy()
    top[:, 2] += thickness
    bottom[:, 2] -= thickness
    vertices = np.vstack((top, bottom))
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
    ]
    for edge in range(4):
        next_edge = (edge + 1) % 4
        faces.extend(
            (
                (edge, next_edge, next_edge + 4),
                (edge, next_edge + 4, edge + 4),
            )
        )
    return vertices, np.asarray(faces, dtype=int)


def _mesh_trace(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    name: str,
    color: str,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=0.95,
        flatshading=True,
        name=name,
        showlegend=False,
        hovertemplate=f"{name}<extra></extra>",
    )


def _number(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if not _is_finite_number(value):
        raise ValueError(f"aircraft preview parameter {key} must be finite numeric data")
    return float(value)


def _positive_number(values: Mapping[str, Any], key: str) -> float:
    value = _number(values, key)
    if value <= 0.0:
        raise ValueError(f"aircraft preview parameter {key} must be positive")
    return value


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
