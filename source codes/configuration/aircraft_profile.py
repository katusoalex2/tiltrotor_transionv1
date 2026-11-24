"""Versioned, extensible aircraft profiles for simulation and bench planning."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PROFILE_SCHEMA_VERSION = 1
MAX_PROFILE_BYTES = 32 * 1024 * 1024
MAX_PROFILE_NODES = 3_000_000
MAX_PROFILE_DEPTH = 64

DEFAULT_DASHBOARD_VALUES: dict[str, Any] = {
    "fuselage_mass_kg": 3.0,
    "wing_mass_kg": 1.0,
    "vtail_mass_kg": 0.5,
    "motor_mass_kg": 0.125,
    "battery_cells": 6,
    "battery_capacity_mah": 5000,
    "battery_c_rating": 30,
    "motor_kv": 920,
    "prop_diameter_in": 12.0,
    "prop_pitch_in": 5.5,
    "prop_blades": 2,
    "motor_positions": [
        [0.3, 0.4, 0.0, True],
        [0.3, -0.4, 0.0, True],
        [-0.3, 0.4, 0.0, False],
        [-0.3, -0.4, 0.0, False],
    ],
    "fuselage_length_m": 1.6,
    "fuselage_width_m": 0.12,
    "fuselage_height_m": 0.14,
    "motor_tilt_pivot_x_m": 0.1,
    "motor_tilt_pivot_z_m": 0.0,
    "max_tilt_rate_deg_s": 30.0,
    "wing_root_le_x_m": 0.10,
    "wing_root_y_m": 0.0,
    "wing_tip_le_x_m": 0.05,
    "wing_tip_y_m": 1.0,
    "wing_root_chord_m": 0.40,
    "wing_tip_chord_m": 0.20,
    "wing_sweep_deg": 0.0,
    "vtail_root_x_m": -0.6,
    "vtail_tip_x_m": -0.5,
    "vtail_tip_y_m": 0.3,
    "vtail_tip_z_m": -0.15,
    "vtail_area_m2": 0.1,
    "vtail_arm_m": 0.5,
    "alt_kp": 1.5,
    "alt_ki": 0.05,
    "alt_kd": 0.1,
    "speed_kp": 0.1,
    "speed_ki": 0.0,
    "speed_kd": 0.01,
    "attitude_kp": 1.0,
    "attitude_ki": 0.0,
    "attitude_kd": 0.05,
    "rate_kp": 0.5,
    "rate_ki": 0.0,
    "rate_kd": 0.01,
    "hover_rate_scale": 1.5,
    "cruise_rate_scale": 0.75,
    "gust_enabled": True,
    "gust_model": "Finite-duration gust",
    "gust_start_s": 5.0,
    "gust_duration_s": 2.0,
    "gust_magnitude_mps": 6.0,
    "gust_direction": "Front",
    "dryden_sigma_mps": 2.5,
    "dryden_tau_s": 1.0,
    "enable_envelope_protection": True,
    "observer_time_constant_s": 0.25,
    "max_alpha_deg": 15.0,
    "max_pitch_deg": 60.0,
    "max_body_rate_deg_s": 180.0,
    "speed_margin_mps": 0.5,
}


def load_aircraft_profile(contents: bytes, filename: str) -> dict[str, Any]:
    """Parse and validate a JSON or YAML profile without executing embedded code."""
    if not isinstance(contents, bytes):
        raise TypeError("aircraft profile contents must be bytes")
    if len(contents) > MAX_PROFILE_BYTES:
        raise ValueError(
            f"aircraft profile exceeds the {MAX_PROFILE_BYTES // (1024 * 1024)} MiB limit"
        )

    suffix = Path(filename).suffix.lower()
    try:
        text = contents.decode("utf-8-sig")
        if suffix == ".json":
            profile = json.loads(text)
        elif suffix in (".yaml", ".yml"):
            profile = yaml.safe_load(text)
        else:
            raise ValueError("aircraft profiles must use a .json, .yaml, or .yml file")
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"could not parse aircraft profile: {exc}") from exc

    validate_aircraft_profile(profile)
    return copy.deepcopy(dict(profile))


def validate_aircraft_profile(profile: Any) -> None:
    """Reject malformed or unsafe profile documents while preserving custom fields."""
    if not isinstance(profile, Mapping):
        raise ValueError("aircraft profile root must be a mapping")
    _validate_data_tree(profile, "profile", set(), [0], 0)

    version = profile.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {PROFILE_SCHEMA_VERSION}")

    metadata = profile.get("metadata")
    aircraft = profile.get("aircraft")
    if not isinstance(metadata, Mapping) or not isinstance(aircraft, Mapping):
        raise ValueError("profile must contain metadata and aircraft mappings")
    if not isinstance(metadata.get("name"), str) or not metadata["name"].strip():
        raise ValueError("metadata.name must be a non-empty string")

    hardware = profile.get("hardware", {})
    if not isinstance(hardware, Mapping):
        raise ValueError("hardware must be a mapping")
    if hardware.get("actuator_outputs_enabled", False) is not False:
        raise ValueError(
            "this application only accepts profiles with hardware actuator outputs disabled"
        )
    if hardware.get("execution_mode", "simulation") not in ("simulation", "bench_hil"):
        raise ValueError("profile execution_mode must be simulation or bench_hil")

    simulation = profile.get("simulation", {})
    if not isinstance(simulation, Mapping):
        raise ValueError("simulation must be a mapping")
    overrides = simulation.get("dashboard_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("simulation.dashboard_overrides must be a mapping")
    if overrides:
        dashboard_values = copy.deepcopy(DEFAULT_DASHBOARD_VALUES)
        dashboard_values.update({
            key: value for key, value in overrides.items()
            if key in DEFAULT_DASHBOARD_VALUES
        })
        _validate_dashboard_values(dashboard_values)

    geometry = aircraft.get("geometry", {})
    if not isinstance(geometry, Mapping):
        raise ValueError("aircraft.geometry must be a mapping")
    rotors = geometry.get("rotors")
    if rotors is not None:
        if not isinstance(rotors, list) or not rotors:
            raise ValueError("aircraft.geometry.rotors must be a non-empty list")
        for index, rotor in enumerate(rotors):
            if not isinstance(rotor, Mapping):
                raise ValueError(f"aircraft.geometry.rotors[{index}] must be a mapping")
            if "tilt_enabled" in rotor and not isinstance(rotor["tilt_enabled"], bool):
                raise ValueError(
                    f"aircraft.geometry.rotors[{index}].tilt_enabled must be a boolean"
                )
            position = rotor.get("position_m")
            if position is not None and (
                not isinstance(position, list)
                or len(position) != 3
                or not all(_is_finite_number(value) for value in position)
            ):
                raise ValueError(
                    f"aircraft.geometry.rotors[{index}].position_m must be a finite 3-vector"
                )
    fuselage = geometry.get("fuselage", {})
    if not isinstance(fuselage, Mapping):
        raise ValueError("aircraft.geometry.fuselage must be a mapping")
    tilt_pivot = geometry.get("tilt_axis_xz_m")
    if tilt_pivot is not None and (
        not isinstance(tilt_pivot, list)
        or len(tilt_pivot) != 2
        or not all(_is_finite_number(value) for value in tilt_pivot)
    ):
        raise ValueError("aircraft.geometry.tilt_axis_xz_m must be a finite 2-vector")


def dashboard_defaults(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return supported dashboard controls, retaining extensible profile data separately."""
    values = copy.deepcopy(DEFAULT_DASHBOARD_VALUES)
    aircraft = profile.get("aircraft", {})
    if not isinstance(aircraft, Mapping):
        return values

    masses = _path(aircraft, "mass_properties.component_masses_kg")
    _copy_fields(masses, values, {
        "fuselage": "fuselage_mass_kg",
        "wing": "wing_mass_kg",
        "vtail": "vtail_mass_kg",
        "motor": "motor_mass_kg",
    })
    battery = _path(aircraft, "propulsion.battery")
    _copy_fields(battery, values, {
        "cells": "battery_cells",
        "capacity_mah": "battery_capacity_mah",
        "c_rating": "battery_c_rating",
    })
    motor = _path(aircraft, "propulsion.motor")
    _copy_fields(motor, values, {
        "kv_rpm_per_volt": "motor_kv",
    })
    propeller = _path(aircraft, "propulsion.propeller")
    _copy_fields(propeller, values, {
        "diameter_in": "prop_diameter_in",
        "pitch_in": "prop_pitch_in",
        "blade_count": "prop_blades",
    })
    rotors = _path(aircraft, "geometry.rotors")
    if isinstance(rotors, list) and rotors:
        values["motor_positions"] = [
            [
                *rotor["position_m"],
                bool(rotor.get("tilt_enabled", False)),
            ]
            for rotor in rotors
            if isinstance(rotor, Mapping)
            and isinstance(rotor.get("position_m"), list)
            and len(rotor["position_m"]) == 3
        ]
    fuselage = _path(aircraft, "geometry.fuselage")
    _copy_fields(fuselage, values, {
        "length_m": "fuselage_length_m",
        "width_m": "fuselage_width_m",
        "height_m": "fuselage_height_m",
    })
    tilt_pivot = _path(aircraft, "geometry.tilt_axis_xz_m")
    if isinstance(tilt_pivot, list) and len(tilt_pivot) == 2:
        values["motor_tilt_pivot_x_m"] = tilt_pivot[0]
        values["motor_tilt_pivot_z_m"] = tilt_pivot[1]
    wing = _path(aircraft, "geometry.wing")
    _copy_fields(wing, values, {
        "root_le_x_m": "wing_root_le_x_m",
        "root_y_m": "wing_root_y_m",
        "tip_le_x_m": "wing_tip_le_x_m",
        "tip_y_m": "wing_tip_y_m",
        "root_chord_m": "wing_root_chord_m",
        "tip_chord_m": "wing_tip_chord_m",
        "sweep_deg": "wing_sweep_deg",
    })
    vtail = _path(aircraft, "geometry.vtail")
    _copy_fields(vtail, values, {
        "root_x_m": "vtail_root_x_m",
        "tip_x_m": "vtail_tip_x_m",
        "tip_y_m": "vtail_tip_y_m",
        "tip_z_m": "vtail_tip_z_m",
        "area_m2": "vtail_area_m2",
        "arm_m": "vtail_arm_m",
    })
    control = _path(aircraft, "control")
    for section, prefix in (
        ("altitude_pid", "alt"),
        ("speed_pid", "speed"),
        ("attitude_pid", "attitude"),
        ("rate_pid", "rate"),
    ):
        gains = control.get(section, {}) if isinstance(control, Mapping) else {}
        _copy_fields(gains, values, {
            "kp": f"{prefix}_kp",
            "ki": f"{prefix}_ki",
            "kd": f"{prefix}_kd",
        })
    if isinstance(control, Mapping):
        _copy_fields(control, values, {
            "hover_rate_scale": "hover_rate_scale",
            "cruise_rate_scale": "cruise_rate_scale",
        })
    _copy_fields(_path(aircraft, "transition"), values, {
        "max_tilt_rate_deg_s": "max_tilt_rate_deg_s",
        "observer_time_constant_s": "observer_time_constant_s",
        "max_alpha_deg": "max_alpha_deg",
        "max_pitch_deg": "max_pitch_deg",
        "max_body_rate_deg_s": "max_body_rate_deg_s",
        "speed_margin_mps": "speed_margin_mps",
    })
    _copy_fields(_path(aircraft, "disturbance"), values, {
        "enabled": "gust_enabled",
        "model": "gust_model",
        "start_s": "gust_start_s",
        "duration_s": "gust_duration_s",
        "magnitude_mps": "gust_magnitude_mps",
        "direction": "gust_direction",
        "dryden_sigma_mps": "dryden_sigma_mps",
        "dryden_tau_s": "dryden_tau_s",
    })
    simulation = profile.get("simulation", {})
    overrides = simulation.get("dashboard_overrides", {}) if isinstance(simulation, Mapping) else {}
    if isinstance(overrides, Mapping):
        values.update({
            key: value for key, value in overrides.items()
            if key in DEFAULT_DASHBOARD_VALUES
        })
    _validate_dashboard_values(values)
    return values


def profile_with_dashboard_values(
    profile: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Save current supported controls while preserving every other profile field."""
    if not isinstance(profile, Mapping):
        raise ValueError("aircraft profile must be a mapping")
    unknown = set(values) - set(DEFAULT_DASHBOARD_VALUES)
    if unknown:
        raise ValueError(f"unsupported dashboard fields: {sorted(unknown)}")
    _validate_dashboard_values(values)
    updated = dict(profile)
    existing_simulation = updated.get("simulation", {})
    if not isinstance(existing_simulation, Mapping):
        raise ValueError("simulation must be a mapping")
    simulation = dict(existing_simulation)
    simulation["dashboard_overrides"] = dict(values)
    updated["simulation"] = simulation
    return updated


def simulation_compatibility_issues(profile: Mapping[str, Any]) -> tuple[str, ...]:
    """Explain where the current simulator cannot model a loaded aircraft yet."""
    aircraft = profile.get("aircraft", {})
    geometry = aircraft.get("geometry", {}) if isinstance(aircraft, Mapping) else {}
    profile_rotors = geometry.get("rotors") if isinstance(geometry, Mapping) else None
    if not isinstance(profile_rotors, list) or not profile_rotors:
        return ("Aircraft profile does not define aircraft.geometry.rotors.",)

    required_paths = (
        "mass_properties.component_masses_kg.fuselage",
        "mass_properties.component_masses_kg.wing",
        "mass_properties.component_masses_kg.vtail",
        "mass_properties.component_masses_kg.motor",
        "propulsion.battery.cells",
        "propulsion.battery.capacity_mah",
        "propulsion.battery.c_rating",
        "propulsion.motor.kv_rpm_per_volt",
        "propulsion.propeller.diameter_in",
        "propulsion.propeller.pitch_in",
        "propulsion.propeller.blade_count",
        "geometry.wing.root_le_x_m",
        "geometry.wing.root_y_m",
        "geometry.wing.tip_le_x_m",
        "geometry.wing.tip_y_m",
        "geometry.wing.root_chord_m",
        "geometry.wing.tip_chord_m",
        "geometry.wing.sweep_deg",
        "geometry.vtail.root_x_m",
        "geometry.vtail.tip_y_m",
        "geometry.vtail.tip_z_m",
        "geometry.vtail.area_m2",
        "geometry.vtail.arm_m",
        "control.altitude_pid.kp",
        "control.altitude_pid.ki",
        "control.altitude_pid.kd",
        "control.speed_pid.kp",
        "control.speed_pid.ki",
        "control.speed_pid.kd",
        "control.attitude_pid.kp",
        "control.attitude_pid.ki",
        "control.attitude_pid.kd",
        "control.rate_pid.kp",
        "control.rate_pid.ki",
        "control.rate_pid.kd",
        "control.hover_rate_scale",
        "control.cruise_rate_scale",
        "transition.max_tilt_rate_deg_s",
    )
    missing = [
        path for path in required_paths
        if _path(aircraft, path) is None or _path(aircraft, path) == {}
    ]

    simulation = profile.get("simulation", {})
    overrides = (
        simulation.get("dashboard_overrides", {})
        if isinstance(simulation, Mapping)
        else {}
    )
    if isinstance(overrides, Mapping) and "motor_positions" in overrides:
        simulated_rotors = overrides["motor_positions"]
        valid_simulated_rotors = all(
            isinstance(rotor, (list, tuple))
            and len(rotor) == 4
            and all(_is_finite_number(value) for value in rotor[:3])
            and isinstance(rotor[3], bool)
            for rotor in simulated_rotors
        )
        simulated_rotor_count = len(simulated_rotors)
    else:
        valid_simulated_rotors = True
        simulated_rotor_count = len(profile_rotors)

    valid_profile_rotors = all(
        isinstance(rotor, Mapping)
        and isinstance(rotor.get("position_m"), list)
        and len(rotor["position_m"]) == 3
        and all(_is_finite_number(value) for value in rotor["position_m"])
        and isinstance(rotor.get("tilt_enabled", False), bool)
        for rotor in profile_rotors
    )
    issues = []
    if missing:
        issues.append(
            "Required simulator inputs are missing: " + ", ".join(missing) + "."
        )
    if len(profile_rotors) != 4 or simulated_rotor_count != 4:
        issues.append(
            "The current flight-dynamics and allocator implementation supports exactly four rotors."
        )
    architecture = aircraft.get("architecture", "tiltrotor")
    if architecture != "tiltrotor":
        issues.append("The current flight-dynamics implementation supports tiltrotor aircraft only.")
    if not valid_profile_rotors or not valid_simulated_rotors:
        issues.append("Every rotor needs finite x/y/z coordinates and a tilt flag.")
    return tuple(issues)


def serialize_aircraft_profile(profile: Mapping[str, Any]) -> bytes:
    """Serialize a validated profile as portable UTF-8 JSON."""
    validate_aircraft_profile(profile)
    return (json.dumps(profile, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def count_profile_values(value: Any) -> int:
    """Count scalar entries without materializing a flattened copy of the profile."""
    if isinstance(value, Mapping):
        return sum(count_profile_values(child) for child in value.values())
    if isinstance(value, list):
        return sum(count_profile_values(child) for child in value)
    return 0 if value is None else 1


def _validate_data_tree(
    value: Any,
    path: str,
    active_containers: set[int],
    node_count: list[int],
    depth: int,
) -> None:
    node_count[0] += 1
    if node_count[0] > MAX_PROFILE_NODES:
        raise ValueError(f"profile exceeds the {MAX_PROFILE_NODES:,} value limit")
    if depth > MAX_PROFILE_DEPTH:
        raise ValueError(f"{path} exceeds the maximum nesting depth of {MAX_PROFILE_DEPTH}")
    if isinstance(value, (Mapping, list)):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} contains a recursive data structure")
        active_containers.add(identity)
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{path} keys must be strings")
                _validate_data_tree(
                    child, f"{path}.{key}", active_containers, node_count, depth + 1
                )
        else:
            for index, child in enumerate(value):
                _validate_data_tree(
                    child, f"{path}[{index}]", active_containers, node_count, depth + 1
                )
        active_containers.remove(identity)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _path(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = mapping
    for segment in dotted_path.split("."):
        if not isinstance(value, Mapping):
            return {}
        value = value.get(segment, {})
    return value


def _copy_fields(source: Any, destination: dict[str, Any], fields: Mapping[str, str]) -> None:
    if not isinstance(source, Mapping):
        return
    for source_name, destination_name in fields.items():
        value = source.get(source_name)
        if value is not None:
            destination[destination_name] = value


def _validate_dashboard_values(values: Mapping[str, Any]) -> None:
    for key, expected in DEFAULT_DASHBOARD_VALUES.items():
        value = values[key]
        if key == "motor_positions":
            if not isinstance(value, list) or not value:
                raise ValueError("aircraft geometry must define at least one rotor")
            for index, position in enumerate(value):
                if (
                    not isinstance(position, (list, tuple))
                    or len(position) != 4
                    or not all(_is_finite_number(coordinate) for coordinate in position[:3])
                    or not isinstance(position[3], bool)
                ):
                    raise ValueError(f"motor_positions[{index}] must contain x/y/z and a tilt flag")
            continue
        if isinstance(expected, bool) and not isinstance(value, bool):
            raise ValueError(f"dashboard parameter {key} must be a boolean")
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not _is_finite_number(value):
                raise ValueError(f"dashboard parameter {key} must be finite numeric data")
        if isinstance(expected, str) and not isinstance(value, str):
            raise ValueError(f"dashboard parameter {key} must be a string")
        if key == "battery_cells" and not isinstance(value, int):
            raise ValueError("dashboard parameter battery_cells must be an integer")
        if key == "prop_blades" and not isinstance(value, int):
            raise ValueError("dashboard parameter prop_blades must be an integer")

    if values["battery_cells"] <= 0 or values["battery_capacity_mah"] <= 0:
        raise ValueError("battery cells and capacity must be positive")
    if values["battery_c_rating"] <= 0 or values["motor_kv"] <= 0:
        raise ValueError("battery C-rating and motor KV must be positive")

    ranges = {
        "prop_blades": (2, 5),
        "max_tilt_rate_deg_s": (1.0, 90.0),
        "gust_start_s": (0.0, 15.0),
        "gust_duration_s": (0.5, 5.0),
        "gust_magnitude_mps": (0.0, 15.0),
        "dryden_sigma_mps": (0.0, 8.0),
        "dryden_tau_s": (0.1, 5.0),
        "observer_time_constant_s": (0.05, 1.0),
        "max_alpha_deg": (5.0, 30.0),
        "max_pitch_deg": (20.0, 60.0),
        "max_body_rate_deg_s": (30.0, 240.0),
        "speed_margin_mps": (0.0, 3.0),
    }
    for key, (minimum, maximum) in ranges.items():
        value = values[key]
        if not minimum <= value <= maximum:
            raise ValueError(f"dashboard parameter {key} must be between {minimum} and {maximum}")

    if values["gust_model"] not in ("Finite-duration gust", "Dryden turbulence"):
        raise ValueError("disturbance.model must be a supported gust model")
    if values["gust_direction"] not in ("Front", "Back", "Left", "Right", "Up", "Down", "Diagonal"):
        raise ValueError("disturbance.direction must be a supported direction")
    if any(values[key] <= 0.0 for key in (
        "fuselage_mass_kg",
        "wing_mass_kg",
        "vtail_mass_kg",
        "motor_mass_kg",
        "fuselage_length_m",
        "fuselage_width_m",
        "fuselage_height_m",
        "prop_diameter_in",
        "prop_pitch_in",
        "wing_root_chord_m",
        "wing_tip_chord_m",
        "wing_tip_y_m",
        "vtail_tip_y_m",
        "vtail_area_m2",
        "vtail_arm_m",
    )):
        raise ValueError("mass, battery, motor, propeller, and primary surface dimensions must be positive")
