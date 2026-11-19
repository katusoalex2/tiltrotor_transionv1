"""Build a non-deployable PX4 SITL mapping from an aircraft profile."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


PX4_ACTUATOR_DOCS = "https://docs.px4.io/main/en/config/actuators.html"
PX4_TILTROTOR_DOCS = "https://docs.px4.io/main/en/frames_vtol/tiltrotor.html"


def build_px4_sitl_mapping(
    profile: Mapping[str, Any],
    *,
    px4_version: str,
    flight_controller: str,
    motor_directions: Mapping[str, str] | None = None,
    tilt_servo_arrangement: str = "UNSPECIFIED",
) -> dict[str, Any]:
    """Preserve profile geometry as an explicitly non-flashable PX4 SITL draft."""
    aircraft = profile.get("aircraft")
    if not isinstance(aircraft, Mapping):
        raise ValueError("profile.aircraft must be a mapping")
    geometry = aircraft.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("profile.aircraft.geometry must be a mapping")
    rotors = geometry.get("rotors")
    if not isinstance(rotors, list) or not rotors:
        raise ValueError("profile.aircraft.geometry.rotors must be a non-empty list")
    actuators = aircraft.get("actuators")
    if not isinstance(actuators, Mapping):
        raise ValueError("profile.aircraft.actuators must be a mapping")
    tilt = actuators.get("tilt", {})
    surfaces = actuators.get("control_surfaces", [])
    if not isinstance(tilt, Mapping) or not isinstance(surfaces, list):
        raise ValueError("profile actuator tilt and control-surface data have invalid shapes")
    if any(not isinstance(surface, Mapping) for surface in surfaces):
        raise ValueError("each profile control surface must be a mapping")

    hardware = profile.get("hardware", {})
    if not isinstance(hardware, Mapping):
        raise ValueError("profile.hardware must be a mapping")
    if hardware.get("actuator_outputs_enabled", False) is not False:
        raise ValueError("PX4 SITL mapping requires hardware actuator outputs to be disabled")

    simulation = profile.get("simulation", {})
    if not isinstance(simulation, Mapping):
        raise ValueError("profile.simulation must be a mapping")
    dashboard_overrides = simulation.get("dashboard_overrides", {})
    if not isinstance(dashboard_overrides, Mapping):
        raise ValueError("profile.simulation.dashboard_overrides must be a mapping")
    rotor_overrides = dashboard_overrides.get("motor_positions")
    if rotor_overrides is not None:
        if not isinstance(rotor_overrides, list) or len(rotor_overrides) != len(rotors):
            raise ValueError("dashboard motor-position overrides must match the profile rotor count")
        for index, position_override in enumerate(rotor_overrides):
            if (
                not isinstance(position_override, (list, tuple))
                or len(position_override) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in position_override[:3]
                )
                or not isinstance(position_override[3], bool)
            ):
                raise ValueError(
                    f"dashboard motor-position override {index + 1} must contain "
                    "finite x/y/z coordinates and a tilt flag"
                )

    arrangement = tilt_servo_arrangement.strip().upper()
    valid_arrangements = {"UNSPECIFIED", "SHARED_SERVO", "INDEPENDENT_SERVOS"}
    if arrangement not in valid_arrangements:
        raise ValueError(
            "tilt servo arrangement must be UNSPECIFIED, SHARED_SERVO, or INDEPENDENT_SERVOS"
        )

    if motor_directions is not None and not isinstance(motor_directions, Mapping):
        raise ValueError("motor directions must be provided as a mapping by rotor ID")
    directions = motor_directions or {}
    direction_values = {"UNVERIFIED", "CW", "CCW"}
    mapped_rotors: list[dict[str, Any]] = []
    unresolved: list[str] = []
    seen_rotor_ids: set[str] = set()
    for index, rotor in enumerate(rotors):
        if not isinstance(rotor, Mapping):
            raise ValueError(f"profile rotor {index + 1} must be a mapping")
        rotor_id = str(rotor.get("id", f"rotor_{index + 1}"))
        if rotor_id in seen_rotor_ids:
            raise ValueError(f"profile rotor IDs must be unique; duplicate ID: {rotor_id}")
        seen_rotor_ids.add(rotor_id)
        if rotor_overrides is None:
            position = rotor.get("position_m")
            tilts = bool(rotor.get("tilt_enabled", False))
        else:
            position_override = rotor_overrides[index]
            position = position_override[:3]
            tilts = position_override[3]
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in position
            )
        ):
            raise ValueError(f"profile rotor {rotor_id} position_m must be a finite 3-vector")
        direction = str(directions.get(rotor_id, "UNVERIFIED")).upper()
        if direction not in direction_values:
            raise ValueError(f"motor direction for {rotor_id} must be CW, CCW, or UNVERIFIED")
        if direction == "UNVERIFIED":
            unresolved.append(f"Confirm motor rotation direction for {rotor_id}.")
        mapped_rotors.append({
            "profile_id": rotor_id,
            "logical_px4_function": f"Motor{index + 1}",
            "position_profile_m": [float(value) for value in position],
            "position_frame_conversion": "UNVERIFIED; PX4 actuator geometry uses FRD about CG",
            "tilt_enabled": tilts,
            "tilt_servo_association": (
                arrangement if tilts else "NOT_APPLICABLE"
            ),
            "rotation_direction": direction,
            "physical_output_assignment": None,
        })

    if any(
        bool(rotor.get("tilt_enabled", False))
        if rotor_overrides is None
        else rotor_overrides[index][3]
        for index, rotor in enumerate(rotors)
    ):
        if arrangement == "UNSPECIFIED":
            unresolved.append("Confirm whether one or multiple servos drive the tilting rotors.")
        unresolved.append("Set and verify tilt-servo direction, travel, endpoints, and feedback.")
    unresolved.append(
        "Confirm the profile coordinate convention and transform motor positions to PX4 FRD about CG."
    )
    if not px4_version.strip() or px4_version.strip().lower() == "unknown":
        unresolved.append("Select and verify the exact PX4 release.")
    if not flight_controller.strip() or flight_controller.strip().lower() == "unknown":
        unresolved.append("Select and verify the flight-controller board and output protocol.")
    if not any(
        rotor["rotation_direction"] in {"CW", "CCW"}
        for rotor in mapped_rotors
    ):
        unresolved.append("No motor rotation directions have been verified.")
    if any(surface.get("fail_position_deg") is None for surface in surfaces if isinstance(surface, Mapping)):
        unresolved.append("Verify control-surface fail positions from the actual hardware.")

    propulsion = aircraft.get("propulsion", {})
    propulsion = propulsion if isinstance(propulsion, Mapping) else {}
    battery = propulsion.get("battery", {})
    motor = propulsion.get("motor", {})
    propeller = propulsion.get("propeller", {})
    battery = battery if isinstance(battery, Mapping) else {}
    motor = motor if isinstance(motor, Mapping) else {}
    propeller = propeller if isinstance(propeller, Mapping) else {}
    for label, section, field in (
        ("Battery internal resistance", battery, "internal_resistance_ohm"),
        ("Battery continuous current", battery, "continuous_current_a"),
        ("Motor winding resistance", motor, "resistance_ohm"),
        ("Motor no-load current", motor, "no_load_current_a"),
        ("Motor continuous current", motor, "continuous_current_a"),
        ("Propeller torque coefficient", propeller, "torque_coefficient_nm_s2_rad2"),
    ):
        if section.get(field) is None:
            unresolved.append(f"Provide measured {label.lower()} for hardware validation.")
    esc = propulsion.get("esc", {})
    esc = esc if isinstance(esc, Mapping) else {}
    if esc.get("efficiency") is None or esc.get("continuous_current_a") is None:
        unresolved.append("Provide verified ESC efficiency and continuous-current limits.")
    motor_dynamics = propulsion.get("motor_dynamics", {})
    motor_dynamics = motor_dynamics if isinstance(motor_dynamics, Mapping) else {}
    if motor_dynamics.get("slew_rate_rpm_s") is None:
        unresolved.append("Provide a measured or manufacturer-specified motor RPM slew limit.")

    metadata = profile.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    px4_version_value = px4_version.strip() or "unknown"
    flight_controller_value = flight_controller.strip() or "unknown"
    return {
        "mapping_schema_version": 1,
        "target": {
            "autopilot": "PX4",
            "px4_version": px4_version_value,
            "flight_controller": flight_controller_value,
            "candidate_airframe_family": (
                "VTOL tiltrotor; verify support for this mixed tilting/fixed-rotor layout "
                "in the selected PX4 release"
            ),
        },
        "scope": "SITL_MAPPING_ONLY",
        "flashable": False,
        "px4_parameter_file_generated": False,
        "actuator_commands_generated": False,
        "source_profile": {
            "schema_version": profile.get("schema_version"),
            "id": metadata.get("id"),
            "name": metadata.get("name"),
            "revision": metadata.get("revision"),
            "description": metadata.get("description"),
        },
        "source_hardware_outputs_enabled": hardware.get("actuator_outputs_enabled", False),
        "actuator_geometry": {
            "coordinate_frame": "Profile coordinates preserved; conversion to PX4 FRD is unverified",
            "rotors": mapped_rotors,
            "tilt": {
                "maximum_rate_deg_s": tilt.get("maximum_rate_deg_s"),
                "angle_limits_deg": tilt.get("angle_limits_deg"),
                "feedback_sensor": tilt.get("feedback_sensor"),
                "servo_arrangement": arrangement,
                "physical_output_assignment": None,
            },
            "control_surfaces": [
                {
                    "profile_id": surface.get("id"),
                    "profile_type": surface.get("type"),
                    "maximum_deflection_deg": surface.get("maximum_deflection_deg"),
                    "rate_limit_deg_s": surface.get("rate_limit_deg_s"),
                    "fail_position_deg": surface.get("fail_position_deg"),
                    "physical_output_assignment": None,
                }
                for surface in surfaces
                if isinstance(surface, Mapping)
            ],
        },
        "qgroundcontrol_mapping": [
            {
                "profile_path": "aircraft.geometry.rotors",
                "setup_area": "Vehicle Setup > Actuators > Geometry > Motors",
                "status": "Positions preserved; verify coordinate frame, motor order, and rotation.",
            },
            {
                "profile_path": "aircraft.geometry.rotors[*].tilt_enabled",
                "setup_area": "Vehicle Setup > Actuators > Geometry > Motor Tilt Servo",
                "status": "Tilt associations remain unassigned until servo topology is confirmed.",
            },
            {
                "profile_path": "aircraft.actuators.control_surfaces",
                "setup_area": "Vehicle Setup > Actuators > Geometry > Control Surfaces",
                "status": "V-tail functions and control signs require hardware verification.",
            },
        ],
        "unresolved_before_hardware_configuration": list(dict.fromkeys(unresolved)),
        "validation_stages": [
            "Verify the candidate PX4 airframe and output functions for the selected release.",
            "Configure and inspect the vehicle in PX4 SITL before connecting hardware.",
            "Verify actuator direction and limits with propellers removed.",
            "Complete PX4 HITL and independent safety review before any flight activity.",
        ],
        "documentation": {
            "px4_actuator_setup": PX4_ACTUATOR_DOCS,
            "px4_tiltrotor_vtol": PX4_TILTROTOR_DOCS,
        },
    }


def serialize_px4_sitl_mapping(
    profile: Mapping[str, Any],
    *,
    px4_version: str,
    flight_controller: str,
    motor_directions: Mapping[str, str] | None = None,
    tilt_servo_arrangement: str = "UNSPECIFIED",
) -> bytes:
    """Serialize a PX4 SITL mapping; this output is never a PX4 parameter file."""
    mapping = build_px4_sitl_mapping(
        profile,
        px4_version=px4_version,
        flight_controller=flight_controller,
        motor_directions=motor_directions,
        tilt_servo_arrangement=tilt_servo_arrangement,
    )
    return (json.dumps(mapping, indent=2, allow_nan=False) + "\n").encode("utf-8")
