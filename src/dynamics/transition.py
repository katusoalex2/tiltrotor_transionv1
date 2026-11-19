"""Transition-envelope, trim, and local control-effectiveness analysis."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares

from src.dynamics.state import AircraftState
from src.dynamics.frames import rotation_earth_to_body
from src.dynamics.tiltrotor import GRAVITY, TiltrotorDynamics


def transition_speed_envelope(
    mass: float,
    wing_area: float,
    max_thrust_per_rotor: float,
    rotor_tilts: list[bool],
    *,
    rho: float = 1.225,
    cl_max: float,
    cd_0: float,
    aspect_ratio: float,
    oswald_efficiency: float = 0.8,
    samples: int = 46,
) -> dict[str, np.ndarray]:
    """Estimate level-flight speed limits from rotor support and the wing drag polar.

    The envelope is quasi-steady: it omits actuator-rate, power, gust, and pitch-moment
    constraints. Rotor thrust is split equally between the configured rotor positions.
    """
    if mass <= 0.0 or wing_area <= 0.0 or max_thrust_per_rotor < 0.0:
        raise ValueError("mass and wing area must be positive; rotor thrust cannot be negative")
    if not rotor_tilts:
        raise ValueError("at least one rotor must be configured")
    if rho <= 0.0 or cl_max <= 0.0 or aspect_ratio <= 0.0 or oswald_efficiency <= 0.0:
        raise ValueError("air density, CL max, aspect ratio, and efficiency must be positive")
    if cd_0 <= 0.0 or samples < 2:
        raise ValueError("zero-lift drag must be positive and at least two tilt samples are required")

    tilt_deg = np.linspace(0.0, 90.0, samples)
    tilt_rad = np.deg2rad(tilt_deg)
    n_tilting = sum(bool(tilts) for tilts in rotor_tilts)
    n_fixed = len(rotor_tilts) - n_tilting

    max_vertical_rotor = max_thrust_per_rotor * (
        n_fixed + n_tilting * np.cos(tilt_rad)
    )
    max_forward_rotor = max_thrust_per_rotor * n_tilting * np.sin(tilt_rad)
    wing_lift_required = np.maximum(mass * GRAVITY - max_vertical_rotor, 0.0)
    minimum_stall_speed = np.sqrt(
        2.0 * wing_lift_required / (rho * wing_area * cl_max)
    )

    parasite_drag_factor = 0.5 * rho * wing_area * cd_0
    induced_drag_factor = wing_lift_required**2 / (
        0.5 * rho * wing_area * np.pi * oswald_efficiency * aspect_ratio
    )
    discriminant = max_forward_rotor**2 - 4.0 * parasite_drag_factor * induced_drag_factor
    feasible = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    lower_drag_speed = np.sqrt(
        np.maximum((max_forward_rotor - root) / (2.0 * parasite_drag_factor), 0.0)
    )
    upper_speed = np.sqrt(
        np.maximum((max_forward_rotor + root) / (2.0 * parasite_drag_factor), 0.0)
    )
    minimum_speed = np.maximum(minimum_stall_speed, lower_drag_speed)
    feasible &= minimum_speed <= upper_speed

    return {
        "tilt_deg": tilt_deg,
        "minimum_speed_mps": np.where(feasible, minimum_speed, np.nan),
        "maximum_speed_mps": np.where(feasible, upper_speed, np.nan),
        "feasible": feasible,
    }


def solve_longitudinal_trim(
    dynamics: TiltrotorDynamics,
    airspeed_mps: float,
    tilt_rad: float,
    *,
    tail_limit_rad: float = 0.2,
    residual_tolerance: float = 0.02,
    max_rotor_omega: float | None = None,
) -> dict[str, float | bool | np.ndarray | str]:
    """Solve steady, level longitudinal force and pitch-moment balance.

    The solver varies pitch, front and rear pair speeds, and symmetric V-tail
    deflection. A reported trim is feasible only when normalized force and moment
    residuals fall below ``residual_tolerance``.
    """
    if dynamics.aero is None:
        raise ValueError("longitudinal trim requires an aerodynamic model")
    if airspeed_mps < 0.0 or not 0.0 <= tilt_rad <= np.pi / 2.0:
        raise ValueError("airspeed must be non-negative and tilt must be between 0 and 90 degrees")
    if tail_limit_rad <= 0.0 or residual_tolerance <= 0.0:
        raise ValueError("tail limit and trim tolerance must be positive")

    aero = dynamics.aero
    rotor = dynamics.rotor
    omega_limit = rotor.max_omega if max_rotor_omega is None else float(max_rotor_omega)
    if not np.isfinite(omega_limit) or omega_limit <= 0.0 or omega_limit > rotor.max_omega:
        raise ValueError("trim motor speed limit must be positive and within the rotor model limit")
    chord = max(float(aero.chord), 0.01)
    hover_omega = np.sqrt(dynamics.mass * GRAVITY / max(4.0 * rotor.k_thrust, 1e-12))
    initial = np.array([0.0, min(hover_omega, omega_limit), min(hover_omega, omega_limit), 0.0])
    lower = np.array([-aero.stall_angle, 0.0, 0.0, -tail_limit_rad])
    upper = np.array([aero.stall_angle, omega_limit, omega_limit, tail_limit_rad])
    wind_ned = np.zeros(3)

    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        alpha, front_omega, rear_omega, tail_deflection = candidate
        state = AircraftState(
            pos=np.array([0.0, 0.0, -100.0]),
            vel=np.array([airspeed_mps, 0.0, 0.0]),
            att=np.array([0.0, alpha, 0.0]),
            tilt=float(tilt_rad),
        )
        dynamics.step(
            state,
            wind_ned,
            float(front_omega),
            float(front_omega),
            float(rear_omega),
            float(rear_omega),
            float(tilt_rad),
            float(tail_deflection),
            float(tail_deflection),
            0.01,
        )
        diagnostic = dynamics.last_diagnostics
        residual = np.array([
            diagnostic["acceleration_ned"][0] / GRAVITY,
            diagnostic["acceleration_ned"][2] / GRAVITY,
            diagnostic["moment_body"][1] / (dynamics.mass * GRAVITY * chord),
        ])
        return residual, diagnostic

    result = least_squares(
        lambda candidate: evaluate(candidate)[0],
        initial,
        bounds=(lower, upper),
        x_scale=np.array([
            max(float(aero.stall_angle), 1e-3),
            max(float(hover_omega), 1.0),
            max(float(hover_omega), 1.0),
            max(float(tail_limit_rad), 1e-3),
        ]),
        max_nfev=300,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    residual, diagnostic = evaluate(result.x)
    max_residual = float(np.max(np.abs(residual)))

    return {
        "alpha_deg": float(np.rad2deg(result.x[0])),
        "motor_rpm": float(np.mean(result.x[1:3]) * 60.0 / (2.0 * np.pi)),
        "front_motor_rpm": float(result.x[1] * 60.0 / (2.0 * np.pi)),
        "rear_motor_rpm": float(result.x[2] * 60.0 / (2.0 * np.pi)),
        "tail_deflection_deg": float(np.rad2deg(result.x[3])),
        "acceleration_ned": diagnostic["acceleration_ned"].copy(),
        "moment_body": diagnostic["moment_body"].copy(),
        "angular_acceleration_body": diagnostic["angular_acceleration_body"].copy(),
        "normalized_residual": residual.copy(),
        "max_normalized_residual": max_residual,
        "feasible": bool(result.success and max_residual <= residual_tolerance),
        "solver_message": str(result.message),
        "evaluations": int(result.nfev),
    }


def control_effectiveness_map(
    dynamics: TiltrotorDynamics,
    airspeed_mps: float,
    tilt_rad: float,
    alpha_rad: float,
    collective_omega: float,
    tail_deflection_rad: float = 0.0,
    *,
    pair_omegas: tuple[float, float] | None = None,
    max_rotor_omega: float | None = None,
    tail_limit_rad: float = 0.2,
    rotor_step_rpm: float = 50.0,
    tail_step_deg: float = 5.0,
) -> tuple[list[str], np.ndarray]:
    """Estimate local moment changes around collective or front/rear pair speeds."""
    if dynamics.aero is None:
        raise ValueError("control effectiveness requires an aerodynamic model")
    if airspeed_mps < 0.0 or not 0.0 <= tilt_rad <= np.pi / 2.0:
        raise ValueError("airspeed must be non-negative and tilt must be between 0 and 90 degrees")

    omega_step = rotor_step_rpm * 2.0 * np.pi / 60.0
    tail_step = np.deg2rad(tail_step_deg)
    patterns = [
        ("Rotor roll (±50 RPM)", np.array([1.0, -1.0, 1.0, -1.0]), "rotor"),
        ("Rotor pitch (±50 RPM)", np.array([1.0, 1.0, -1.0, -1.0]), "rotor"),
        ("Rotor yaw (±50 RPM)", np.array([1.0, -1.0, -1.0, 1.0]), "rotor"),
        ("V-tail symmetric (±5°)", np.zeros(4), "symmetric"),
        ("V-tail differential (±5°)", np.zeros(4), "differential"),
    ]
    omega_limit = dynamics.rotor.max_omega if max_rotor_omega is None else float(max_rotor_omega)
    if not np.isfinite(omega_limit) or omega_limit <= 0.0 or omega_limit > dynamics.rotor.max_omega:
        raise ValueError("motor speed limit must be positive and within the rotor model limit")
    if pair_omegas is None:
        base_omega = np.full(4, collective_omega, dtype=float)
    else:
        pair_speeds = np.asarray(pair_omegas, dtype=float)
        if pair_speeds.shape != (2,) or not np.all(np.isfinite(pair_speeds)) or np.any(
            (pair_speeds < 0.0) | (pair_speeds > omega_limit)
        ):
            raise ValueError("pair speeds must be finite and within motor bounds")
        base_omega = np.repeat(pair_speeds, 2)
    labels: list[str] = []
    effectiveness = np.zeros((3, len(patterns)), dtype=float)

    def evaluate(omegas: np.ndarray, delta_left: float, delta_right: float) -> np.ndarray:
        state = AircraftState(
            pos=np.array([0.0, 0.0, -100.0]),
            vel=np.array([airspeed_mps, 0.0, 0.0]),
            att=np.array([0.0, alpha_rad, 0.0]),
            tilt=float(tilt_rad),
        )
        dynamics.step(
            state,
            np.zeros(3),
            *omegas.tolist(),
            float(tilt_rad),
            float(delta_left),
            float(delta_right),
            0.01,
        )
        return dynamics.last_diagnostics["moment_body"].copy()

    for column, (label, pattern, kind) in enumerate(patterns):
        labels.append(label)
        if kind == "rotor":
            plus = np.clip(base_omega + omega_step * pattern, 0.0, omega_limit)
            minus = np.clip(base_omega - omega_step * pattern, 0.0, omega_limit)
            moment_plus = evaluate(plus, tail_deflection_rad, tail_deflection_rad)
            moment_minus = evaluate(minus, tail_deflection_rad, tail_deflection_rad)
        elif kind == "symmetric":
            plus_delta = np.clip(tail_deflection_rad + tail_step, -tail_limit_rad, tail_limit_rad)
            minus_delta = np.clip(tail_deflection_rad - tail_step, -tail_limit_rad, tail_limit_rad)
            moment_plus = evaluate(base_omega, plus_delta, plus_delta)
            moment_minus = evaluate(base_omega, minus_delta, minus_delta)
        else:
            plus_left = np.clip(tail_deflection_rad + tail_step, -tail_limit_rad, tail_limit_rad)
            plus_right = np.clip(tail_deflection_rad - tail_step, -tail_limit_rad, tail_limit_rad)
            minus_left = np.clip(tail_deflection_rad - tail_step, -tail_limit_rad, tail_limit_rad)
            minus_right = np.clip(tail_deflection_rad + tail_step, -tail_limit_rad, tail_limit_rad)
            moment_plus = evaluate(base_omega, plus_left, plus_right)
            moment_minus = evaluate(base_omega, minus_left, minus_right)
        effectiveness[:, column] = 0.5 * (moment_plus - moment_minus)

    return labels, effectiveness


def transition_stability_map(
    dynamics: TiltrotorDynamics,
    airspeeds_mps: list[float] | np.ndarray,
    tilt_angles_rad: list[float] | np.ndarray,
    *,
    tail_limit_rad: float = 0.2,
    perturbation_rad: float = np.deg2rad(0.5),
    trim_residual_tolerance: float = 0.02,
    max_rotor_omega: float | None = None,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    """Map local static-stability derivatives and actuator response over transition.

    Derivatives are evaluated about the longitudinal trim solution, or the
    optimizer's best-fit point when trim is infeasible. Only trim-feasible
    points above the aerodynamic model's low-speed cutoff are assessment
    candidates; the returned residual and mask must be considered with the
    derivative values.
    """
    if dynamics.aero is None:
        raise ValueError("transition stability analysis requires an aerodynamic model")

    airspeeds = np.asarray(airspeeds_mps, dtype=float)
    tilts = np.asarray(tilt_angles_rad, dtype=float)
    if (
        airspeeds.ndim != 1
        or airspeeds.size == 0
        or not np.all(np.isfinite(airspeeds))
        or np.any(airspeeds < 0.0)
        or (airspeeds.size > 1 and np.any(np.diff(airspeeds) <= 0.0))
    ):
        raise ValueError("airspeeds must be a non-empty, increasing finite non-negative sequence")
    if (
        tilts.ndim != 1
        or tilts.size == 0
        or not np.all(np.isfinite(tilts))
        or np.any((tilts < 0.0) | (tilts > np.pi / 2.0))
        or (tilts.size > 1 and np.any(np.diff(tilts) <= 0.0))
    ):
        raise ValueError("tilt angles must be an increasing finite sequence between 0 and 90 degrees")
    if (
        not np.isfinite(tail_limit_rad)
        or tail_limit_rad <= 0.0
        or not np.isfinite(perturbation_rad)
        or perturbation_rad <= 0.0
        or not np.isfinite(trim_residual_tolerance)
        or trim_residual_tolerance <= 0.0
    ):
        raise ValueError("tail limit, perturbation, and trim tolerance must be positive and finite")

    aero = dynamics.aero
    wing_area = float(aero.wing_area)
    reference_chord = float(aero.chord)
    if not np.all(np.isfinite([wing_area, reference_chord, aero.rho])) or min(
        wing_area, reference_chord, float(aero.rho)
    ) <= 0.0:
        raise ValueError("aerodynamic reference area, chord, and density must be positive and finite")

    mass_properties = dynamics.mass_props
    span_value = getattr(mass_properties, "wingspan", None)
    reference_span = (
        float(span_value)
        if span_value is not None
        else float(np.sqrt(wing_area * float(aero.AR)))
    )
    if not np.isfinite(reference_span) or reference_span <= 0.0:
        raise ValueError("reference span must be positive and finite")

    shape = (airspeeds.size, tilts.size)
    dynamic_pressure = np.broadcast_to(
        0.5 * float(aero.rho) * airspeeds[:, np.newaxis] ** 2,
        shape,
    ).copy()
    trim_feasible = np.zeros(shape, dtype=bool)
    trim_residual = np.full(shape, np.nan, dtype=float)
    trim_motor_rpm = np.full(shape, np.nan, dtype=float)
    trim_front_motor_rpm = np.full(shape, np.nan, dtype=float)
    trim_rear_motor_rpm = np.full(shape, np.nan, dtype=float)
    trim_pitch_moment_residual = np.full(shape, np.nan, dtype=float)
    cm_alpha = np.full(shape, np.nan, dtype=float)
    cl_beta = np.full(shape, np.nan, dtype=float)
    cn_beta = np.full(shape, np.nan, dtype=float)
    coefficient_valid = np.broadcast_to(
        (airspeeds >= 0.5)[:, np.newaxis],
        shape,
    ).copy()
    control_effectiveness = np.full(shape + (3, 5), np.nan, dtype=float)
    effectiveness_labels: tuple[str, ...] | None = None

    def evaluate_moment(
        airspeed: float,
        alpha: float,
        beta: float,
        tilt: float,
        pair_omegas: tuple[float, float],
        tail_deflection: float,
    ) -> np.ndarray:
        velocity_body = np.array([
            airspeed * np.cos(alpha) * np.cos(beta),
            airspeed * np.sin(beta),
            airspeed * np.sin(alpha) * np.cos(beta),
        ])
        state = AircraftState(
            pos=np.array([0.0, 0.0, -100.0]),
            vel=velocity_body,
            att=np.zeros(3),
            tilt=float(tilt),
        )
        dynamics.step(
            state,
            np.zeros(3),
            pair_omegas[0],
            pair_omegas[0],
            pair_omegas[1],
            pair_omegas[1],
            float(tilt),
            float(tail_deflection),
            float(tail_deflection),
            0.01,
        )
        return np.asarray(dynamics.last_diagnostics["moment_body"], dtype=float).copy()

    for speed_index, airspeed in enumerate(airspeeds):
        for tilt_index, tilt in enumerate(tilts):
            trim = solve_longitudinal_trim(
                dynamics,
                float(airspeed),
                float(tilt),
                tail_limit_rad=tail_limit_rad,
                residual_tolerance=trim_residual_tolerance,
                max_rotor_omega=max_rotor_omega,
            )
            trim_feasible[speed_index, tilt_index] = bool(trim["feasible"])
            trim_residual[speed_index, tilt_index] = float(trim["max_normalized_residual"])
            trim_motor_rpm[speed_index, tilt_index] = float(trim["motor_rpm"])
            trim_front_motor_rpm[speed_index, tilt_index] = float(trim["front_motor_rpm"])
            trim_rear_motor_rpm[speed_index, tilt_index] = float(trim["rear_motor_rpm"])
            trim_pitch_moment_residual[speed_index, tilt_index] = float(
                np.asarray(trim["normalized_residual"], dtype=float)[2]
            )

            pitch_rad = float(np.deg2rad(trim["alpha_deg"]))
            tail_deflection = float(np.deg2rad(trim["tail_deflection_deg"]))
            rotor_omega = float(trim["motor_rpm"]) * 2.0 * np.pi / 60.0
            pair_omegas = (
                float(trim["front_motor_rpm"]) * 2.0 * np.pi / 60.0,
                float(trim["rear_motor_rpm"]) * 2.0 * np.pi / 60.0,
            )
            labels, effectiveness = control_effectiveness_map(
                dynamics,
                airspeed_mps=float(airspeed),
                tilt_rad=float(tilt),
                alpha_rad=pitch_rad,
                collective_omega=rotor_omega,
                pair_omegas=pair_omegas,
                max_rotor_omega=max_rotor_omega,
                tail_deflection_rad=tail_deflection,
                tail_limit_rad=tail_limit_rad,
            )
            if effectiveness_labels is None:
                effectiveness_labels = tuple(labels)
            control_effectiveness[speed_index, tilt_index] = effectiveness

            q_bar = dynamic_pressure[speed_index, tilt_index]
            if not coefficient_valid[speed_index, tilt_index] or q_bar <= 0.0:
                continue

            trim_velocity_body = rotation_earth_to_body(
                0.0,
                pitch_rad,
                0.0,
            ) @ np.array([airspeed, 0.0, 0.0])
            alpha_trim = float(np.arctan2(trim_velocity_body[2], trim_velocity_body[0]))
            beta_trim = float(np.arcsin(np.clip(
                trim_velocity_body[1] / airspeed,
                -1.0,
                1.0,
            )))
            moment_alpha_plus = evaluate_moment(
                float(airspeed),
                alpha_trim + perturbation_rad,
                beta_trim,
                float(tilt),
                pair_omegas,
                tail_deflection,
            )
            moment_alpha_minus = evaluate_moment(
                float(airspeed),
                alpha_trim - perturbation_rad,
                beta_trim,
                float(tilt),
                pair_omegas,
                tail_deflection,
            )
            moment_beta_plus = evaluate_moment(
                float(airspeed),
                alpha_trim,
                beta_trim + perturbation_rad,
                float(tilt),
                pair_omegas,
                tail_deflection,
            )
            moment_beta_minus = evaluate_moment(
                float(airspeed),
                alpha_trim,
                beta_trim - perturbation_rad,
                float(tilt),
                pair_omegas,
                tail_deflection,
            )
            moment_scale_pitch = q_bar * wing_area * reference_chord
            moment_scale_lateral = q_bar * wing_area * reference_span
            cm_alpha[speed_index, tilt_index] = (
                (moment_alpha_plus[1] - moment_alpha_minus[1])
                / (2.0 * perturbation_rad * moment_scale_pitch)
            )
            cl_beta[speed_index, tilt_index] = (
                (moment_beta_plus[0] - moment_beta_minus[0])
                / (2.0 * perturbation_rad * moment_scale_lateral)
            )
            cn_beta[speed_index, tilt_index] = (
                (moment_beta_plus[2] - moment_beta_minus[2])
                / (2.0 * perturbation_rad * moment_scale_lateral)
            )

    assessment_mask = trim_feasible & coefficient_valid
    return {
        "airspeeds_mps": airspeeds.copy(),
        "tilt_angles_rad": tilts.copy(),
        "tilt_angles_deg": np.rad2deg(tilts),
        "dynamic_pressure_pa": dynamic_pressure,
        "cm_alpha_per_rad": cm_alpha,
        "cl_beta_per_rad": cl_beta,
        "cn_beta_per_rad": cn_beta,
        "trim_feasible": trim_feasible,
        "trim_residual": trim_residual,
        "trim_motor_rpm": trim_motor_rpm,
        "trim_front_motor_rpm": trim_front_motor_rpm,
        "trim_rear_motor_rpm": trim_rear_motor_rpm,
        "trim_pitch_moment_residual": trim_pitch_moment_residual,
        "coefficient_valid": coefficient_valid,
        "assessment_mask": assessment_mask,
        "control_effectiveness_nm": control_effectiveness,
        "control_effectiveness_labels": effectiveness_labels or (),
    }


def assess_motor_electrical_point(
    motor_rpm: float,
    rotor_count: int,
    parameters: Mapping[str, float | int | None],
) -> dict[str, float | bool | tuple[str, ...]]:
    """Estimate steady motor and battery current from a quadratic propeller torque model.

    The estimate requires explicit motor, propeller, ESC, and battery data. It is
    an engineering screen, not a substitute for measured propulsion maps.
    """
    if rotor_count <= 0 or not np.isfinite(motor_rpm) or motor_rpm < 0.0:
        raise ValueError("rotor count must be positive and motor RPM must be finite and non-negative")
    return assess_rotor_electrical_point(np.full(rotor_count, motor_rpm), parameters)


def assess_rotor_electrical_point(
    motor_rpms: list[float] | np.ndarray,
    parameters: Mapping[str, float | int | None],
) -> dict[str, float | bool | tuple[str, ...]]:
    """Screen individual motor currents and their combined battery demand."""
    motor_rpms = np.asarray(motor_rpms, dtype=float)
    if (
        motor_rpms.ndim != 1
        or motor_rpms.size == 0
        or not np.all(np.isfinite(motor_rpms))
        or np.any(motor_rpms < 0.0)
    ):
        raise ValueError("motor RPMs must be a non-empty finite non-negative sequence")
    required = (
        "motor_kv_rpm_per_volt",
        "motor_resistance_ohm",
        "motor_no_load_current_a",
        "motor_continuous_current_a",
        "propeller_torque_coefficient_nm_s2_rad2",
        "battery_nominal_voltage_v",
        "battery_internal_resistance_ohm",
        "battery_capacity_ah",
        "battery_c_rating",
        "battery_continuous_current_a",
        "esc_efficiency",
        "esc_continuous_current_a",
    )
    missing = tuple(
        name for name in required
        if parameters.get(name) is None
    )
    if missing:
        return {
            "assessed": False,
            "feasible": False,
            "missing_parameters": missing,
            "motor_current_a": float("nan"),
            "battery_current_a": float("nan"),
            "battery_voltage_v": float("nan"),
        }

    values = {name: float(parameters[name]) for name in required}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("electrical parameters must be finite")
    if (
        values["motor_kv_rpm_per_volt"] <= 0.0
        or values["motor_resistance_ohm"] < 0.0
        or values["motor_no_load_current_a"] < 0.0
        or values["motor_continuous_current_a"] <= 0.0
        or values["propeller_torque_coefficient_nm_s2_rad2"] <= 0.0
        or values["battery_nominal_voltage_v"] <= 0.0
        or values["battery_internal_resistance_ohm"] < 0.0
        or values["battery_capacity_ah"] <= 0.0
        or values["battery_c_rating"] <= 0.0
        or values["battery_continuous_current_a"] <= 0.0
        or not 0.0 < values["esc_efficiency"] <= 1.0
        or values["esc_continuous_current_a"] <= 0.0
    ):
        raise ValueError("electrical limits and model parameters must be physically positive")

    kv = values["motor_kv_rpm_per_volt"]
    motor_torque_constant = 60.0 / (2.0 * np.pi * kv)
    angular_speed = motor_rpms * 2.0 * np.pi / 60.0
    propeller_torque = (
        values["propeller_torque_coefficient_nm_s2_rad2"] * angular_speed**2
    )
    motor_current = (
        propeller_torque / motor_torque_constant
        + values["motor_no_load_current_a"]
    )
    motor_voltage = (
        motor_rpms / kv
        + motor_current * values["motor_resistance_ohm"]
    )
    motor_power = motor_voltage * motor_current
    battery_power = float(np.sum(motor_power)) / values["esc_efficiency"]
    battery_resistance = values["battery_internal_resistance_ohm"]
    nominal_voltage = values["battery_nominal_voltage_v"]
    discriminant = nominal_voltage**2 - 4.0 * battery_resistance * battery_power
    if discriminant < 0.0:
        battery_current = float("inf")
        battery_voltage = 0.0
    elif battery_resistance == 0.0:
        battery_current = battery_power / nominal_voltage
        battery_voltage = nominal_voltage
    else:
        battery_current = (
            nominal_voltage - np.sqrt(discriminant)
        ) / (2.0 * battery_resistance)
        battery_voltage = nominal_voltage - battery_current * battery_resistance

    battery_current_limit = min(
        values["battery_capacity_ah"] * values["battery_c_rating"],
        values["battery_continuous_current_a"],
    )
    feasible = bool(
        np.all(motor_current <= values["motor_continuous_current_a"])
        and np.all(motor_current <= values["esc_continuous_current_a"])
        and battery_current <= battery_current_limit
        and np.all(motor_voltage <= battery_voltage)
    )
    return {
        "assessed": True,
        "feasible": feasible,
        "missing_parameters": (),
        "motor_current_a": float(np.max(motor_current)),
        "battery_current_a": float(battery_current),
        "battery_voltage_v": float(battery_voltage),
    }


def constrained_transition_corridor(
    dynamics: TiltrotorDynamics,
    *,
    max_thrust_per_rotor: float,
    rotor_tilts: list[bool],
    airspeeds_mps: list[float] | np.ndarray,
    tilt_angles_deg: list[float] | np.ndarray,
    max_tilt_rate_deg_s: float,
    transition_duration_s: float,
    gust_speed_margin_mps: float = 0.0,
    electrical_parameters: Mapping[str, float | int | None] | None = None,
    tail_limit_rad: float = 0.2,
    trim_residual_tolerance: float = 0.02,
    cl_max: float,
    cd_0: float,
    aspect_ratio: float,
    oswald_efficiency: float = 0.8,
) -> dict[str, object]:
    """Build a grid corridor filtered by trim, static signs, gust reserve, and actuator limits.

    Electrical and motor-slew constraints are fail-closed: cells are not fully
    qualified unless the required model inputs are present and those limits pass.
    The gust check is an airspeed reserve, not a transient gust-response proof.
    """
    speeds = np.asarray(airspeeds_mps, dtype=float)
    tilts = np.asarray(tilt_angles_deg, dtype=float)
    if (
        speeds.ndim != 1
        or speeds.size == 0
        or not np.all(np.isfinite(speeds))
        or np.any(speeds < 0.0)
        or (speeds.size > 1 and np.any(np.diff(speeds) <= 0.0))
    ):
        raise ValueError("airspeeds must be a non-empty, increasing finite non-negative sequence")
    if (
        tilts.ndim != 1
        or tilts.size < 2
        or not np.all(np.isfinite(tilts))
        or not np.allclose(tilts, np.linspace(0.0, 90.0, tilts.size))
    ):
        raise ValueError("tilt angles must be an increasing grid from 0 to 90 degrees")
    if (
        not np.isfinite(max_tilt_rate_deg_s)
        or max_tilt_rate_deg_s <= 0.0
        or not np.isfinite(transition_duration_s)
        or transition_duration_s <= 0.0
        or not np.isfinite(gust_speed_margin_mps)
        or gust_speed_margin_mps < 0.0
    ):
        raise ValueError("tilt rate and transition duration must be positive; gust margin cannot be negative")
    if dynamics.aero is None:
        raise ValueError("constrained corridor requires an aerodynamic model")
    if len(rotor_tilts) != 4:
        raise ValueError("constrained corridor requires four rotors")
    if (
        not np.isfinite(max_thrust_per_rotor)
        or max_thrust_per_rotor < 0.0
        or not np.isfinite(dynamics.rotor.k_thrust)
        or dynamics.rotor.k_thrust <= 0.0
        or not np.isfinite(dynamics.rotor.max_omega)
        or dynamics.rotor.max_omega <= 0.0
    ):
        raise ValueError("rotor thrust limit and model parameters must be finite and valid")

    envelope = transition_speed_envelope(
        mass=dynamics.mass,
        wing_area=float(dynamics.aero.wing_area),
        max_thrust_per_rotor=max_thrust_per_rotor,
        rotor_tilts=rotor_tilts,
        rho=float(dynamics.aero.rho),
        cl_max=cl_max,
        cd_0=cd_0,
        aspect_ratio=aspect_ratio,
        oswald_efficiency=oswald_efficiency,
        samples=tilts.size,
    )
    if not np.allclose(envelope["tilt_deg"], tilts):
        raise ValueError("tilt grid must match the analytic envelope grid")

    trim_omega_limit = None
    if max_thrust_per_rotor > 0.0:
        trim_omega_limit = min(
            float(dynamics.rotor.max_omega),
            float(np.sqrt(max_thrust_per_rotor / dynamics.rotor.k_thrust)),
        )
    stability = transition_stability_map(
        dynamics,
        airspeeds_mps=speeds,
        tilt_angles_rad=np.deg2rad(tilts),
        tail_limit_rad=tail_limit_rad,
        trim_residual_tolerance=trim_residual_tolerance,
        max_rotor_omega=trim_omega_limit,
    )
    speed_grid = speeds[:, np.newaxis]
    minimum_speed = envelope["minimum_speed_mps"][np.newaxis, :]
    maximum_speed = envelope["maximum_speed_mps"][np.newaxis, :]
    analytic_mask = (
        envelope["feasible"][np.newaxis, :]
        & (speed_grid >= minimum_speed)
        & (speed_grid <= maximum_speed)
    )
    gust_mask = (
        analytic_mask
        & (speed_grid >= minimum_speed + gust_speed_margin_mps)
        & (speed_grid <= maximum_speed - gust_speed_margin_mps)
    )
    trim_mask = np.asarray(stability["trim_feasible"], dtype=bool)
    pair_max_rpm = np.maximum(
        np.asarray(stability["trim_front_motor_rpm"], dtype=float),
        np.asarray(stability["trim_rear_motor_rpm"], dtype=float),
    )
    max_rpm_from_thrust = np.sqrt(max_thrust_per_rotor / dynamics.rotor.k_thrust) * 60.0 / (2.0 * np.pi)
    rotor_thrust_mask = np.isfinite(pair_max_rpm) & (pair_max_rpm <= max_rpm_from_thrust + 1e-6)
    pitch_residual = np.asarray(
        stability["trim_pitch_moment_residual"],
        dtype=float,
    )
    pitch_moment_mask = np.isfinite(pitch_residual) & (
        np.abs(pitch_residual) <= trim_residual_tolerance
    )
    assessment_mask = np.asarray(stability["assessment_mask"], dtype=bool)
    cm_alpha = np.asarray(stability["cm_alpha_per_rad"], dtype=float)
    cl_beta = np.asarray(stability["cl_beta_per_rad"], dtype=float)
    cn_beta = np.asarray(stability["cn_beta_per_rad"], dtype=float)
    static_stability_mask = (
        assessment_mask
        & (cm_alpha < 0.0)
        & (cl_beta < 0.0)
        & (cn_beta > 0.0)
    )
    tilt_reachable = tilts <= max_tilt_rate_deg_s * transition_duration_s
    actuator_rate_mask = np.broadcast_to(
        tilt_reachable[np.newaxis, :],
        analytic_mask.shape,
    ).copy()
    known_constraints_mask = (
        gust_mask
        & trim_mask
        & rotor_thrust_mask
        & pitch_moment_mask
        & static_stability_mask
        & actuator_rate_mask
    )

    motor_slew_rate = None if electrical_parameters is None else (
        electrical_parameters.get("motor_slew_rate_rpm_s")
    )
    motor_slew_assessed = np.zeros(analytic_mask.shape, dtype=bool)
    motor_slew_feasible = np.zeros(analytic_mask.shape, dtype=bool)
    if motor_slew_rate is not None:
        slew_limit = float(motor_slew_rate)
        if not np.isfinite(slew_limit) or slew_limit <= 0.0:
            raise ValueError("motor slew rate must be positive and finite")
        trim_front_rpm = np.asarray(stability["trim_front_motor_rpm"], dtype=float)
        trim_rear_rpm = np.asarray(stability["trim_rear_motor_rpm"], dtype=float)
        if np.isclose(speeds[0], 0.0):
            initial_front_rpm = trim_front_rpm[0, 0]
            initial_rear_rpm = trim_rear_rpm[0, 0]
            initial_trim_feasible = bool(trim_mask[0, 0])
        else:
            hover_trim = solve_longitudinal_trim(
                dynamics,
                0.0,
                0.0,
                tail_limit_rad=tail_limit_rad,
                residual_tolerance=trim_residual_tolerance,
                max_rotor_omega=trim_omega_limit,
            )
            initial_front_rpm = float(hover_trim["front_motor_rpm"])
            initial_rear_rpm = float(hover_trim["rear_motor_rpm"])
            initial_trim_feasible = bool(hover_trim["feasible"])
        if initial_trim_feasible and np.isfinite(initial_front_rpm) and np.isfinite(initial_rear_rpm):
            motor_slew_assessed = (
                analytic_mask
                & trim_mask
                & pitch_moment_mask
                & np.isfinite(trim_front_rpm)
                & np.isfinite(trim_rear_rpm)
            )
            required_average_slew = np.maximum(
                np.abs(trim_front_rpm - initial_front_rpm),
                np.abs(trim_rear_rpm - initial_rear_rpm),
            ) / transition_duration_s
            motor_slew_feasible = motor_slew_assessed & (
                required_average_slew <= slew_limit
            )

    electrical_assessed = np.zeros(analytic_mask.shape, dtype=bool)
    electrical_feasible = np.zeros(analytic_mask.shape, dtype=bool)
    electrical_missing: tuple[str, ...] = ()
    if electrical_parameters is not None:
        electrical_missing = tuple(
            assess_motor_electrical_point(
                0.0,
                len(rotor_tilts),
                electrical_parameters,
            )["missing_parameters"]
        )
        trim_front_rpm = np.asarray(stability["trim_front_motor_rpm"], dtype=float)
        trim_rear_rpm = np.asarray(stability["trim_rear_motor_rpm"], dtype=float)
        for index in np.ndindex(analytic_mask.shape):
            if (
                not analytic_mask[index]
                or not trim_mask[index]
                or not pitch_moment_mask[index]
                or not np.isfinite(trim_front_rpm[index])
                or not np.isfinite(trim_rear_rpm[index])
            ):
                continue
            point = assess_rotor_electrical_point(
                np.repeat([trim_front_rpm[index], trim_rear_rpm[index]], 2),
                electrical_parameters,
            )
            electrical_assessed[index] = bool(point["assessed"])
            electrical_feasible[index] = bool(point["feasible"])

    qualified_mask = (
        known_constraints_mask
        & motor_slew_assessed
        & motor_slew_feasible
        & electrical_assessed
        & electrical_feasible
    )
    return {
        "airspeeds_mps": speeds.copy(),
        "tilt_angles_deg": tilts.copy(),
        "analytic_envelope": envelope,
        "stability_map": stability,
        "analytic_mask": analytic_mask,
        "gust_mask": gust_mask,
        "trim_mask": trim_mask,
        "rotor_thrust_mask": rotor_thrust_mask,
        "pitch_moment_mask": pitch_moment_mask,
        "static_stability_mask": static_stability_mask,
        "actuator_rate_mask": actuator_rate_mask,
        "known_constraints_mask": known_constraints_mask,
        "motor_slew_assessed_mask": motor_slew_assessed,
        "motor_slew_feasible_mask": motor_slew_feasible,
        "electrical_assessed_mask": electrical_assessed,
        "electrical_feasible_mask": electrical_feasible,
        "electrical_missing_parameters": electrical_missing,
        "qualified_mask": qualified_mask,
    }
