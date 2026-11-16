"""Transition-envelope, trim, and local control-effectiveness analysis."""

from __future__ import annotations

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
) -> dict[str, float | bool | np.ndarray | str]:
    """Solve steady, level longitudinal force and pitch-moment balance.

    The solver varies body angle of attack, equal rotor speed, and symmetric V-tail
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
    chord = max(float(aero.chord), 0.01)
    hover_omega = np.sqrt(dynamics.mass * GRAVITY / max(4.0 * rotor.k_thrust, 1e-12))
    initial = np.array([0.0, min(hover_omega, rotor.max_omega), 0.0])
    lower = np.array([-aero.stall_angle, 0.0, -tail_limit_rad])
    upper = np.array([aero.stall_angle, rotor.max_omega, tail_limit_rad])
    wind_ned = np.zeros(3)

    def evaluate(candidate: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        alpha, omega, tail_deflection = candidate
        state = AircraftState(
            pos=np.array([0.0, 0.0, -100.0]),
            vel=np.array([airspeed_mps, 0.0, 0.0]),
            att=np.array([0.0, alpha, 0.0]),
            tilt=float(tilt_rad),
        )
        dynamics.step(
            state,
            wind_ned,
            float(omega),
            float(omega),
            float(omega),
            float(omega),
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
        "motor_rpm": float(result.x[1] * 60.0 / (2.0 * np.pi)),
        "tail_deflection_deg": float(np.rad2deg(result.x[2])),
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
    tail_limit_rad: float = 0.2,
    rotor_step_rpm: float = 50.0,
    tail_step_deg: float = 5.0,
) -> tuple[list[str], np.ndarray]:
    """Estimate local body-moment effectiveness for rotor and V-tail control inputs."""
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
    base_omega = np.full(4, collective_omega, dtype=float)
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
            plus = np.clip(base_omega + omega_step * pattern, 0.0, dynamics.rotor.max_omega)
            minus = np.clip(base_omega - omega_step * pattern, 0.0, dynamics.rotor.max_omega)
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
        rotor_omega: float,
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
            rotor_omega,
            rotor_omega,
            rotor_omega,
            rotor_omega,
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
            )
            trim_feasible[speed_index, tilt_index] = bool(trim["feasible"])
            trim_residual[speed_index, tilt_index] = float(trim["max_normalized_residual"])

            pitch_rad = float(np.deg2rad(trim["alpha_deg"]))
            tail_deflection = float(np.deg2rad(trim["tail_deflection_deg"]))
            rotor_omega = float(trim["motor_rpm"]) * 2.0 * np.pi / 60.0
            labels, effectiveness = control_effectiveness_map(
                dynamics,
                airspeed_mps=float(airspeed),
                tilt_rad=float(tilt),
                alpha_rad=pitch_rad,
                collective_omega=rotor_omega,
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
                rotor_omega,
                tail_deflection,
            )
            moment_alpha_minus = evaluate_moment(
                float(airspeed),
                alpha_trim - perturbation_rad,
                beta_trim,
                float(tilt),
                rotor_omega,
                tail_deflection,
            )
            moment_beta_plus = evaluate_moment(
                float(airspeed),
                alpha_trim,
                beta_trim + perturbation_rad,
                float(tilt),
                rotor_omega,
                tail_deflection,
            )
            moment_beta_minus = evaluate_moment(
                float(airspeed),
                alpha_trim,
                beta_trim - perturbation_rad,
                float(tilt),
                rotor_omega,
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
        "coefficient_valid": coefficient_valid,
        "assessment_mask": assessment_mask,
        "control_effectiveness_nm": control_effectiveness,
        "control_effectiveness_labels": effectiveness_labels or (),
    }
