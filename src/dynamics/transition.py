"""Transition-envelope, trim, and local control-effectiveness analysis."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from src.dynamics.state import AircraftState
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
