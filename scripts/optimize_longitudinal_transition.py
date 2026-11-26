"""Optimize a longitudinal tiltrotor transition and plot its model outputs."""

import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.aero.wing_polar import wing_lift_drag_coefficients
from src.dynamics.transition import assess_rotor_electrical_point
from src.dynamics.nacelle import tilt_deg_from_deflection_rad


AIRCRAFT = load_aircraft_parameters()
if len(AIRCRAFT.motor_positions) != 4 or sum(position[3] for position in AIRCRAFT.motor_positions) != 2:
    raise ValueError("the reduced pair-thrust optimizer requires two tilting and two fixed rotors")
MASS_KG = AIRCRAFT.mass
WING_AREA_M2 = AIRCRAFT.wing_area
RHO_KG_M3 = AIRCRAFT.air_density
CL0 = AIRCRAFT.wing_lift_coefficient_zero
CL_ALPHA = AIRCRAFT.wing_lift_slope
MAX_ALPHA_RAD = np.deg2rad(AIRCRAFT.wing_stall_angle_deg)
VTAIL_LIMIT_RAD = AIRCRAFT.vtail_max_deflection_rad
CD0 = AIRCRAFT.wing_drag_coeff
ASPECT_RATIO = AIRCRAFT.wing_aspect_ratio
OSWALD_EFFICIENCY = AIRCRAFT.oswald_efficiency
MAX_PAIR_THRUST_N = 2 * AIRCRAFT.max_thrust_per_rotor
MAX_REAR_PAIR_THRUST_N = MAX_PAIR_THRUST_N * (
    1.0 - AIRCRAFT.rear_thrust_reserve_fraction
)
MAX_TILT_RATE_RAD_S = AIRCRAFT.max_tilt_rate
INITIAL_ALTITUDE_M = 100.0
MIN_ALTITUDE_M = 90.0
TARGET_SPEED_MPS = 12.0
DURATION_S = 15.0
SEGMENTS = 60
DT_S = DURATION_S / SEGMENTS
GRAVITY_MPS2 = AIRCRAFT.gravity
SCHEDULE_MIN_FORWARD_TILT_DEG = 15.0
SCHEDULE_MIN_FORWARD_TILT_RAD = np.deg2rad(SCHEDULE_MIN_FORWARD_TILT_DEG)
PRE_ACCELERATION_SEGMENTS = int(
    np.ceil(
        SCHEDULE_MIN_FORWARD_TILT_RAD
        / (MAX_TILT_RATE_RAD_S * DT_S)
    )
) + 1


def _validate_motor_speed_limit_fraction(motor_speed_limit_fraction):
    if (
        not np.isfinite(motor_speed_limit_fraction)
        or not 0.0 < motor_speed_limit_fraction <= 1.0
    ):
        raise ValueError("motor speed limit fraction must be in (0, 1]")
    return float(motor_speed_limit_fraction)


def build_transition_schedule_reference(time_s):
    """Reference a one-second nacelle pre-tilt before forward acceleration."""
    time_s = np.asarray(time_s, dtype=float)
    pre_acceleration_time_s = PRE_ACCELERATION_SEGMENTS * DT_S
    speed_fraction = np.clip(
        (time_s - pre_acceleration_time_s)
        / (DURATION_S - pre_acceleration_time_s),
        0.0,
        1.0,
    )
    speed_reference = TARGET_SPEED_MPS * speed_fraction
    tilt_reference = np.interp(
        time_s,
        [0.0, DT_S, pre_acceleration_time_s, DURATION_S],
        [0.0, 0.0, SCHEDULE_MIN_FORWARD_TILT_RAD, np.pi / 2.0],
    )
    return speed_reference, tilt_reference


def add_rotor_speed_outputs(data, motor_speed_limit_fraction=1.0):
    """Convert pair thrust to per-rotor RPM and enforce active RPM limits."""
    motor_speed_limit_fraction = _validate_motor_speed_limit_fraction(
        motor_speed_limit_fraction
    )
    front_pair_thrust = np.asarray(data["tilting_pair_thrust_n"], dtype=float)
    rear_pair_thrust = np.asarray(data["fixed_pair_thrust_n"], dtype=float)
    if (
        front_pair_thrust.shape != rear_pair_thrust.shape
        or not np.isfinite(front_pair_thrust).all()
        or not np.isfinite(rear_pair_thrust).all()
        or np.any(front_pair_thrust < 0.0)
        or np.any(rear_pair_thrust < 0.0)
    ):
        raise ValueError("pair thrust histories must be same-shaped, finite, and non-negative")

    rpm_per_rad_s = 60.0 / (2.0 * np.pi)
    front_rotor_rpm = np.sqrt(
        front_pair_thrust / (2.0 * AIRCRAFT.k_thrust)
    ) * rpm_per_rad_s
    rear_rotor_rpm = np.sqrt(
        rear_pair_thrust / (2.0 * AIRCRAFT.k_thrust)
    ) * rpm_per_rad_s
    front_rpm_limit = (
        AIRCRAFT.estimated_motor_rpm * motor_speed_limit_fraction
    )
    rear_rpm_limit = front_rpm_limit * np.sqrt(
        1.0 - AIRCRAFT.rear_thrust_reserve_fraction
    )
    front_rpm_margin = front_rpm_limit - front_rotor_rpm
    rear_rpm_margin = rear_rpm_limit - rear_rotor_rpm
    tolerance_rpm = 1e-3
    if (
        np.any(front_rpm_margin < -tolerance_rpm)
        or np.any(rear_rpm_margin < -tolerance_rpm)
    ):
        raise RuntimeError(
            "transition solution exceeds the configured front or rear rotor RPM limit"
        )

    return {
        **data,
        "front_rotor_rpm": front_rotor_rpm,
        "rear_rotor_rpm": rear_rotor_rpm,
        "front_rpm_limit": np.full_like(front_rotor_rpm, front_rpm_limit),
        "rear_rpm_limit": np.full_like(rear_rotor_rpm, rear_rpm_limit),
        "front_rpm_margin": front_rpm_margin,
        "rear_rpm_margin": rear_rpm_margin,
    }


def longitudinal_dynamics(
    vx,
    vz,
    tilt,
    tilting_pair_thrust,
    fixed_pair_thrust,
    alpha_rad=0.0,
    vtail_deflection_rad=0.0,
):
    """Return still-air forces and pitch moment for the reduced-order model."""
    speed_squared = vx**2 + vz**2
    airspeed = ca.sqrt(speed_squared)
    dynamic_pressure = 0.5 * RHO_KG_M3 * speed_squared
    lift_coefficient, drag_coefficient = wing_lift_drag_coefficients(
        alpha_rad,
        cl_zero=CL0,
        lift_slope_per_rad=CL_ALPHA,
        stall_angle_rad=MAX_ALPHA_RAD,
        cd_zero=CD0,
        aspect_ratio=ASPECT_RATIO,
        oswald_efficiency=OSWALD_EFFICIENCY,
    )
    lift = dynamic_pressure * WING_AREA_M2 * lift_coefficient
    drag = dynamic_pressure * WING_AREA_M2 * drag_coefficient
    tail_alpha = (
        alpha_rad
        + vtail_deflection_rad
        * np.cos(np.deg2rad(AIRCRAFT.vtail_dihedral_deg))
    )
    tail_lift_coefficient = ca.fmin(
        ca.fmax(
            AIRCRAFT.vtail_lift_slope * tail_alpha,
            -AIRCRAFT.vtail_max_lift_coefficient,
        ),
        AIRCRAFT.vtail_max_lift_coefficient,
    )
    tail_drag_coefficient = AIRCRAFT.vtail_drag_coeff + (
        tail_lift_coefficient**2
        / (
            np.pi
            * AIRCRAFT.vtail_oswald_efficiency
            * AIRCRAFT.vtail_aspect_ratio
        )
    )
    tail_active = ca.if_else(airspeed >= 0.5, 1.0, 0.0)
    tail_lift = (
        tail_active * dynamic_pressure * AIRCRAFT.vtail_area
        * tail_lift_coefficient
    )
    tail_drag = (
        tail_active * dynamic_pressure * AIRCRAFT.vtail_area
        * tail_drag_coefficient
    )

    flight_path_angle = ca.atan2(vz, ca.sqrt(vx**2 + 1e-8))
    pitch = flight_path_angle + alpha_rad

    # Resolve wing lift and drag into body axes, then rotate into NED axes.
    aero_force_x_body = -drag * ca.cos(alpha_rad) + lift * ca.sin(alpha_rad)
    aero_force_z_body = -drag * ca.sin(alpha_rad) - lift * ca.cos(alpha_rad)
    force_x_body = (
        tilting_pair_thrust * ca.sin(tilt) + aero_force_x_body - tail_drag
    )
    force_z_body = (
        -tilting_pair_thrust * ca.cos(tilt)
        - fixed_pair_thrust
        + aero_force_z_body
        - tail_lift * np.cos(np.deg2rad(AIRCRAFT.vtail_dihedral_deg))
    )

    force_x_ned = (
        ca.cos(pitch) * force_x_body + ca.sin(pitch) * force_z_body
    )
    force_z_ned = (
        -ca.sin(pitch) * force_x_body + ca.cos(pitch) * force_z_body
    )
    acceleration_x = force_x_ned / MASS_KG
    acceleration_up = -force_z_ned / MASS_KG - GRAVITY_MPS2
    tail_lift_vertical = (
        tail_lift
        * np.cos(np.deg2rad(AIRCRAFT.vtail_dihedral_deg))
        * ca.cos(pitch)
    )
    tail_drag_vertical = -tail_drag * ca.sin(pitch)
    wing_lift_vertical = lift * ca.cos(flight_path_angle)
    wing_drag_vertical = -drag * ca.sin(flight_path_angle)
    front_vertical_thrust = tilting_pair_thrust * ca.cos(tilt - pitch)
    front_horizontal_thrust = tilting_pair_thrust * ca.sin(tilt - pitch)
    fixed_vertical_thrust = fixed_pair_thrust * ca.cos(pitch)
    fixed_horizontal_thrust = -fixed_pair_thrust * ca.sin(pitch)
    vertical_force_up = -force_z_ned
    vertical_support_margin = vertical_force_up - MASS_KG * GRAVITY_MPS2
    wing_pitch_moment = (
        -AIRCRAFT.wing_aerodynamic_center_x_m * aero_force_z_body
        + dynamic_pressure
        * WING_AREA_M2
        * AIRCRAFT.wing_mean_chord_m
        * (AIRCRAFT.wing_pitch_moment_zero
           + AIRCRAFT.wing_pitch_moment_slope * alpha_rad)
    )
    vtail_pitch_moment = (
        -AIRCRAFT.vtail_arm_length
        * tail_lift
        * np.cos(np.deg2rad(AIRCRAFT.vtail_dihedral_deg))
    )
    rotor_pitch_moment = 0.0
    pivot_x, pivot_z = AIRCRAFT.tilt_pivot_xz_m
    for x, _y, _z, is_tilting in AIRCRAFT.motor_positions:
        if is_tilting:
            relative_x = x - pivot_x
            # The hub offset and thrust axis rotate together, so their
            # cross product about the hinge has the same pitch arm at every tilt.
            moment_arm = relative_x + pivot_x * ca.cos(tilt) + pivot_z * ca.sin(tilt)
            rotor_pitch_moment += 0.5 * tilting_pair_thrust * moment_arm
        else:
            rotor_pitch_moment += 0.5 * fixed_pair_thrust * x
    pitch_moment = wing_pitch_moment + vtail_pitch_moment + rotor_pitch_moment
    return {
        "airspeed": ca.sqrt(speed_squared),
        "alpha_rad": alpha_rad,
        "flight_path_angle": flight_path_angle,
        "lift_coefficient": lift_coefficient,
        "dynamic_pressure": dynamic_pressure,
        "lift": lift,
        "drag": drag,
        "tail_lift_n": tail_lift,
        "tail_drag_n": tail_drag,
        "tail_vertical_force_n": tail_lift_vertical + tail_drag_vertical,
        "force_x_body": force_x_body,
        "force_z_body": force_z_body,
        "force_x_ned": force_x_ned,
        "force_z_ned": force_z_ned,
        "acceleration_x": acceleration_x,
        "acceleration_up": acceleration_up,
        "pitch": pitch,
        "wing_lift_vertical_n": wing_lift_vertical,
        "wing_drag_vertical_n": wing_drag_vertical,
        "front_vertical_thrust_n": front_vertical_thrust,
        "front_horizontal_thrust_n": front_horizontal_thrust,
        "fixed_vertical_thrust_n": fixed_vertical_thrust,
        "fixed_horizontal_thrust_n": fixed_horizontal_thrust,
        "wing_pitch_moment_n_m": wing_pitch_moment,
        "vtail_pitch_moment_n_m": vtail_pitch_moment,
        "rotor_pitch_moment_n_m": rotor_pitch_moment,
        "pitch_moment_n_m": pitch_moment,
        "required_vertical_force_n": MASS_KG * (GRAVITY_MPS2 + acceleration_up),
        "vertical_support_margin_n": vertical_support_margin,
        "vertical_force_balance_residual_n": (
            vertical_force_up
            - MASS_KG * (GRAVITY_MPS2 + acceleration_up)
        ),
    }


def solve_transition(motor_speed_limit_fraction=1.0):
    """Solve using an assumed fraction of the configured estimated RPM limit."""
    motor_speed_limit_fraction = _validate_motor_speed_limit_fraction(
        motor_speed_limit_fraction
    )
    max_rotor_omega = AIRCRAFT.max_omega * motor_speed_limit_fraction
    max_rotor_thrust = AIRCRAFT.k_thrust * max_rotor_omega**2
    max_front_pair_thrust = 2.0 * max_rotor_thrust
    max_rear_pair_thrust = max_front_pair_thrust * (
        1.0 - AIRCRAFT.rear_thrust_reserve_fraction
    )

    opti = ca.Opti()
    vx = opti.variable(SEGMENTS + 1)
    vz = opti.variable(SEGMENTS + 1)
    altitude = opti.variable(SEGMENTS + 1)
    tilt = opti.variable(SEGMENTS + 1)
    alpha = opti.variable(SEGMENTS + 1)
    vtail_deflection = opti.variable(SEGMENTS + 1)
    tilt_rate = opti.variable(SEGMENTS)
    tilting_thrust = opti.variable(SEGMENTS + 1)
    fixed_thrust = opti.variable(SEGMENTS + 1)
    max_altitude_loss = opti.variable()

    opti.subject_to(vx[0] == 0.0)
    opti.subject_to(vz[0] == 0.0)
    opti.subject_to(altitude[0] == INITIAL_ALTITUDE_M)
    opti.subject_to(tilt[0] == 0.0)
    opti.subject_to(alpha[0] == 0.0)
    opti.subject_to(vtail_deflection[0] == 0.0)
    opti.subject_to(tilt_rate[0] == 0.0)
    opti.subject_to(tilt_rate[-1] == 0.0)
    opti.subject_to(vx[-1] == TARGET_SPEED_MPS)
    opti.subject_to(vz[-1] == 0.0)
    opti.subject_to(altitude[-1] == INITIAL_ALTITUDE_M)
    opti.subject_to(tilt[-1] == np.pi / 2.0)

    opti.subject_to(opti.bounded(0.0, vx, TARGET_SPEED_MPS))
    opti.subject_to(opti.bounded(-4.0, vz, 4.0))
    opti.subject_to(opti.bounded(MIN_ALTITUDE_M, altitude, 110.0))
    opti.subject_to(opti.bounded(0.0, tilt, np.pi / 2.0))
    for node in range(1, PRE_ACCELERATION_SEGMENTS + 1):
        opti.subject_to(vx[node] == 0.0)
    for node in range(PRE_ACCELERATION_SEGMENTS, SEGMENTS + 1):
        opti.subject_to(tilt[node] >= SCHEDULE_MIN_FORWARD_TILT_RAD)
    opti.subject_to(opti.bounded(0.0, alpha, MAX_ALPHA_RAD))
    opti.subject_to(opti.bounded(
        -VTAIL_LIMIT_RAD,
        vtail_deflection,
        VTAIL_LIMIT_RAD,
    ))
    opti.subject_to(opti.bounded(
        -MAX_TILT_RATE_RAD_S,
        tilt_rate,
        MAX_TILT_RATE_RAD_S,
    ))
    opti.subject_to(opti.bounded(0.0, tilting_thrust, max_front_pair_thrust))
    opti.subject_to(opti.bounded(0.0, fixed_thrust, max_rear_pair_thrust))
    opti.subject_to(max_altitude_loss >= 0.0)

    forward_accelerations = []
    upward_accelerations = []
    for k in range(SEGMENTS):
        point = longitudinal_dynamics(
            vx[k],
            vz[k],
            tilt[k],
            tilting_thrust[k],
            fixed_thrust[k],
            alpha[k],
            vtail_deflection[k],
        )
        ax = point["acceleration_x"]
        az_up = point["acceleration_up"]
        forward_accelerations.append(ax)
        upward_accelerations.append(az_up)
        opti.subject_to(opti.bounded(-3.0, ax, 3.0))
        opti.subject_to(opti.bounded(-2.0, az_up, 2.0))
        opti.subject_to(point["pitch_moment_n_m"] == 0.0)
        opti.subject_to(vx[k + 1] == vx[k] + ax * DT_S)
        opti.subject_to(vx[k + 1] >= vx[k])
        opti.subject_to(vz[k + 1] == vz[k] + az_up * DT_S)
        opti.subject_to(
            altitude[k + 1]
            == altitude[k] + 0.5 * (vz[k] + vz[k + 1]) * DT_S
        )
        opti.subject_to(tilt[k + 1] == tilt[k] + tilt_rate[k] * DT_S)
        opti.subject_to(
            max_altitude_loss >= INITIAL_ALTITUDE_M - altitude[k]
        )

    terminal = longitudinal_dynamics(
        vx[-1],
        vz[-1],
        tilt[-1],
        tilting_thrust[-1],
        fixed_thrust[-1],
        alpha[-1],
        vtail_deflection[-1],
    )
    opti.subject_to(terminal["acceleration_x"] == 0.0)
    opti.subject_to(terminal["acceleration_up"] == 0.0)
    opti.subject_to(terminal["pitch_moment_n_m"] == 0.0)

    normalized_thrust = (
        ca.sumsqr(tilting_thrust / MAX_PAIR_THRUST_N)
        + ca.sumsqr(fixed_thrust / MAX_PAIR_THRUST_N)
    ) * DT_S
    altitude_tracking = ca.sumsqr(
        (altitude - INITIAL_ALTITUDE_M) / 10.0
    ) * DT_S
    time = np.linspace(0.0, DURATION_S, SEGMENTS + 1)
    speed_reference, tilt_reference = build_transition_schedule_reference(time)
    speed_tracking = ca.sumsqr((vx - speed_reference) / TARGET_SPEED_MPS) * DT_S
    tilt_rate_cost = ca.sumsqr(tilt_rate / MAX_TILT_RATE_RAD_S) * DT_S
    tilt_rate_smoothness = ca.sumsqr(
        ca.diff(tilt_rate) / MAX_TILT_RATE_RAD_S
    )
    alpha_smoothness = ca.sumsqr(ca.diff(alpha) / MAX_ALPHA_RAD)
    vtail_deflection_cost = ca.sumsqr(
        vtail_deflection / VTAIL_LIMIT_RAD
    ) * DT_S
    vtail_smoothness = ca.sumsqr(
        ca.diff(vtail_deflection) / VTAIL_LIMIT_RAD
    )
    thrust_smoothness = (
        ca.sumsqr(ca.diff(tilting_thrust) / MAX_PAIR_THRUST_N)
        + ca.sumsqr(ca.diff(fixed_thrust) / MAX_PAIR_THRUST_N)
    )
    opti.minimize(
        10.0 * max_altitude_loss
        + 0.1 * normalized_thrust
        + 0.5 * altitude_tracking
        + 1.0 * speed_tracking
        + 0.1 * tilt_rate_cost
        + 0.1 * tilt_rate_smoothness
        + 0.1 * alpha_smoothness
        + 0.1 * vtail_deflection_cost
        + 0.1 * vtail_smoothness
        + 0.2 * thrust_smoothness
    )

    opti.set_initial(vx, speed_reference)
    opti.set_initial(vz, np.zeros(SEGMENTS + 1))
    opti.set_initial(altitude, np.full(SEGMENTS + 1, INITIAL_ALTITUDE_M))
    opti.set_initial(tilt, tilt_reference)
    opti.set_initial(alpha, np.zeros(SEGMENTS + 1))
    opti.set_initial(vtail_deflection, np.zeros(SEGMENTS + 1))
    opti.set_initial(tilt_rate, np.diff(tilt_reference) / DT_S)
    hover_pair_thrust = min(
        MASS_KG * GRAVITY_MPS2 / 2.0,
        0.9 * max_front_pair_thrust,
        0.9 * max_rear_pair_thrust,
    )
    opti.set_initial(
        tilting_thrust,
        np.full(SEGMENTS + 1, hover_pair_thrust),
    )
    opti.set_initial(
        fixed_thrust,
        np.full(SEGMENTS + 1, hover_pair_thrust),
    )
    opti.set_initial(max_altitude_loss, 1.0)

    opti.solver(
        "ipopt",
        {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 1500,
            "print_time": False,
        },
    )
    try:
        solution = opti.solve()
    except RuntimeError as error:
        if opti.stats().get("return_status") == "Infeasible_Problem_Detected":
            raise RuntimeError(
                "No feasible transition satisfies per-node pitch-moment balance "
                "with the configured rotor and V-tail limits."
            ) from error
        raise
    if not solution.stats().get("success", False):
        raise RuntimeError("The longitudinal corridor optimization did not converge")

    output = {
        "time_s": time,
        "vx_mps": np.asarray(solution.value(vx), dtype=float).ravel(),
        "vz_mps": np.asarray(solution.value(vz), dtype=float).ravel(),
        "altitude_m": np.asarray(solution.value(altitude), dtype=float).ravel(),
        "tilt_rad": np.asarray(solution.value(tilt), dtype=float).ravel(),
        "alpha_rad": np.asarray(solution.value(alpha), dtype=float).ravel(),
        "vtail_deflection_rad": np.asarray(
            solution.value(vtail_deflection), dtype=float
        ).ravel(),
        "tilt_rate_rad_s": np.append(
            np.asarray(solution.value(tilt_rate), dtype=float).ravel(),
            0.0,
        ),
        "tilting_pair_thrust_n": np.asarray(
            solution.value(tilting_thrust), dtype=float
        ).ravel(),
        "fixed_pair_thrust_n": np.asarray(
            solution.value(fixed_thrust), dtype=float
        ).ravel(),
        "max_altitude_loss_m": float(solution.value(max_altitude_loss)),
    }
    return add_rotor_speed_outputs(output, motor_speed_limit_fraction)


def add_model_outputs(data):
    model_keys = (
        "airspeed",
        "flight_path_angle",
        "lift_coefficient",
        "lift",
        "drag",
        "force_x_body",
        "force_z_body",
        "force_x_ned",
        "force_z_ned",
        "acceleration_x",
        "acceleration_up",
        "pitch",
        "wing_lift_vertical_n",
        "wing_drag_vertical_n",
        "front_vertical_thrust_n",
        "front_horizontal_thrust_n",
        "fixed_vertical_thrust_n",
        "fixed_horizontal_thrust_n",
        "required_vertical_force_n",
        "vertical_support_margin_n",
        "vertical_force_balance_residual_n",
        "tail_lift_n",
        "tail_drag_n",
        "tail_vertical_force_n",
        "wing_pitch_moment_n_m",
        "vtail_pitch_moment_n_m",
        "rotor_pitch_moment_n_m",
        "pitch_moment_n_m",
    )
    outputs = {key: np.empty_like(data["time_s"]) for key in model_keys}
    for index in range(data["time_s"].size):
        point = longitudinal_dynamics(
            data["vx_mps"][index],
            data["vz_mps"][index],
            data["tilt_rad"][index],
            data["tilting_pair_thrust_n"][index],
            data["fixed_pair_thrust_n"][index],
            data["alpha_rad"][index],
            data["vtail_deflection_rad"][index],
        )
        for key in model_keys:
            outputs[key][index] = float(point[key])
    electrical_rows = [
        assess_rotor_electrical_point(
            np.repeat(
                [
                    data["front_rotor_rpm"][index],
                    data["rear_rotor_rpm"][index],
                ],
                2,
            ),
            AIRCRAFT.electrical_parameters,
        )
        for index in range(data["time_s"].size)
    ]
    electrical_outputs = {
        "electrical_assessed": np.array(
            [row["assessed"] for row in electrical_rows], dtype=bool
        ),
        "electrical_feasible": np.array(
            [row["feasible"] for row in electrical_rows], dtype=bool
        ),
        "electrical_motor_current_a": np.array(
            [row["motor_current_a"] for row in electrical_rows], dtype=float
        ),
        "electrical_battery_current_a": np.array(
            [row["battery_current_a"] for row in electrical_rows], dtype=float
        ),
        "electrical_battery_voltage_v": np.array(
            [row["battery_voltage_v"] for row in electrical_rows], dtype=float
        ),
        "electrical_missing_parameters": np.array([
            ";".join(row["missing_parameters"])
            for row in electrical_rows
        ], dtype=str),
    }
    return {**data, **outputs, **electrical_outputs}


def plot_results(data):
    os.makedirs("figures", exist_ok=True)
    time = data["time_s"]
    tilt_deg = tilt_deg_from_deflection_rad(data["tilt_rad"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(time, tilt_deg, label="Optimized")
    _, schedule_tilt_reference = build_transition_schedule_reference(time)
    axes[0, 0].plot(
        time,
        tilt_deg_from_deflection_rad(schedule_tilt_reference),
        "--",
        label="Pre-tilt schedule reference",
    )
    cosine = 45.0 * (1.0 - np.cos(np.pi * time / DURATION_S))
    axes[0, 0].plot(time, cosine, ":", label="Cosine reference")
    axes[0, 0].set_ylabel("Tilt (deg)")
    axes[0, 0].set_title("Optimized tilt schedule")
    axes[0, 0].legend()

    axes[0, 1].plot(
        time,
        -np.rad2deg(data["tilt_rate_rad_s"]),
        color="tab:orange",
    )
    axes[0, 1].axhline(
        np.rad2deg(MAX_TILT_RATE_RAD_S),
        color="black",
        linestyle=":",
        label="Rate limit",
    )
    axes[0, 1].axhline(-np.rad2deg(MAX_TILT_RATE_RAD_S), color="black", linestyle=":")
    axes[0, 1].set_ylabel("Tilt rate (deg/s)")
    axes[0, 1].set_title("Actuator-rate constraint")
    axes[0, 1].legend()

    axes[1, 0].plot(time, data["vx_mps"], label="Forward speed")
    axes[1, 0].plot(time, data["vz_mps"], label="Vertical speed")
    schedule_speed_reference, _ = build_transition_schedule_reference(time)
    axes[1, 0].plot(
        time,
        schedule_speed_reference,
        "--",
        label="Delayed speed reference",
    )
    axes[1, 0].axhline(TARGET_SPEED_MPS, color="black", linestyle=":", label="Target")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[1, 0].set_title("Optimized velocity")
    axes[1, 0].legend()

    axes[1, 1].plot(time, data["altitude_m"], color="tab:green")
    axes[1, 1].axhline(MIN_ALTITUDE_M, color="red", linestyle=":", label="Altitude floor")
    axes[1, 1].axhline(INITIAL_ALTITUDE_M, color="black", linestyle="--", label="Start/target")
    axes[1, 1].set_ylabel("Altitude (m)")
    axes[1, 1].set_title("Optimized altitude")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.3)
    fig.suptitle("Optimized longitudinal transition corridor")
    fig.tight_layout()
    fig.savefig(
        "figures/optimized_longitudinal_transition.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(time, data["tilting_pair_thrust_n"], label="Tilting pair")
    axes[0, 0].plot(time, data["fixed_pair_thrust_n"], label="Fixed pair")
    axes[0, 0].plot(
        time,
        data["front_horizontal_thrust_n"],
        label="Front pair horizontal component",
    )
    axes[0, 0].axhline(MAX_PAIR_THRUST_N, color="black", linestyle=":")
    axes[0, 0].set_ylabel("Pair thrust (N)")
    axes[0, 0].set_title("Rotor thrust and front horizontal component")
    axes[0, 0].legend()

    axes[0, 1].plot(time, data["lift"], label="Lift")
    axes[0, 1].plot(time, data["drag"], label="Drag")
    axes[0, 1].set_ylabel("Aerodynamic force (N)")
    axes[0, 1].set_title("Wing forces")
    axes[0, 1].legend()

    axes[1, 0].plot(time, data["force_x_body"], label="Body X")
    axes[1, 0].plot(time, data["force_z_body"], label="Body Z")
    axes[1, 0].set_ylabel("Body force (N)")
    axes[1, 0].set_title("Resultant body-axis forces")
    axes[1, 0].legend()

    axes[1, 1].plot(time, data["acceleration_x"], label="Forward")
    axes[1, 1].plot(time, data["acceleration_up"], label="Upward")
    axes[1, 1].set_ylabel("Acceleration (m/s²)")
    axes[1, 1].set_title("Inertial acceleration")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Time (s)")
        axis.grid(True, alpha=0.3)
    fig.suptitle("Optimized force and acceleration curves")
    fig.tight_layout()
    fig.savefig(
        "figures/optimized_transition_force_curves.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time, np.rad2deg(data["alpha_rad"]), label="Angle of attack")
    axes[0].axhline(
        np.rad2deg(MAX_ALPHA_RAD),
        color="red",
        linestyle=":",
        label="Configured stall-angle limit",
    )
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Stall-bounded angle of attack")
    axes[0].legend()

    axes[1].plot(time, data["wing_lift_vertical_n"], label="Wing lift, vertical")
    axes[1].plot(time, data["wing_drag_vertical_n"], label="Wing drag, vertical")
    axes[1].plot(time, data["front_vertical_thrust_n"], label="Front thrust, vertical")
    axes[1].plot(time, data["fixed_vertical_thrust_n"], label="Rear thrust, vertical")
    axes[1].plot(
        time,
        data["required_vertical_force_n"],
        "k--",
        label="Required vertical force",
    )
    axes[1].set_ylabel("Vertical force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Wing/rotor vertical-support contributions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        "figures/optimized_transition_lift_support.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    csv_columns = (
        "time_s",
        "nacelle_angle_deg",
        "nacelle_rate_deg_s",
        "alpha_rad",
        "vtail_deflection_rad",
        "vx_mps",
        "vz_mps",
        "altitude_m",
        "tilting_pair_thrust_n",
        "fixed_pair_thrust_n",
        "front_rotor_rpm",
        "front_rpm_limit",
        "front_rpm_margin",
        "rear_rotor_rpm",
        "rear_rpm_limit",
        "rear_rpm_margin",
        "airspeed",
        "lift_coefficient",
        "lift",
        "drag",
        "tail_lift_n",
        "tail_drag_n",
        "tail_vertical_force_n",
        "wing_pitch_moment_n_m",
        "vtail_pitch_moment_n_m",
        "rotor_pitch_moment_n_m",
        "pitch_moment_n_m",
        "wing_lift_vertical_n",
        "front_vertical_thrust_n",
        "front_horizontal_thrust_n",
        "fixed_vertical_thrust_n",
        "fixed_horizontal_thrust_n",
        "required_vertical_force_n",
        "vertical_support_margin_n",
        "vertical_force_balance_residual_n",
        "electrical_assessed",
        "electrical_feasible",
        "electrical_motor_current_a",
        "electrical_battery_current_a",
        "electrical_battery_voltage_v",
        "electrical_missing_parameters",
        "force_x_body",
        "force_z_body",
        "acceleration_x",
        "acceleration_up",
        "pitch",
    )
    csv_data = data | {
        "nacelle_angle_deg": tilt_deg,
        "nacelle_rate_deg_s": -np.rad2deg(data["tilt_rate_rad_s"]),
    }
    with open(
        "figures/optimized_transition_curves.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
        writer.writerows(
            [csv_data[key][index] for key in csv_columns]
            for index in range(data["time_s"].size)
        )


def main():
    data = add_model_outputs(solve_transition())
    plot_results(data)
    print(f"Maximum altitude loss: {data['max_altitude_loss_m']:.3f} m")
    print(f"Minimum altitude: {data['altitude_m'].min():.3f} m")
    print(f"Final airspeed: {data['vx_mps'][-1]:.3f} m/s")
    print(f"Maximum tilt rate: {np.rad2deg(np.max(np.abs(data['tilt_rate_rad_s']))):.3f} deg/s")
    print(f"Maximum angle of attack: {np.rad2deg(data['alpha_rad'].max()):.3f} deg")
    print(f"Maximum wing lift: {data['lift'].max():.3f} N")
    print(
        f"Maximum absolute pitch-moment residual: "
        f"{np.abs(data['pitch_moment_n_m']).max():.3e} N m"
    )
    print(
        f"Maximum V-tail deflection: "
        f"{np.rad2deg(np.max(np.abs(data['vtail_deflection_rad']))):.3f} deg "
        f"(limit +/-{np.rad2deg(VTAIL_LIMIT_RAD):.2f} deg)"
    )
    print(
        f"Maximum front rotor RPM: {data['front_rotor_rpm'].max():.1f} / "
        f"{data['front_rpm_limit'][0]:.1f}"
    )
    print(
        f"Maximum rear rotor RPM: {data['rear_rotor_rpm'].max():.1f} / "
        f"{data['rear_rpm_limit'][0]:.1f} (reserve-adjusted limit)"
    )
    print(
        f"Minimum RPM margins: front={data['front_rpm_margin'].min():.1f}, "
        f"rear={data['rear_rpm_margin'].min():.1f}"
    )
    if not np.any(data["electrical_assessed"]):
        missing = sorted({
            name
            for row in data["electrical_missing_parameters"]
            for name in row.split(";")
            if name
        })
        print(f"Electrical screening: unknown; configure YAML fields: {', '.join(missing)}")
    elif np.all(data["electrical_feasible"]):
        print("Electrical screening: all trajectory points pass the configured steady-state model")
    else:
        failed_points = np.count_nonzero(
            data["electrical_assessed"] & ~data["electrical_feasible"]
        )
        print(f"Electrical screening: {failed_points} trajectory points exceed configured limits")
    print(
        "Maximum vertical force-balance residual: "
        f"{np.abs(data['vertical_force_balance_residual_n']).max():.3e} N"
    )
    print("Saved figures/optimized_longitudinal_transition.png")
    print("Saved figures/optimized_transition_force_curves.png")
    print("Saved figures/optimized_transition_lift_support.png")
    print("Saved figures/optimized_transition_curves.csv")
    print(
        "Model scope: still-air 2D point-mass model with stall-bounded lift and "
        "quasi-steady pitch-moment balance at every node. No pitch rotational "
        "dynamics, V-tail rate limit, rotor-wash, or gust model; electrical "
        "assessment remains a separate steady-state screen."
    )


if __name__ == "__main__":
    main()
