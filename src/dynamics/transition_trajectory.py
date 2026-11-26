"""Longitudinal transition collocation with the configured aircraft loads."""

from dataclasses import dataclass

import casadi as ca
import numpy as np

from src.aero.wing_polar import low_speed_aerodynamic_factor, wing_lift_drag_coefficients
from src.configuration.aircraft_parameters import AircraftParameters


RPM_PER_RAD_S = 30.0 / np.pi


@dataclass(frozen=True)
class TransitionLimits:
    front_omega: float
    rear_omega: float
    tail_rad: float
    motor_slew_rpm_s: float = 6000.0
    tail_rate_deg_s: float = 60.0
    pitch_rate_deg_s: float = 25.0
    pitch_limit_deg: float = 25.0
    longitudinal_accel_mps2: float = 3.0
    vertical_accel_mps2: float = 2.0
    climb_rate_mps: float = 3.0
    altitude_loss_m: float = 3.0
    pitch_reserve_nm: float = 0.6
    min_pair_thrust_n: float = 4.0
    stall_reserve_deg: float = 3.0
    wingborne_speed_mps: float = 12.0


def longitudinal_loads(
    aircraft: AircraftParameters, low_speed_full_load_mps=0.5, *, exact_kinematics=False,
) -> ca.Function:
    """Return acceleration, pitch moment, and flow angles for symmetric flight.

    The state is [forward speed, climb rate, altitude change, pitch, pitch
    rate]. Actuators are [tilt, front rad/s, rear rad/s, symmetric tail angle].
    This is the longitudinal slice of the rigid-body model, outside ground effect.
    """
    state = ca.SX.sym("state", 5)
    control = ca.SX.sym("control", 4)
    vx, climb, _height, pitch, _pitch_rate = ca.vertsplit(state)
    tilt, front_omega, rear_omega, tail = ca.vertsplit(control)
    front_xz = np.array([pos[:3:2] for pos in aircraft.motor_positions if pos[3]])
    rear_x = [pos[0] for pos in aircraft.motor_positions if not pos[3]]
    if len(front_xz) != 2 or len(rear_x) != 2:
        raise ValueError("transition trajectory requires two tilting and two fixed rotors")
    front = 2.0 * aircraft.k_thrust * front_omega**2
    rear = 2.0 * aircraft.k_thrust * rear_omega**2
    pivot_x, pivot_z = aircraft.tilt_pivot_xz_m
    front_arm = float(front_xz[:, 0].mean()) - pivot_x + pivot_x * ca.cos(tilt) + pivot_z * ca.sin(tilt)
    rear_arm = float(np.mean(rear_x))

    speed_squared = vx**2 + climb**2
    speed = ca.sqrt(speed_squared + 1e-12)
    if exact_kinematics:
        # Keep the aft-flow quadrant in disturbed replay. Only regularize the
        # zero-flow origin, where the rigid-body aerodynamics are inactive.
        moving = speed_squared > 1e-12
        path_angle = ca.atan2(climb, ca.if_else(moving, vx, 1e-6))
        u_body = vx * ca.cos(pitch) + climb * ca.sin(pitch)
        w_body = vx * ca.sin(pitch) - climb * ca.cos(pitch)
        alpha = ca.atan2(w_body, ca.if_else(moving, u_body, 1e-6))
    else:
        path_angle = ca.atan2(climb, ca.fmax(vx, 0.05))
        alpha = pitch - path_angle
        u_body = speed * ca.cos(alpha)
        w_body = speed * ca.sin(alpha)
    disk_area = np.pi * (aircraft.prop_diameter_m / 2.0)**2
    induced = ca.sqrt((front + rear) / (8.0 * aircraft.air_density * disk_area) + 1e-12)
    u_eff = u_body + induced * ca.sin(tilt)
    w_eff = w_body + induced * ca.cos(tilt)
    angle_u = (
        ca.if_else(u_eff**2 + w_eff**2 > 1e-12, u_eff, 1e-6)
        if exact_kinematics else ca.fmax(u_eff, 1e-6)
    )
    alpha_eff = ca.atan2(w_eff, angle_u)
    cl_wing, cd_wing = wing_lift_drag_coefficients(
        alpha_eff,
        cl_zero=aircraft.wing_lift_coefficient_zero,
        lift_slope_per_rad=aircraft.wing_lift_slope,
        stall_angle_rad=np.deg2rad(aircraft.wing_stall_angle_deg),
        cd_zero=aircraft.wing_drag_coeff,
        aspect_ratio=aircraft.wing_aspect_ratio,
        oswald_efficiency=aircraft.oswald_efficiency,
    )
    active = low_speed_aerodynamic_factor(speed, low_speed_full_load_mps)
    q_wing_area = active * 0.5 * aircraft.air_density * (u_eff**2 + w_eff**2) * aircraft.wing_area
    lift, drag = q_wing_area * cl_wing, q_wing_area * cd_wing
    tail_cosine = np.cos(np.deg2rad(aircraft.vtail_dihedral_deg))
    cl_tail = ca.fmin(aircraft.vtail_max_lift_coefficient, ca.fmax(
        -aircraft.vtail_max_lift_coefficient, aircraft.vtail_lift_slope * (alpha + tail_cosine * tail),
    ))
    cd_tail = aircraft.vtail_drag_coeff + cl_tail**2 / (
        np.pi * aircraft.vtail_oswald_efficiency * aircraft.vtail_aspect_ratio
    )
    q_tail_area = active * 0.5 * aircraft.air_density * speed_squared * aircraft.vtail_area
    tail_lift = q_tail_area * cl_tail * tail_cosine
    tail_drag = q_tail_area * cd_tail
    wing_z_body = -drag * ca.sin(alpha) - lift * ca.cos(alpha)
    wing_moment = q_wing_area * aircraft.wing_mean_chord_m * (
        aircraft.wing_pitch_moment_zero + aircraft.wing_pitch_moment_slope * alpha_eff
    ) - aircraft.wing_aerodynamic_center_x_m * wing_z_body
    moment = front_arm * front + rear_arm * rear + wing_moment - aircraft.vtail_arm_length * tail_lift
    fx = (
        front * ca.sin(tilt - pitch) - rear * ca.sin(pitch)
        - drag * ca.cos(path_angle) - lift * ca.sin(path_angle)
        - tail_drag * ca.cos(pitch) - tail_lift * ca.sin(pitch)
    )
    f_up = (
        front * ca.cos(tilt - pitch) + rear * ca.cos(pitch)
        - drag * ca.sin(path_angle) + lift * ca.cos(path_angle)
        - tail_drag * ca.sin(pitch) + tail_lift * ca.cos(pitch)
    )
    return ca.Function("longitudinal_loads", [state, control], [ca.vertcat(
        fx / aircraft.mass, f_up / aircraft.mass - aircraft.gravity, moment,
        alpha, alpha_eff, front, rear, front_arm - rear_arm * ca.cos(tilt), lift,
        lift * ca.cos(path_angle), -drag * ca.sin(path_angle),
        tail_lift * ca.cos(pitch) - tail_drag * ca.sin(pitch),
        wing_moment, front_arm * front, rear_arm * rear, -aircraft.vtail_arm_length * tail_lift,
    )])


def optimize_trajectory(
    aircraft, limits, target_speed, guess, *, duration_s=32.0,
    low_speed_full_load_mps=0.5, tilt_schedule_exponent=None, flow_guard_extra_deg=0.0,
    waypoint=None, margin_objective=False,
):
    """Solve motion and actuator continuity at nodes and interval midpoints."""
    load = longitudinal_loads(aircraft, low_speed_full_load_mps)
    steps = guess.shape[1] - 1
    dt = duration_s / steps
    opti = ca.Opti()
    motion_scale = ca.diag(ca.DM([30.0, 3.0, 3.0, 0.25, 0.35]))
    control_scale = ca.diag(ca.DM([np.pi / 2, 12000 / RPM_PER_RAD_S, 12000 / RPM_PER_RAD_S, 0.2]))
    x_scaled = opti.variable(5, steps + 1)
    u_scaled = opti.variable(4, steps + 1)
    trim_scaled = opti.variable(4, steps + 1)
    x = motion_scale @ x_scaled
    u = control_scale @ u_scaled
    trim_scale = ca.diag(ca.DM([0.25, 12000 / RPM_PER_RAD_S, 12000 / RPM_PER_RAD_S, 0.2]))
    trim = trim_scale @ trim_scaled
    hover_omega = np.sqrt(aircraft.mass * aircraft.gravity / (4 * aircraft.k_thrust))
    opti.subject_to(x[:, 0] == 0)
    opti.subject_to(u[:, 0] == ca.DM([0.0, hover_omega, hover_omega, 0.0]))
    opti.subject_to(x[0, -1] == target_speed)
    opti.subject_to(x[1:3, -1] == 0)
    opti.subject_to(x[4, -1] == 0)
    opti.subject_to(u[0, -1] == np.pi / 2)
    opti.subject_to(load(x[:, -1], u[:, -1])[:3] == 0)
    if waypoint is not None:
        node, speed, deflection = waypoint
        if not 0 < node < steps:
            raise ValueError("waypoint node must be interior")
        opti.subject_to(x[0, node]**2 + x[1, node]**2 == speed**2)
        opti.subject_to(u[0, node] == deflection)
    rates = ca.diff(u, 1, 1) / dt
    opti.subject_to(opti.bounded(0, rates[0, :], aircraft.max_tilt_rate))
    opti.subject_to(opti.bounded(-limits.motor_slew_rpm_s / RPM_PER_RAD_S, rates[1:3, :], limits.motor_slew_rpm_s / RPM_PER_RAD_S))
    opti.subject_to(opti.bounded(-np.deg2rad(limits.tail_rate_deg_s), rates[3, :], np.deg2rad(limits.tail_rate_deg_s)))
    opti.subject_to(ca.diff(x[0, :], 1, 1) >= 0)
    opti.subject_to(opti.bounded(0, u[1, :], limits.front_omega))
    opti.subject_to(opti.bounded(0, u[2, :], limits.rear_omega))
    opti.subject_to(opti.bounded(-limits.tail_rad, u[3, :], limits.tail_rad))
    max_trim_pitch = np.deg2rad(aircraft.wing_stall_angle_deg)
    opti.subject_to(opti.bounded(-max_trim_pitch, trim[0, :], max_trim_pitch))
    opti.subject_to(opti.bounded(0, trim[1, :], limits.front_omega))
    opti.subject_to(opti.bounded(0, trim[2, :], limits.rear_omega))
    opti.subject_to(opti.bounded(-limits.tail_rad, trim[3, :], limits.tail_rad))

    stall = np.deg2rad(aircraft.wing_stall_angle_deg - limits.stall_reserve_deg)
    front_max = 2 * aircraft.k_thrust * limits.front_omega**2
    rear_max = 2 * aircraft.k_thrust * limits.rear_omega**2

    def constrain(state, control, forces):
        vx, climb, height, pitch, pitch_rate = ca.vertsplit(state)
        ax, az, _moment, alpha, alpha_eff, front, rear, arm = ca.vertsplit(forces[:8])
        opti.subject_to(opti.bounded(0, vx, target_speed + 0.02))
        opti.subject_to(opti.bounded(-limits.climb_rate_mps, climb, limits.climb_rate_mps))
        opti.subject_to(opti.bounded(-limits.altitude_loss_m, height, limits.altitude_loss_m))
        opti.subject_to(opti.bounded(-np.deg2rad(limits.pitch_limit_deg), pitch, np.deg2rad(limits.pitch_limit_deg)))
        opti.subject_to(opti.bounded(-np.deg2rad(limits.pitch_rate_deg_s), pitch_rate, np.deg2rad(limits.pitch_rate_deg_s)))
        opti.subject_to(opti.bounded(-limits.longitudinal_accel_mps2, ax, limits.longitudinal_accel_mps2))
        opti.subject_to(opti.bounded(-limits.vertical_accel_mps2, az, limits.vertical_accel_mps2))
        speed = ca.sqrt(vx**2 + climb**2 + 1e-12)
        free_alpha_limit = stall + (np.pi - stall) * ca.fmin(1, ca.fmax(0, (4.0 - speed) / 3.5))
        effective_alpha_limit = stall + (np.pi / 2 - stall) * ca.fmin(
            1, ca.fmax(0, (limits.wingborne_speed_mps - speed) / 8.0),
        )
        incidence_buffer = np.deg2rad(flow_guard_extra_deg) * ca.fmin(1, ca.fmax(0, speed - 0.5))
        opti.subject_to(opti.bounded(-free_alpha_limit + incidence_buffer, alpha, free_alpha_limit - incidence_buffer))
        opti.subject_to(opti.bounded(
            -effective_alpha_limit + incidence_buffer - 1e-6, alpha_eff,
            effective_alpha_limit - incidence_buffer + 1e-6,
        ))
        # Differential thrust can change pitch moment while holding the
        # body-vertical rotor force. Reserve authority in both directions.
        needed = limits.pitch_reserve_nm / arm
        rear_change = needed * ca.cos(control[0])
        opti.subject_to(opti.bounded(ca.fmax(needed, limits.min_pair_thrust_n), front, front_max - needed))
        opti.subject_to(opti.bounded(ca.fmax(rear_change, limits.min_pair_thrust_n), rear, rear_max - rear_change))

    def rhs(state, forces):
        return ca.vertcat(forces[0], forces[1], state[1], state[4], forces[2] / aircraft.inertia_yy)

    loads = load.map(steps + 1)(x, u)
    for node in range(steps + 1):
        constrain(x[:, node], u[:, node], loads[:, node])
        # Each speed–tilt node must also admit independent steady level trim.
        trim_speed = ca.sqrt(x[0, node]**2 + x[1, node]**2 + 1e-12)
        trim_load = load(
            ca.vertcat(trim_speed, 0, 0, trim[0, node], 0),
            ca.vertcat(u[0, node], trim[1:, node]),
        )
        opti.subject_to(trim_load[:3] == 0)
    for step in range(steps):
        left, right = x[:, step], x[:, step + 1]
        left_rate, right_rate = rhs(left, loads[:, step]), rhs(right, loads[:, step + 1])
        middle = (left + right) / 2 + dt / 8 * (left_rate - right_rate)
        middle_control = (u[:, step] + u[:, step + 1]) / 2
        middle_load = load(middle, middle_control)
        constrain(middle, middle_control, middle_load)
        integral = dt / 6 * (left_rate + 4 * rhs(middle, middle_load) + right_rate)
        opti.subject_to(ca.solve(motion_scale, right - left - integral) == 0)

    speed_reference = guess[0, :].reshape(1, -1)
    tilt_reference = guess[5, :].reshape(1, -1)
    tilt_tracking_weight = 0.3
    if tilt_schedule_exponent is not None:
        # Smaller exponents favor tilting early. Larger ones leave more
        # acceleration to the low-tilt part of the transition.
        ratio = ca.fmax(x[0, :] / target_speed, 0.0)
        offset = 0.025
        start = offset**tilt_schedule_exponent
        scale = (1.0 + offset)**tilt_schedule_exponent - start
        tilt_reference = (np.pi / 2) * ((ratio + offset)**tilt_schedule_exponent - start) / scale
        tilt_tracking_weight = 24.0
    control_change = ca.diff(u_scaled, 1, 1) / dt
    extra_cost = 0
    if margin_objective:
        omega = u[1:3, :]
        kv = aircraft.motor_kv_rpm_per_volt
        current = aircraft.motor_no_load_current_a + aircraft.propeller_torque_coefficient_nm_s2_rad2 * omega**2 * kv / RPM_PER_RAD_S
        voltage = omega * RPM_PER_RAD_S / kv + aircraft.motor_resistance_ohm * current
        power = 2 * ca.sum1(voltage * current) / aircraft.esc_efficiency
        peak_loss = opti.variable()
        opti.subject_to(peak_loss >= 0)
        opti.subject_to(peak_loss >= -x[2, :])
        extra_cost = 35 * peak_loss + 0.3 * ca.sum2(power) / 20 + 4 * ca.sumsqr(x_scaled[4, :]) + ca.sumsqr(u_scaled[3, :])
    opti.minimize(
        8.0 * ca.sumsqr(x_scaled[2, :]) + 0.3 * ca.sumsqr(x_scaled[1, :])
        + 1.0 * ca.sumsqr(x_scaled[4, :]) + 2.0 * ca.sumsqr(control_change)
        + 5.0 * ca.sumsqr(ca.diff(control_change, 1, 1))
        + 2.0 * ca.sumsqr((x[0, :] - speed_reference) / target_speed)
        + tilt_tracking_weight * ca.sumsqr(u[0, :] - tilt_reference) + 0.05 * ca.sumsqr(u_scaled[1:3, :])
        + 0.02 * ca.sumsqr(ca.diff(trim_scaled, 1, 1))
        + extra_cost
    )
    opti.set_initial(x_scaled, np.linalg.solve(np.asarray(motion_scale), guess[:5]))
    opti.set_initial(u_scaled, np.linalg.solve(np.asarray(control_scale), guess[5:]))
    opti.set_initial(trim_scaled, np.linalg.solve(np.asarray(trim_scale), guess[[3, 6, 7, 8]]))
    opti.solver("ipopt", {"print_time": False, "expand": True}, {
        "print_level": 0, "sb": "yes", "max_iter": 1500, "tol": 1e-7,
        "acceptable_tol": 1e-6,
    })
    solution = opti.solve()
    return np.linspace(0, duration_s, steps + 1), solution.value(x), solution.value(u)
