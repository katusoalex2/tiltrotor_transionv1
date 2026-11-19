"""Optimize a longitudinal tiltrotor transition and plot its model outputs."""

import os

import casadi as ca
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MASS_KG = 5.0
WING_AREA_M2 = 1.5
RHO_KG_M3 = 1.225
CL0 = 0.1
CD0 = 0.02
ASPECT_RATIO = 6.0
OSWALD_EFFICIENCY = 0.8
MAX_PAIR_THRUST_N = 40.0
MAX_TILT_RATE_RAD_S = 0.35
INITIAL_ALTITUDE_M = 100.0
MIN_ALTITUDE_M = 90.0
TARGET_SPEED_MPS = 12.0
DURATION_S = 15.0
SEGMENTS = 60
DT_S = DURATION_S / SEGMENTS
GRAVITY_MPS2 = 9.81


def longitudinal_dynamics(vx, vz, tilt, tilting_pair_thrust, fixed_pair_thrust):
    """Return point-mass forces and accelerations in body and NED axes."""
    speed_squared = vx**2 + vz**2
    dynamic_pressure = 0.5 * RHO_KG_M3 * speed_squared
    lift_coefficient = CL0
    drag_coefficient = CD0 + lift_coefficient**2 / (
        np.pi * OSWALD_EFFICIENCY * ASPECT_RATIO
    )
    lift = dynamic_pressure * WING_AREA_M2 * lift_coefficient
    drag = dynamic_pressure * WING_AREA_M2 * drag_coefficient

    force_x_body = tilting_pair_thrust * ca.sin(tilt) - drag
    force_z_body = (
        -tilting_pair_thrust * ca.cos(tilt)
        - fixed_pair_thrust
        - lift
    )

    flight_path_angle = ca.atan2(vz, ca.sqrt(vx**2 + 1e-8))
    pitch = flight_path_angle
    force_x_ned = (
        ca.cos(pitch) * force_x_body - ca.sin(pitch) * force_z_body
    )
    force_z_ned = (
        ca.sin(pitch) * force_x_body + ca.cos(pitch) * force_z_body
    )
    acceleration_x = force_x_ned / MASS_KG
    acceleration_up = -force_z_ned / MASS_KG - GRAVITY_MPS2
    return {
        "dynamic_pressure": dynamic_pressure,
        "lift": lift,
        "drag": drag,
        "force_x_body": force_x_body,
        "force_z_body": force_z_body,
        "force_x_ned": force_x_ned,
        "force_z_ned": force_z_ned,
        "acceleration_x": acceleration_x,
        "acceleration_up": acceleration_up,
        "pitch": pitch,
    }


def solve_transition():
    opti = ca.Opti()
    vx = opti.variable(SEGMENTS + 1)
    vz = opti.variable(SEGMENTS + 1)
    altitude = opti.variable(SEGMENTS + 1)
    tilt = opti.variable(SEGMENTS + 1)
    tilt_rate = opti.variable(SEGMENTS)
    tilting_thrust = opti.variable(SEGMENTS + 1)
    fixed_thrust = opti.variable(SEGMENTS + 1)
    max_altitude_loss = opti.variable()

    opti.subject_to(vx[0] == 0.0)
    opti.subject_to(vz[0] == 0.0)
    opti.subject_to(altitude[0] == INITIAL_ALTITUDE_M)
    opti.subject_to(tilt[0] == 0.0)
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
    opti.subject_to(opti.bounded(
        -MAX_TILT_RATE_RAD_S,
        tilt_rate,
        MAX_TILT_RATE_RAD_S,
    ))
    opti.subject_to(opti.bounded(0.0, tilting_thrust, MAX_PAIR_THRUST_N))
    opti.subject_to(opti.bounded(0.0, fixed_thrust, MAX_PAIR_THRUST_N))
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
        )
        ax = point["acceleration_x"]
        az_up = point["acceleration_up"]
        forward_accelerations.append(ax)
        upward_accelerations.append(az_up)
        opti.subject_to(opti.bounded(-3.0, ax, 3.0))
        opti.subject_to(opti.bounded(-2.0, az_up, 2.0))
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
    )
    opti.subject_to(terminal["acceleration_x"] == 0.0)
    opti.subject_to(terminal["acceleration_up"] == 0.0)

    normalized_thrust = (
        ca.sumsqr(tilting_thrust / MAX_PAIR_THRUST_N)
        + ca.sumsqr(fixed_thrust / MAX_PAIR_THRUST_N)
    ) * DT_S
    altitude_tracking = ca.sumsqr(
        (altitude - INITIAL_ALTITUDE_M) / 10.0
    ) * DT_S
    speed_reference = TARGET_SPEED_MPS * np.linspace(
        0.0,
        1.0,
        SEGMENTS + 1,
    )
    speed_tracking = ca.sumsqr((vx - speed_reference) / TARGET_SPEED_MPS) * DT_S
    tilt_rate_cost = ca.sumsqr(tilt_rate / MAX_TILT_RATE_RAD_S) * DT_S
    tilt_rate_smoothness = ca.sumsqr(
        ca.diff(tilt_rate) / MAX_TILT_RATE_RAD_S
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
        + 0.2 * thrust_smoothness
    )

    time = np.linspace(0.0, DURATION_S, SEGMENTS + 1)
    opti.set_initial(vx, TARGET_SPEED_MPS * time / DURATION_S)
    opti.set_initial(vz, np.zeros(SEGMENTS + 1))
    opti.set_initial(altitude, np.full(SEGMENTS + 1, INITIAL_ALTITUDE_M))
    opti.set_initial(tilt, np.pi / 2.0 * time / DURATION_S)
    opti.set_initial(
        tilt_rate,
        np.full(SEGMENTS, (np.pi / 2.0) / DURATION_S),
    )
    hover_pair_thrust = MASS_KG * GRAVITY_MPS2 / 2.0
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
    solution = opti.solve()
    if not solution.stats().get("success", False):
        raise RuntimeError("The longitudinal corridor optimization did not converge")

    output = {
        "time_s": time,
        "vx_mps": np.asarray(solution.value(vx), dtype=float).ravel(),
        "vz_mps": np.asarray(solution.value(vz), dtype=float).ravel(),
        "altitude_m": np.asarray(solution.value(altitude), dtype=float).ravel(),
        "tilt_rad": np.asarray(solution.value(tilt), dtype=float).ravel(),
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
    return output


def add_model_outputs(data):
    model_keys = (
        "lift",
        "drag",
        "force_x_body",
        "force_z_body",
        "force_x_ned",
        "force_z_ned",
        "acceleration_x",
        "acceleration_up",
        "pitch",
    )
    outputs = {key: np.empty_like(data["time_s"]) for key in model_keys}
    for index in range(data["time_s"].size):
        point = longitudinal_dynamics(
            data["vx_mps"][index],
            data["vz_mps"][index],
            data["tilt_rad"][index],
            data["tilting_pair_thrust_n"][index],
            data["fixed_pair_thrust_n"][index],
        )
        for key in model_keys:
            outputs[key][index] = float(point[key])
    return {**data, **outputs}


def plot_results(data):
    os.makedirs("figures", exist_ok=True)
    time = data["time_s"]
    tilt_deg = np.rad2deg(data["tilt_rad"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(time, tilt_deg, label="Optimized")
    axes[0, 0].plot(
        time,
        90.0 * time / DURATION_S,
        "--",
        label="Linear reference",
    )
    cosine = 45.0 * (1.0 - np.cos(np.pi * time / DURATION_S))
    axes[0, 0].plot(time, cosine, ":", label="Cosine reference")
    axes[0, 0].set_ylabel("Tilt (deg)")
    axes[0, 0].set_title("Optimized tilt schedule")
    axes[0, 0].legend()

    axes[0, 1].plot(
        time,
        np.rad2deg(data["tilt_rate_rad_s"]),
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
    axes[0, 0].axhline(MAX_PAIR_THRUST_N, color="black", linestyle=":")
    axes[0, 0].set_ylabel("Pair thrust (N)")
    axes[0, 0].set_title("Optimized rotor thrust")
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

    csv_columns = (
        "time_s",
        "tilt_rad",
        "tilt_rate_rad_s",
        "vx_mps",
        "vz_mps",
        "altitude_m",
        "tilting_pair_thrust_n",
        "fixed_pair_thrust_n",
        "lift",
        "drag",
        "force_x_body",
        "force_z_body",
        "acceleration_x",
        "acceleration_up",
        "pitch",
    )
    np.savetxt(
        "figures/optimized_transition_curves.csv",
        np.column_stack([data[key] for key in csv_columns]),
        delimiter=",",
        header=",".join(csv_columns),
        comments="",
    )


def main():
    data = add_model_outputs(solve_transition())
    plot_results(data)
    print(f"Maximum altitude loss: {data['max_altitude_loss_m']:.3f} m")
    print(f"Minimum altitude: {data['altitude_m'].min():.3f} m")
    print(f"Final airspeed: {data['vx_mps'][-1]:.3f} m/s")
    print(f"Maximum tilt rate: {np.rad2deg(np.max(np.abs(data['tilt_rate_rad_s']))):.3f} deg/s")
    print("Saved figures/optimized_longitudinal_transition.png")
    print("Saved figures/optimized_transition_force_curves.png")
    print("Saved figures/optimized_transition_curves.csv")
    print(
        "Model scope: 2D longitudinal point-mass model with fixed CL0; "
        "no pitch-moment, rotor-wash, gust, or electrical constraints."
    )


if __name__ == "__main__":
    main()
