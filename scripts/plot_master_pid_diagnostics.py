flowchart TB
    U(["User"]) --> DASH["Dashboard<br/>scripts/dashboard.py"]
    U --> STAND["Standalone scripts<br/>scripts/"]
    U --> TEST["Tests<br/>tests/"]

    subgraph INPUTS["Inputs and configuration"]
        PROFILE["Aircraft profile<br/>YAML / JSON / Excel"]
        PARAM["Geometry, mass, inertia,<br/>propulsion and PID gains"]
        MISSION["Mission, tilt reference<br/>and operating limits"]
        WIND["Wind and gust settings"]
    end

    PROFILE --> DASH
    DASH --> PARAM
    DASH --> MISSION
    DASH --> WIND

    subgraph CORE["Main dashboard simulation"]
        CONTROL["Guidance and control<br/>src/control/"]
        ALLOC["Control allocation<br/>src/allocation/"]
        PHYSICS["6-DOF aircraft dynamics<br/>src/dynamics/"]
        AERO["Wing and V-tail aerodynamics<br/>src/aero/"]
        DIST["Wind models<br/>src/wind/"]
        STATE["AircraftState<br/>position, velocity, attitude, rates, tilt"]
        CONTROL --> ALLOC
        ALLOC -->|"rotor, tilt and tail commands"| PHYSICS
        AERO -->|"forces and moments"| PHYSICS
        DIST -->|"wind"| CONTROL
        DIST -->|"wind"| PHYSICS
        PHYSICS -->|"next state"| STATE
        STATE -->|"feedback"| CONTROL
    end

    PARAM --> CONTROL
    PARAM --> PHYSICS
    MISSION --> CONTROL
    WIND --> DIST
    DASH -->|"run_simulation"| CORE
    CORE --> LOG["Recorded samples and diagnostics"]
    LOG --> RESULTS["Dashboard graphs, targets,<br/>corridor and stability assessments"]
    RESULTS --> REPORT["Optional Word / PDF report"]

    subgraph STUDIES["Additional analyses: separate workflows"]
        SCHED["Cosine / linear tilt references"]
        FORCES["Fixed-speed force sweeps"]
        ENVELOPE["Analytical envelope,<br/>trim and constrained corridor"]
        OPT["Reduced longitudinal and<br/>multi-corridor optimization"]
        MC["Monte Carlo, gust and<br/>controller comparisons"]
        PID["Exploratory four-rotor PID diagnostic"]
    end

    DASH -. "interactive studies" .-> STUDIES
    STAND --> STUDIES
    PARAM -. "selected studies" .-> STUDIES
    STUDIES --> CURVES["Study-specific curves and data"]
    CURVES --> U
    RESULTS --> U
    REPORT --> U
    DASH -.-> PX4["Optional PX4 SITL mapping draft<br/>not actuator commands or firmware"]
    TEST -. "checks selected components" .-> CORE
    TEST -. "checks selected studies" .-> STUDIES"""Exploratory four-rotor PID diagnostics; not a flight-ready controller."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.control.architecture import ControlLawArchitecture
from src.control.trajectory import ReferenceTrajectory
from src.dynamics.rotor import RotorModel
from src.dynamics.state import AircraftState
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.wind.gust import DrydenGust


PROFILE_PATH = ROOT / "figures" / "KADA UAM aircraft_profile.json"
CORRIDOR_PATH = ROOT / "configs" / "robust_corridor.csv"
OUTPUT_PATH = ROOT / "simulation_results" / "master_pid_diagnostics.png"

# The profile says 3000 RPM; 3000 rad/s is a separate, unverified assumption.
ASSUMED_MAX_OMEGA_RAD_S = 3000.0
ALTITUDE_GAINS = {"kp": 1.5, "ki": 0.05, "kd": 0.8}
SPEED_GAINS = {"kp": 0.0, "ki": 0.0, "kd": 0.01}
ATTITUDE_GAINS = {"kp": 10.0, "ki": 0.0, "kd": 0.1}
RATE_GAINS = {"kp": 3.0, "ki": 0.1, "kd": 0.02}


def simulate(
    duration=20.0,
    dt=0.01,
    pitch_limit_deg=60.0,
    *,
    altitude_gains=None,
    gust_sigma=3.5,
    gust_seed=99,
):
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and time step must be positive and finite")
    if not np.isfinite(pitch_limit_deg) or not 0 < pitch_limit_deg < 90:
        raise ValueError("pitch limit must be finite and between 0 and 90 degrees")
    if not np.isfinite(gust_sigma) or gust_sigma < 0:
        raise ValueError("gust sigma must be finite and non-negative")
    if round(duration / dt) < 1:
        raise ValueError("duration must include at least one simulation step")
    with PROFILE_PATH.open(encoding="utf-8") as file:
        aircraft = json.load(file)["aircraft"]
    masses = aircraft["mass_properties"]["component_masses_kg"]
    rotors = aircraft["geometry"]["rotors"]
    mass = masses["fuselage"] + masses["wing"] + masses["vtail"] + len(rotors) * masses["motor"]
    positions = tuple((*rotor["position_m"], rotor["tilt_enabled"]) for rotor in rotors)
    propulsion = aircraft["propulsion"]
    k_thrust = propulsion["propeller"]["thrust_coefficient_n_s2_rad2"]
    rotor = RotorModel(k_thrust, ASSUMED_MAX_OMEGA_RAD_S)
    tail_config = aircraft["geometry"]["vtail"]
    vtail = VtailModel(tail_config["area_m2"], tail_config["arm_m"])
    aero_config = aircraft["aerodynamics"]
    aero = AeroModel(
        aero_config["reference_area_m2"],
        vtail,
        prop_diameter_m=propulsion["propeller"]["diameter_in"] * 0.0254,
    )
    aero.AR = aero_config["aspect_ratio"]
    aero.CL_alpha = aero_config["lift_curve_slope_per_rad"]
    aero.CD_0 = aero_config["zero_lift_drag_coefficient"]
    aero.stall_angle = np.deg2rad(aero_config["stall_angle_deg"])
    inertia = np.asarray(aircraft["mass_properties"]["inertia_tensor_kg_m2"])
    dynamics = TiltrotorDynamics(mass, rotor, inertia=inertia, aero=aero)
    controller = ControlLawArchitecture(
        mass,
        k_thrust,
        ASSUMED_MAX_OMEGA_RAD_S,
        positions,
        vtail_model=vtail,
        aero_model=aero,
        tilt_pivot_xz=aircraft["geometry"]["tilt_axis_xz_m"],
        altitude_gains=ALTITUDE_GAINS if altitude_gains is None else altitude_gains,
        speed_gains=SPEED_GAINS,
        attitude_gains=ATTITUDE_GAINS,
        rate_gains=RATE_GAINS,
        max_moment_nm=25.0,
    )
    trajectory = ReferenceTrajectory(str(CORRIDOR_PATH), speed_max=12.0)
    wind = DrydenGust(
        sigma=(gust_sigma, gust_sigma, gust_sigma * 0.6),
        airspeed=12.0,
        seed=gust_seed,
    )
    max_tilt_step = np.deg2rad(aircraft["actuators"]["tilt"]["maximum_rate_deg_s"]) * dt
    state = AircraftState(pos=np.array([0.0, 0.0, -100.0]))
    log = {name: [] for name in (
        "time", "north", "east", "north_speed", "east_speed", "altitude",
        "reference_altitude", "climb_rate", "roll", "roll_rate", "pitch",
        "tilt", "reference_tilt", "yaw", "pitch_rate", "yaw_rate", "wind",
    )}
    stop_reason = None
    for i in range(round(duration / dt)):
        time = i * dt
        ref = trajectory.at(time)
        wind_ned = wind.step(dt)
        tilt = float(np.clip(ref["tilt"], state.tilt - max_tilt_step, state.tilt + max_tilt_step))
        limited_ref = {**ref, "tilt": tilt}
        command = controller.compute(state, limited_ref, dt, wind_ned, actuator_tilt_rad=tilt)
        state = dynamics.step(
            state, wind_ned,
            command["omega_fl"], command["omega_fr"], command["omega_rl"], command["omega_rr"],
            command["tilt"], command["delta_left"], command["delta_right"], dt,
        )
        values = np.concatenate((state.pos, state.vel, state.att, state.omega, [state.tilt]))
        if not np.all(np.isfinite(values)):
            stop_reason = f"non-finite state at {time + dt:.2f} s"
            break
        actual_time = time + dt
        actual_ref = trajectory.at(actual_time)
        sample = (
            actual_time, state.pos[0], state.pos[1], state.vel[0], state.vel[1],
            -state.pos[2], actual_ref["altitude"], -state.vel[2],
            np.rad2deg(state.att[0]), np.rad2deg(state.omega[0]),
            np.rad2deg(state.att[1]), np.rad2deg(state.tilt), np.rad2deg(actual_ref["tilt"]),
            np.rad2deg(state.att[2]), np.rad2deg(state.omega[1]),
            np.rad2deg(state.omega[2]), np.linalg.norm(wind_ned),
        )
        for key, value in zip(log, sample):
            log[key].append(float(value))
        if abs(state.att[1]) >= np.deg2rad(pitch_limit_deg):
            stop_reason = f"pitch exceeded {pitch_limit_deg:g} deg at {actual_time:.2f} s"
            break
        if state.pos[2] >= 0.0:
            stop_reason = f"altitude reached ground at {actual_time:.2f} s"
            break
    return {name: np.asarray(values) for name, values in log.items()}, stop_reason


def plot_results(log, stop_reason, output_path=OUTPUT_PATH):
    if not len(log["time"]):
        raise ValueError("simulation produced no finite samples")
    fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
    panels = (
        ("north", "North position (m)"), ("north_speed", "North velocity (m/s)"),
        ("east", "East position (m)"), ("east_speed", "East velocity (m/s)"),
        ("altitude", "Altitude (m)"), ("climb_rate", "Climb rate (m/s)"),
        ("roll", "Roll (deg)"), ("roll_rate", "Body roll rate p (deg/s)"),
        ("pitch", "Pitch (deg)"), ("tilt", "Nacelle tilt (deg)"),
    )
    for ax, (key, label) in zip(axes.flat, panels):
        ax.plot(log["time"], log[key], color="tab:blue", label="Simulated")
        if key == "altitude":
            ax.plot(log["time"], log["reference_altitude"], "k--", label="Corridor reference")
        if key == "tilt":
            ax.plot(log["time"], log["reference_tilt"], "k--", label="Requested tilt")
        if key in ("altitude", "tilt"):
            ax.legend(fontsize=8)
        if stop_reason:
            ax.axvline(log["time"][-1], color="tab:red", linestyle=":", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    status = f"STOPPED: {stop_reason}" if stop_reason else "Completed requested duration"
    fig.suptitle("Four-rotor PID diagnostics — exploratory simulation only\n" + status)
    fig.text(
        0.5, 0.01,
        "Truth-fed model; 3000 rad/s motor limit assumed, not verified. Not flight-ready.",
        ha="center", color="darkred", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    log, stop_reason = simulate()
    plot_results(log, stop_reason)
    error = log["altitude"] - log["reference_altitude"]
    print(f"Saved diagnostic plot: {OUTPUT_PATH}")
    print(f"Maximum altitude error before stop: {np.max(np.abs(error)):.2f} m")
    print(f"Final tilt: {log['tilt'][-1]:.1f} deg; requested: {log['reference_tilt'][-1]:.1f} deg")
    if stop_reason:
        print(f"Simulation stopped: {stop_reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
