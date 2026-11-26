"""Interactive versions of the project's additional transition studies."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.control.reference import TransitionReference
from src.dynamics.nacelle import (
    CRUISE_TILT_DEG, HOVER_TILT_DEG, TILT_CONVENTION,
    deflection_rad_from_tilt_deg, tilt_deg_from_deflection_deg, tilt_deg_from_deflection_rad,
)
from src.dynamics.tiltrotor import GRAVITY
from src.dynamics.transition import transition_speed_envelope


def _line_chart(time, series, y_label, alt):
    frame = pd.DataFrame({"Time (s)": time, **series})
    st.line_chart(
        frame,
        x="Time (s)",
        x_label="Time (s)",
        y_label=y_label,
        alt=alt,
    )


def _schedule_analysis(max_tilt_rate_deg_s):
    st.caption(
        "Reference comparison only. These controls do not change the main simulation. "
        "The timed mission already uses a cosine tilt reference."
    )
    duration = st.slider("Tilt transition duration (s)", 3.0, 60.0, 10.0, 0.5, key="extra_tilt_duration")
    hold = st.slider("Initial tilt hold (s)", 0.0, 20.0, 2.0, 0.5, key="extra_tilt_hold")
    time = np.linspace(0.0, hold + duration + 2.0, 501)
    reference = TransitionReference(100.0, 20.0, duration, tilt_hold_time=hold)
    cosine = tilt_deg_from_deflection_rad([reference.at(float(t))["tilt"] for t in time])
    linear = tilt_deg_from_deflection_deg(90.0 * np.clip((time - hold) / duration, 0.0, 1.0))
    _line_chart(time, {"Cosine": cosine, "Linear": linear}, "Project tilt (deg)",
                "Cosine and linear tilt references versus time")
    _line_chart(time, {
        "Cosine": np.gradient(cosine, time),
        "Linear": np.gradient(linear, time),
        "Positive rate limit": np.full_like(time, max_tilt_rate_deg_s),
        "Negative rate limit": np.full_like(time, -max_tilt_rate_deg_s),
    }, "Tilt rate (deg/s)", "Tilt reference rates and configured rate limits")
    peak_rate = 45.0 * np.pi / duration
    st.caption(f"Cosine peak reference rate: {peak_rate:.1f} deg/s; configured actuator limit: {max_tilt_rate_deg_s:.1f} deg/s.")
    if peak_rate > max_tilt_rate_deg_s:
        st.warning("The cosine reference exceeds the configured actuator-rate limit; it is not an achievable command without rate limiting.")


def _force_analysis(dynamics):
    st.caption(
        "Instantaneous fixed-hover-speed sweeps using the current dashboard geometry and "
        "aerodynamics. Level attitude, zero wind, no tail forces and no rotor wash; "
        "these are not trim solutions or flight trajectories."
    )
    max_speed = st.slider("Force sweep maximum airspeed (m/s)", 5.0, 40.0, 20.0, 1.0, key="extra_force_speed")
    speeds = np.linspace(0.0, max_speed, 5)
    tilts = np.linspace(0.0, 90.0, 46)
    rotor = dynamics.rotor
    count = len(dynamics.mass_props.motor_positions)
    if rotor.k_thrust <= 0.0 or rotor.max_omega <= 0.0 or count == 0:
        st.error("Positive rotor thrust coefficient, motor speed and rotor count are required for the force sweep.")
        return
    required_omega = np.sqrt(dynamics.mass * GRAVITY / (count * rotor.k_thrust))
    omega = min(required_omega, rotor.max_omega)
    if required_omega > rotor.max_omega:
        st.warning("The configured rotors cannot produce the requested hover thrust; the sweep uses maximum rotor speed.")
    st.caption(f"Each rotor is held at {omega:.1f} rad/s; thrust is not re-trimmed as tilt or airspeed changes.")
    keys = ("rotor_forward", "rotor_up", "force_forward", "force_up", "accel_forward", "accel_up", "pitch_moment")
    values = {key: np.zeros((len(speeds), len(tilts))) for key in keys}
    for speed_index, speed in enumerate(speeds):
        for tilt_index, tilt_deg in enumerate(tilts):
            tilt = deflection_rad_from_tilt_deg(tilt_deg)
            rotor_force, rotor_moment, _ = dynamics._rotor_forces_and_moments(
                np.full(count, omega), tilt, 100.0
            )
            aero_force, aero_moment, _, _, _ = dynamics.aero.compute_forces_and_moments(
                np.array([speed, 0.0, 0.0]), np.zeros(3), np.zeros(3), 0.0, tilt
            )
            force = rotor_force + aero_force
            point = (
                rotor_force[0], -rotor_force[2], force[0], -force[2],
                force[0] / dynamics.mass, -force[2] / dynamics.mass - GRAVITY,
                (rotor_moment + aero_moment)[1],
            )
            for key, value in zip(keys, point):
                values[key][speed_index, tilt_index] = value
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Rotor forward force", "Rotor upward force", "Total forward force", "Total upward force",
    ))
    for speed_index, speed in enumerate(speeds):
        for key, row, col in (
            ("rotor_forward", 1, 1), ("rotor_up", 1, 2),
            ("force_forward", 2, 1), ("force_up", 2, 2),
        ):
            fig.add_trace(go.Scatter(
                x=tilts, y=values[key][speed_index], name=f"{speed:g} m/s",
                legendgroup=str(speed_index), showlegend=(row == 1 and col == 1),
            ), row=row, col=col)
    fig.update_xaxes(title_text=f"Rotor angle ({TILT_CONVENTION})")
    fig.update_yaxes(title_text="Force (N)")
    fig.update_layout(height=650, legend_title_text="Airspeed")
    st.plotly_chart(fig, width="stretch", alt="Fixed-rotor-speed forces versus tilt for different airspeeds")
    for key, label in (
        ("accel_forward", "Forward acceleration (m/s²)"),
        ("accel_up", "Upward acceleration (m/s²)"),
        ("pitch_moment", "Pitch moment (N m)"),
    ):
        figure = go.Figure(go.Heatmap(
            x=tilts, y=speeds, z=values[key], colorbar={"title": label},
        ))
        figure.update_layout(
            title=label, xaxis_title=f"Rotor angle ({TILT_CONVENTION})",
            yaxis_title="Airspeed (m/s)", height=350,
        )
        st.plotly_chart(figure, width="stretch", alt=f"{label} across nacelle tilt and airspeed")


def _envelope_analysis(dynamics):
    st.warning(
        "Exploratory analytical envelope only: vertical and forward maximum thrust are "
        "screened separately and may not be simultaneously achievable. No moment trim, "
        "actuator-rate, electrical or gust constraints are established."
    )
    speed_limit = st.slider("Displayed speed range (m/s)", 5.0, 200.0, 30.0, 5.0, key="extra_envelope_speed")
    aero = dynamics.aero
    rotor = dynamics.rotor
    envelope = transition_speed_envelope(
        mass=dynamics.mass,
        wing_area=aero.wing_area,
        max_thrust_per_rotor=rotor.compute_thrust(rotor.max_omega, 100.0),
        rotor_tilts=[bool(position[3]) for position in dynamics.mass_props.motor_positions],
        rho=aero.rho,
        cl_max=aero.CL_0 + aero.CL_alpha * aero.stall_angle,
        cd_0=aero.CD_0,
        aspect_ratio=aero.AR,
        samples=181,
    )
    angle = tilt_deg_from_deflection_deg(envelope["tilt_deg"])
    if not np.any(envelope["feasible"]):
        st.warning("No analytical envelope points were found for the current inputs.")
        return
    speeds = np.linspace(0.0, speed_limit, 301)
    feasible = (
        envelope["feasible"][:, None]
        & (speeds[None, :] >= envelope["minimum_speed_mps"][:, None])
        & (speeds[None, :] <= envelope["maximum_speed_mps"][:, None])
    )
    figure = go.Figure(go.Heatmap(
        x=speeds, y=angle, z=np.where(feasible, 1.0, np.nan),
        colorscale=[[0, "rgba(150,150,150,0.35)"], [1, "rgba(150,150,150,0.35)"]],
        showscale=False, hovertemplate="Airspeed: %{x:.2f} m/s<br>Tilt: %{y:.1f}°<extra>Analytical screen</extra>",
        name="Analytical screen",
    ))
    figure.add_trace(go.Scatter(x=envelope["minimum_speed_mps"], y=angle, name="Minimum speed", mode="lines"))
    figure.add_trace(go.Scatter(
        x=envelope["maximum_speed_mps"], y=angle, name="Maximum speed",
        mode="lines",
    ))
    figure.add_trace(go.Scatter(
        x=[0.0, 20.0], y=[HOVER_TILT_DEG, CRUISE_TILT_DEG], name="Illustrative tilt-scaled reference",
        mode="lines", line={"color": "black", "dash": "dot"},
    ))
    figure.update_layout(
        title="Analytical tilt versus speed envelope",
        xaxis_title="Airspeed (m/s)", yaxis_title=f"Rotor angle ({TILT_CONVENTION})",
        xaxis_range=[0, speed_limit], yaxis_range=[0, 90], height=500,
    )
    st.plotly_chart(figure, width="stretch", alt="Exploratory analytical tilt and speed envelope")
    st.caption(
        "The dotted 0–20 m/s line is an illustrative reference, not a simulated trajectory. "
        "The displayed speed range is a chart setting, not a validated aircraft speed limit."
    )


def _optimizer_analysis():
    st.warning(
        "Separate fixed-parameter 2D point-mass optimization, not the live aircraft simulation. "
        "It enforces quasi-steady pitch-moment balance and configured V-tail travel, "
        "but omits pitch rotational dynamics, tail rate, rotor wash and gust constraints. "
        "An infeasible result means the selected model scenario has no solution under those limits."
    )
    st.caption("Scenario: 5 kg, 1.5 m² wing, 15 s, 12 m/s target, 40 N per rotor pair. Dashboard aircraft controls do not change this scenario.")
    if st.button("Run reduced-order optimization", key="extra_run_optimizer"):
        st.session_state.pop("extra_optimizer_result", None)
        try:
            from scripts.optimize_longitudinal_transition import add_model_outputs, solve_transition

            with st.spinner("Solving the reduced longitudinal optimization..."):
                st.session_state["extra_optimizer_result"] = add_model_outputs(solve_transition())
        except (ImportError, RuntimeError, ValueError) as exc:
            st.error(f"Reduced-order optimization failed: {exc}")
    result = st.session_state.get("extra_optimizer_result")
    if result is None:
        st.info("Run the optimizer to display its interactive curves.")
        return
    time = result["time_s"]
    duration = float(time[-1])
    _line_chart(time, {
        "Optimized": tilt_deg_from_deflection_rad(result["tilt_rad"]),
        "Linear": tilt_deg_from_deflection_deg(90.0 * time / duration),
        "Cosine": tilt_deg_from_deflection_deg(45.0 * (1.0 - np.cos(np.pi * time / duration))),
    }, "Tilt (deg)", "Optimized, linear and cosine tilt schedules")
    _line_chart(time, {"Tilt rate": -np.rad2deg(result["tilt_rate_rad_s"]),
                       "Positive limit": np.full_like(time, np.rad2deg(0.35)),
                       "Negative limit": np.full_like(time, -np.rad2deg(0.35))},
                "Tilt rate (deg/s)", "Optimized tilt rate and model limits")
    for series, label, alt in (
        ({"Forward": result["vx_mps"], "Vertical": result["vz_mps"]}, "Velocity (m/s)", "Optimized forward and vertical velocity"),
        ({"Altitude": result["altitude_m"], "Target": np.full_like(time, 100.0)}, "Altitude (m)", "Optimized altitude and target"),
        ({"Tilting pair": result["tilting_pair_thrust_n"], "Fixed pair": result["fixed_pair_thrust_n"]}, "Thrust (N)", "Optimized rotor pair thrusts"),
        ({"Lift": result["lift"], "Drag": result["drag"]}, "Force (N)", "Reduced-model lift and drag"),
        ({"Body X": result["force_x_body"], "Body Z": result["force_z_body"]}, "Force (N)", "Reduced-model body forces"),
        ({"Forward": result["acceleration_x"], "Upward": result["acceleration_up"]}, "Acceleration (m/s²)", "Reduced-model accelerations"),
    ):
        _line_chart(time, series, label, alt)


def _pid_diagnostic():
    st.warning(
        "Experimental four-rotor diagnostic, separate from the main dashboard simulation. "
        "Uses the illustrative KADA profile, truth-fed wind and an unverified 3000 rad/s "
        "motor limit. It is not a validated tune or a flight-ready controller."
    )
    st.caption("Only the altitude PID and gust scenario are adjusted here; this does not change the main dashboard PID or aircraft profile.")
    kp = st.number_input("Diagnostic altitude Kp", min_value=0.0, max_value=20.0, value=1.5, step=0.1, key="extra_pid_kp")
    ki = st.number_input("Diagnostic altitude Ki", min_value=0.0, max_value=10.0, value=0.05, step=0.01, key="extra_pid_ki")
    kd = st.number_input("Diagnostic altitude Kd", min_value=0.0, max_value=20.0, value=0.8, step=0.1, key="extra_pid_kd")
    sigma = st.slider("Diagnostic gust sigma (m/s)", 0.0, 5.0, 3.5, 0.1, key="extra_pid_sigma")
    seed = st.number_input("Diagnostic gust seed", min_value=0, max_value=100000, value=99, step=1, key="extra_pid_seed")
    duration = st.slider("Diagnostic duration (s)", 1.0, 20.0, 20.0, 1.0, key="extra_pid_duration")
    from scripts.plot_master_pid_diagnostics import CORRIDOR_PATH, PROFILE_PATH, simulate

    try:
        file_versions = tuple(
            (stat.st_mtime_ns, stat.st_size)
            for stat in (path.stat() for path in (PROFILE_PATH, CORRIDOR_PATH))
        )
    except OSError as exc:
        st.error(f"Cannot access the diagnostic profile or corridor: {exc}")
        return
    signature = (kp, ki, kd, sigma, seed, duration, file_versions)
    if st.button("Run PID diagnostic", key="extra_run_pid"):
        st.session_state.pop("extra_pid_result", None)
        try:
            with st.spinner("Running the exploratory PID diagnostic..."):
                log, reason = simulate(
                    duration=duration,
                    altitude_gains={"kp": kp, "ki": ki, "kd": kd},
                    gust_sigma=sigma,
                    gust_seed=seed,
                )
            if not len(log["time"]):
                st.error(f"No finite diagnostic samples were produced: {reason}")
            else:
                st.session_state["extra_pid_result"] = {"signature": signature, "log": log, "reason": reason}
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"PID diagnostic failed: {exc}")
    result = st.session_state.get("extra_pid_result")
    if result is None:
        st.info("Run the PID diagnostic to display its curves.")
        return
    if result["signature"] != signature:
        st.info("Inputs changed. Run the PID diagnostic again to see updated curves.")
        return
    log, reason = result["log"], result["reason"]
    if reason:
        st.warning(f"Simulation stopped: {reason}. No later samples are shown.")
    else:
        st.info("The requested duration completed; this alone does not establish a safe or successful transition.")
    error = np.max(np.abs(log["altitude"] - log["reference_altitude"]))
    st.metric("Maximum absolute altitude error over recorded samples", f"{error:.2f} m")
    time = log["time"]
    for series, label, alt in (
        ({"North": log["north"]}, "Position (m)", "Diagnostic north position"),
        ({"North": log["north_speed"]}, "Velocity (m/s)", "Diagnostic north velocity"),
        ({"East": log["east"]}, "Position (m)", "Diagnostic east position"),
        ({"East": log["east_speed"]}, "Velocity (m/s)", "Diagnostic east velocity"),
        ({"Altitude": log["altitude"], "Reference": log["reference_altitude"]}, "Altitude (m)", "Diagnostic altitude and corridor reference"),
        ({"Climb rate": log["climb_rate"]}, "Climb rate (m/s)", "Diagnostic climb rate"),
        ({"Roll": log["roll"], "Pitch": log["pitch"], "Yaw": log["yaw"]}, "Angle (deg)", "Diagnostic roll pitch and yaw"),
        ({"p": log["roll_rate"], "q": log["pitch_rate"], "r": log["yaw_rate"]}, "Body rate (deg/s)", "Diagnostic body angular rates"),
        ({"Actual": log["tilt"], "Requested": log["reference_tilt"]}, "Tilt (deg)", "Diagnostic actual and requested nacelle tilt"),
        ({"Wind magnitude": log["wind"]}, "Wind speed (m/s)", "Diagnostic wind magnitude"),
    ):
        _line_chart(time, series, label, alt)


@st.fragment
def render_transition_analyses(dynamics, max_tilt_rate_deg_s):
    st.subheader("Additional interactive transition analyses")
    st.caption(
        "Choose a study below. These additional charts are not included in the downloadable simulation report."
    )
    analysis = st.selectbox(
        "Choose an analysis",
        ["Select an analysis", "Tilt schedules", "Force sweeps", "Analytical tilt-speed envelope",
         "Reduced-order optimization", "Exploratory PID diagnostic"],
        key="additional_transition_analysis",
    )
    if analysis == "Tilt schedules":
        _schedule_analysis(max_tilt_rate_deg_s)
    elif analysis == "Force sweeps":
        _force_analysis(dynamics)
    elif analysis == "Analytical tilt-speed envelope":
        _envelope_analysis(dynamics)
    elif analysis == "Reduced-order optimization":
        _optimizer_analysis()
    elif analysis == "Exploratory PID diagnostic":
        _pid_diagnostic()
