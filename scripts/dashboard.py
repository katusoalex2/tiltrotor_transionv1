
# Cache Buster: 2026-11-11-00-37-15
"""KADA UAM Transition phase configurators _kONKUK"""
from collections.abc import Mapping

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import hashlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================================
# 1. FIX PATH FIRST (Crucial for Streamlit Cloud)
# ==========================================================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# ==========================================================
# 2. NOW IMPORT SRC MODULES
# ==========================================================
from src.aero.airfoil import generate_naca4
from src.dynamics.cad_geometry import CadGeometry
from src.allocation.allocator import ControlAllocator
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.dynamics.transition import (
    control_effectiveness_map,
    constrained_transition_corridor,
    solve_longitudinal_trim,
    transition_stability_map,
    transition_speed_envelope,
)
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.trajectory import ReferenceTrajectory
from src.control.architecture import ControlLawArchitecture
from src.control.envelope_protection import EnvelopeProtection
from src.control.transition_observer import TransitionStateObserver
from src.dynamics.mass_properties import MassProperties
from src.dynamics.frames import rotation_body_to_earth
from src.control.mission import MissionManager, MissionProfile, Waypoint
from src.configuration.aircraft_profile import (
    dashboard_defaults,
    count_profile_values,
    load_aircraft_profile,
    profile_with_dashboard_values,
    serialize_aircraft_profile,
    simulation_compatibility_issues,
)
from src.configuration.aircraft_profile_excel import (
    load_aircraft_profile_excel,
    serialize_aircraft_profile_excel,
)
from src.configuration.px4_sitl_mapping import (
    build_px4_sitl_mapping,
    serialize_px4_sitl_mapping,
)
from src.configuration.aircraft_profile_preview import (
    build_aircraft_profile_figure,
    build_transition_reference_animation,
)
from src.wind.gust import DiscreteGust as FiniteDurationGust, DrydenGust
from src.simulation.failures import ACTUATOR_CHANNELS
from src.simulation.monte_carlo import MonteCarloValidator
from src.simulation.report import ReportItem, SimulationReport
from src.simulation.transition_targets import assess_transition_targets
from scripts.dashboard_transition_analyses import render_transition_analyses

st.set_page_config(page_title="KADA UAM Transition phase configurators _kONKUK", layout="wide")

# ==========================================================
# 3. LOAD CONFIG
# ==========================================================
cfg_path = os.path.join(base_dir, "configs", "blue_uam.yaml")
profile_template_path = os.path.join(base_dir, "configs", "uam_profile_template.yaml")
with open(profile_template_path, "rb") as profile_file:
    template_profile_bytes = profile_file.read()
uploaded_profile = st.sidebar.file_uploader(
    "Load versioned aircraft profile (YAML/JSON/Excel)",
    type=["yaml", "yml", "json", "xlsx"],
    key="aircraft_profile_upload",
)
if uploaded_profile is not None:
    profile_bytes = uploaded_profile.getvalue()
    profile_filename = uploaded_profile.name
else:
    profile_bytes = template_profile_bytes
    profile_filename = "uam_profile_template.yaml"

profile_fingerprint = hashlib.sha256(profile_bytes).hexdigest()
if st.session_state.get("_cached_aircraft_profile_fingerprint") == profile_fingerprint:
    active_profile = st.session_state["_cached_aircraft_profile"]
else:
    try:
        if profile_filename.lower().endswith(".xlsx"):
            active_profile = load_aircraft_profile_excel(profile_bytes)
        else:
            active_profile = load_aircraft_profile(profile_bytes, profile_filename)
    except (TypeError, ValueError) as exc:
        st.sidebar.error(f"Aircraft profile was not loaded: {exc}")
        st.stop()
    st.session_state["_cached_aircraft_profile"] = active_profile
    st.session_state["_cached_aircraft_profile_fingerprint"] = profile_fingerprint

try:
    profile_defaults = dashboard_defaults(active_profile)
except ValueError as exc:
    st.sidebar.error(f"Aircraft profile parameters are invalid: {exc}")
    st.stop()
if st.session_state.get("_aircraft_profile_fingerprint") != profile_fingerprint:
    for key, value in profile_defaults.items():
        st.session_state[key] = value
    for index, position in enumerate(profile_defaults["motor_positions"]):
        st.session_state[f"geo_mx{index}"] = float(position[0])
        st.session_state[f"geo_my{index}"] = float(position[1])
        st.session_state[f"geo_mz{index}"] = float(position[2])
        st.session_state[f"geo_tilt{index}"] = bool(position[3])
    st.session_state["_aircraft_profile_fingerprint"] = profile_fingerprint

for key, value in profile_defaults.items():
    if key != "motor_positions" and st.session_state.get(key) is None:
        st.session_state[key] = value
for index, position in enumerate(profile_defaults["motor_positions"]):
    for key, value in zip(
        (f"geo_mx{index}", f"geo_my{index}", f"geo_mz{index}", f"geo_tilt{index}"),
        (*position[:3], position[3]),
    ):
        if st.session_state.get(key) is None:
            st.session_state[key] = value

# Initialize the CAD Geometry engine

geo_path = os.path.join(base_dir, "configs", "geometry_offsets.yaml")
cad = CadGeometry(geo_path)

GUST_DIRECTION_VECTORS = {
    "Front": np.array([1.0, 0.0, 0.0]),
    "Back": np.array([-1.0, 0.0, 0.0]),
    "Right": np.array([0.0, 1.0, 0.0]),
    "Left": np.array([0.0, -1.0, 0.0]),
    "Up": np.array([0.0, 0.0, -1.0]),
    "Down": np.array([0.0, 0.0, 1.0]),
    "Diagonal": np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0),
}

st.title("KADA UAM Transition phase configurators _kONKUK")
st.markdown("[Jump to additional interactive transition analyses](#additional-interactive-transition-analyses)")


st.sidebar.title("Aircraft and mission setup")
st.sidebar.caption(
    f"Active profile: {active_profile['metadata']['name']} "
    f"(schema {active_profile['schema_version']})"
)
side_tabs = st.sidebar.tabs(["Body", "Prop", "Geom", "Aero", "PID", "Gust", "Mission", "Safety"])

with side_tabs[0]:
    st.markdown("**Component Masses (kg)**")
    mass_fuselage = st.number_input("Fuselage Mass", step=0.1, key="fuselage_mass_kg")
    mass_wing = st.number_input("Wing Mass", step=0.1, key="wing_mass_kg")
    mass_vtail = st.number_input("V-Tail Mass", step=0.1, key="vtail_mass_kg")
    mass_motor = st.number_input("Single Motor Mass", step=0.01, key="motor_mass_kg")
    
    # Calculate total mass automatically
    mass = (
        mass_fuselage
        + mass_wing
        + mass_vtail
        + len(profile_defaults["motor_positions"]) * mass_motor
    )
    st.info(f"Total Mass: **{mass:.2f} kg**")
    
with side_tabs[1]:
    st.markdown("**Battery**")
    c1, c2 = st.columns(2)
    battery_cells = c1.number_input("Cells (S)", step=1, key="battery_cells")
    battery_mah = c2.number_input("Capacity (mAh)", step=100, key="battery_capacity_mah")
    
    c1, c2 = st.columns(2)
    battery_c_rating = c1.number_input("C-Rating", step=5, key="battery_c_rating")
    battery_v_nom = battery_cells * 3.7
    battery_v_max = battery_cells * 4.2
    
    c2.metric("Max Voltage", f"{battery_v_max:.1f} V")
    st.info(f"Max Discharge Current: **{(battery_mah/1000)*battery_c_rating:.1f} A**")
    
    st.markdown("**Motor**")
    motor_kv = st.number_input("Motor KV (RPM/V)", step=50, key="motor_kv")
    
    st.markdown("**Propeller**")
    c1, c2 = st.columns(2)
    prop_dia = c1.number_input("Diameter (in)", step=0.5, key="prop_diameter_in")
    prop_pitch = c2.number_input("Pitch (in)", step=0.5, key="prop_pitch_in")
    
    num_blades = st.number_input(
        "Number of Blades", step=1, min_value=2, max_value=5, key="prop_blades"
    )
    max_tilt_rate_deg_s = st.slider(
        "Maximum nacelle tilt rate (deg/s)",
        1.0,
        90.0,
        step=1.0,
        key="max_tilt_rate_deg_s",
    )
    
    # --- Physics Calculation ---
    # 1. Max RPM (without load)
    max_rpm = motor_kv * battery_v_max * 0.85  # 85% efficiency
    max_omega = max_rpm * 2 * np.pi / 60
    
    # 2. Dynamic Thrust Coefficient (approximate based on prop geometry)
    # Base k for 12x5.5 2-blade prop
    base_k = 1e-5
    dia_factor = (prop_dia / 12.0) ** 4
    pitch_factor = (prop_pitch / 5.5) ** 0.5
    blade_factor = (num_blades / 2.0) ** 0.8
    k_thrust = base_k * dia_factor * pitch_factor * blade_factor
    
    st.info(f"Calculated $k_{{thrust}}$: **{k_thrust:.2e}**")
    prop_col1, prop_col2 = st.columns(2)
    prop_col1.metric("Maximum motor speed", f"{max_rpm:.0f} RPM")
    prop_col2.metric("Ideal four-rotor thrust", f"{4 * k_thrust * max_omega**2:.1f} N")
    st.caption("Thrust estimate uses T = k·ω² and excludes ground effect and electrical limits.")

with side_tabs[2]:
    st.markdown("**Fuselage envelope (3D preview)**")
    st.number_input(
        "Length (m)", min_value=0.01, step=0.1, key="fuselage_length_m"
    )
    fuselage_col1, fuselage_col2 = st.columns(2)
    fuselage_col1.number_input(
        "Width (m)", min_value=0.01, step=0.01, key="fuselage_width_m"
    )
    fuselage_col2.number_input(
        "Height (m)", min_value=0.01, step=0.01, key="fuselage_height_m"
    )

    st.markdown("**Wing Planform**")
    c1, c2 = st.columns(2)
    root_x = c1.number_input("Root LE X (m)", step=0.05, key="wing_root_le_x_m")
    root_y = c2.number_input("Root Y (m)", step=0.05, key="wing_root_y_m")
    
    tip_x = c1.number_input("Tip LE X (m)", step=0.05, key="wing_tip_le_x_m")
    tip_y = c2.number_input("Tip Y (m)", step=0.05, key="wing_tip_y_m")
    
    root_c = c1.number_input("Root Chord (m)", step=0.05, key="wing_root_chord_m")
    tip_c = c2.number_input("Tip Chord (m)", step=0.05, key="wing_tip_chord_m")
    
    sweep = st.number_input("Sweep Angle (deg)", step=5.0, key="wing_sweep_deg")
    
    st.markdown("**V-Tail**")
    vt_root_x = st.number_input("V-Tail Root X (m)", step=0.05, key="vtail_root_x_m")
    st.number_input("V-Tail Tip LE X (m)", step=0.05, key="vtail_tip_x_m")
    vt_tip_y = st.number_input("V-Tail Tip Y (m)", step=0.05, key="vtail_tip_y_m")
    vt_tip_z = st.number_input("V-Tail Tip Z (m)", step=0.05, key="vtail_tip_z_m")
    vtail_area = st.number_input("V-Tail Area", step=0.01, key="vtail_area_m2")
    vtail_arm = st.number_input("V-Tail Arm (m)", step=0.05, key="vtail_arm_m")
    
    st.markdown("**Shared rotor tilt axis location (X/Z plane, m)**")
    pivot_col1, pivot_col2 = st.columns(2)
    tilt_pivot_x = pivot_col1.number_input("Pivot X", step=0.05, key="motor_tilt_pivot_x_m")
    tilt_pivot_z = pivot_col2.number_input(
        "Pivot Z (down positive)", step=0.05, key="motor_tilt_pivot_z_m"
    )

    st.markdown("**Motors**")
    def_motors = profile_defaults["motor_positions"]
    motor_positions = []
    for i in range(len(def_motors)):
        c1, c2, c3, c4 = st.columns(4)
        mx = c1.number_input(f"X{i+1}", step=0.05, key=f"geo_mx{i}")
        my = c2.number_input(f"Y{i+1}", step=0.05, key=f"geo_my{i}")
        mz = c3.number_input(f"Z{i+1}", step=0.01, key=f"geo_mz{i}")
        tilts = c4.checkbox(f"Tilt{i+1}", key=f"geo_tilt{i}")
        motor_positions.append([mx, my, mz, tilts])

with side_tabs[3]:
    st.markdown("**Aero Properties**")
    wing_area = 2.0 * ((root_c + tip_c) / 2.0) * tip_y
    st.info(f"Calculated Wing Area: {wing_area:.2f} m²")
    
    st.markdown("**Calculated Geometry**")
    # Re-calculate the wing properties for display
    wing_props = cad.compute_wing_properties(
        [root_x, root_y, 0.0], 
        [root_x - root_c, root_y, 0.0], 
        [tip_x, tip_y, 0.0], 
        [tip_x - tip_c, tip_y, 0.0]
    )
    
    st.write(f"Mean Aerodynamic Chord (MAC): {wing_props['mac']:.3f} m")
    st.write(f"Aerodynamic Center (AC) X: {wing_props['ac_x']:.3f} m")
    

with side_tabs[4]:
    alt_kp = st.number_input("Alt Kp", step=0.1, key="alt_kp")
    alt_ki = st.number_input("Alt Ki", step=0.01, key="alt_ki")
    alt_kd = st.number_input("Alt Kd", step=0.01, key="alt_kd")
    spd_kp = st.number_input("Spd Kp", step=0.01, key="speed_kp")
    spd_ki = st.number_input("Spd Ki", step=0.01, key="speed_ki")
    spd_kd = st.number_input("Spd Kd", step=0.01, key="speed_kd")
    att_kp = st.number_input("Att Kp", step=0.1, key="attitude_kp")
    att_ki = st.number_input("Att Ki", step=0.1, key="attitude_ki")
    att_kd = st.number_input("Att Kd", step=0.01, key="attitude_kd")
    rate_kp = st.number_input("Rate Kp", step=0.05, key="rate_kp")
    rate_ki = st.number_input("Rate Ki", step=0.01, key="rate_ki")
    rate_kd = st.number_input("Rate Kd", step=0.01, key="rate_kd")
    hover_rate_scale = st.slider(
        "Hover rate-gain scale", 0.5, 2.5, step=0.05, key="hover_rate_scale"
    )
    cruise_rate_scale = st.slider(
        "Cruise rate-gain scale", 0.25, 1.5, step=0.05, key="cruise_rate_scale"
    )

with side_tabs[5]:
    gust_enabled = st.checkbox("Enable wind disturbance", key="gust_enabled")
    gust_model = st.selectbox(
        "Wind model",
        ["Finite-duration gust", "Dryden turbulence"],
        disabled=not gust_enabled,
        key="gust_model",
    )
    if gust_enabled and gust_model == "Finite-duration gust":
        gust_start = st.slider("Start Time (s)", 0.0, 15.0, step=0.5, key="gust_start_s")
        gust_duration = st.slider(
            "Duration (s)", 0.5, 5.0, step=0.5, key="gust_duration_s"
        )
        gust_mag = st.slider(
            "Magnitude (m/s)", 0.0, 15.0, step=0.5, key="gust_magnitude_mps"
        )
        gust_dir = st.selectbox(
            "Direction",
            ["Front", "Back", "Left", "Right", "Up", "Down", "Diagonal"],
            key="gust_direction",
        )
        dryden_sigma = profile_defaults["dryden_sigma_mps"]
        dryden_tau_s = profile_defaults["dryden_tau_s"]
    elif gust_enabled:
        dryden_sigma = st.slider(
            "Longitudinal and lateral gust σ (m/s)", 0.0, 8.0, step=0.25,
            key="dryden_sigma_mps",
        )
        dryden_tau_s = st.slider(
            "Gust correlation time (s)", 0.1, 5.0, step=0.1,
            key="dryden_tau_s",
        )
        gust_start, gust_duration, gust_mag, gust_dir = 0.0, 63.0, 0.0, "Dryden turbulence"
    else:
        gust_start, gust_duration, gust_mag, gust_dir = 0.0, 0.0, 0.0, "None"
        dryden_sigma, dryden_tau_s = 0.0, 1.0

with side_tabs[6]:
    flight_plan = st.selectbox(
        "Flight plan",
        ["Timed transition", "Waypoint route"],
        help="Use the existing hover-to-cruise profile or sequence through the editable NED waypoints.",
        key="flight_plan",
    )
    mission_mode = st.selectbox(
        "Autopilot mode",
        ["AUTO", "HOLD", "MANUAL"],
        help="HOLD captures the initial position; MANUAL tracks the setpoints below.",
        key="mission_mode",
    )
    manual_altitude = st.number_input(
        "Manual altitude (m)", min_value=0.0, value=20.0, step=1.0, key="manual_altitude"
    )
    manual_speed = st.number_input(
        "Manual forward speed (m/s)", min_value=0.0, value=5.0, step=0.5, key="manual_speed"
    )
    manual_tilt_deg = st.slider(
        "Manual nacelle tilt (deg)", 0.0, 90.0, 30.0, 1.0, key="manual_tilt_deg"
    )

    default_waypoints = pd.DataFrame([
        {"north_m": 0.0, "east_m": 0.0, "altitude_m": 20.0, "speed_mps": 0.0, "tilt_deg": 0.0},
        {"north_m": 30.0, "east_m": 0.0, "altitude_m": 20.0, "speed_mps": 6.0, "tilt_deg": 45.0},
        {"north_m": 120.0, "east_m": 0.0, "altitude_m": 20.0, "speed_mps": 12.0, "tilt_deg": 90.0},
    ])
    waypoint_table = st.data_editor(
        default_waypoints,
        num_rows="dynamic",
        hide_index=True,
        key="mission_waypoints",
    )
    st.caption("Waypoints use local NED position (north/east in metres); altitude is positive up.")
    waypoint_columns = ["north_m", "east_m", "altitude_m", "speed_mps", "tilt_deg"]
    route_values = waypoint_table[waypoint_columns]
    has_incomplete_waypoint = route_values.isna().any(axis=1).any()
    waypoint_rows = tuple(
        tuple(float(value) for value in row)
        for row in route_values.dropna().itertuples(index=False, name=None)
    )
    if flight_plan == "Waypoint route":
        if has_incomplete_waypoint:
            st.error("Complete or remove every blank waypoint row before running the waypoint route.")
            st.stop()
        if not waypoint_rows:
            st.error("Add at least one complete waypoint before selecting a waypoint route.")
            st.stop()
        route_array = np.asarray(waypoint_rows, dtype=float)
        if not np.all(np.isfinite(route_array)):
            st.error("Waypoint values must be finite numbers.")
            st.stop()
        if np.any(route_array[:, 2] < 0.0) or np.any(route_array[:, 3] < 0.0):
            st.error("Waypoint altitude and speed must be non-negative.")
            st.stop()
        if np.any((route_array[:, 4] < 0.0) | (route_array[:, 4] > 90.0)):
            st.error("Waypoint tilt must be between 0 and 90 degrees.")
            st.stop()

with side_tabs[7]:
    st.markdown("**Transition protection**")
    enable_envelope_protection = st.checkbox(
        "Enable envelope protection", key="enable_envelope_protection"
    )
    observer_time_constant_s = st.slider(
        "Observer filter time constant (s)", 0.05, 1.0, step=0.05,
        key="observer_time_constant_s",
    )
    max_alpha_deg = st.slider(
        "Maximum angle of attack (deg)", 5.0, 30.0, step=1.0,
        key="max_alpha_deg",
    )
    max_pitch_deg = st.slider(
        "Maximum pitch protection limit (deg)", 20.0, 60.0, step=1.0,
        key="max_pitch_deg",
    )
    max_body_rate_deg_s = st.slider(
        "Maximum body rate (deg/s)", 30.0, 240.0, step=5.0,
        key="max_body_rate_deg_s",
    )
    speed_margin_mps = st.slider(
        "Transition-envelope speed margin (m/s)", 0.0, 3.0, step=0.1,
        key="speed_margin_mps",
    )
    st.markdown("**Monte Carlo validation**")
    monte_carlo_trials = st.number_input(
        "Trial count", min_value=2, max_value=30, value=10, step=1,
        key="monte_carlo_trials",
    )
    monte_carlo_seed = st.number_input(
        "Random seed", min_value=0, value=42, step=1, key="monte_carlo_seed"
    )
    monte_carlo_gust_sigma = st.slider(
        "Maximum gust σ (m/s)", 0.0, 8.0, 3.0, 0.25,
        key="monte_carlo_gust_sigma",
    )
    monte_carlo_failure_probability = st.slider(
        "Actuator failure probability", 0.0, 1.0, 0.1, 0.05,
        key="monte_carlo_failure_probability",
    )
    st.caption("Actuator faults begin between 5 and 45 s; each trial uses a reproducible seed.")

dashboard_profile_values = {
    key: st.session_state[key]
    for key in profile_defaults
    if key != "motor_positions"
}
dashboard_profile_values["motor_positions"] = [
    [float(x), float(y), float(z), bool(tilts)]
    for x, y, z, tilts in motor_positions
]
export_profile = profile_with_dashboard_values(active_profile, dashboard_profile_values)
dashboard_values_fingerprint = hashlib.sha256(
    (profile_fingerprint + repr(dashboard_profile_values)).encode("utf-8")
).hexdigest()
with st.sidebar.expander("Aircraft profile export"):
    st.caption(
        "Profiles preserve custom aircraft, sensor, actuator, control, and hardware fields. "
        "Upload or download JSON, YAML, and the editable multi-sheet Excel workbook. "
        "Only the four-rotor tiltrotor subset is currently simulated."
    )
    if st.button("Prepare current profile exports", key="prepare_aircraft_profile"):
        st.session_state["aircraft_profile_download_bytes"] = serialize_aircraft_profile(
            export_profile
        )
        st.session_state["aircraft_profile_excel_download_bytes"] = (
            serialize_aircraft_profile_excel(export_profile)
        )
        template_profile = load_aircraft_profile(
            template_profile_bytes,
            "uam_profile_template.yaml",
        )
        st.session_state["aircraft_profile_template_excel_bytes"] = (
            serialize_aircraft_profile_excel(template_profile)
        )
        st.session_state["aircraft_profile_download_fingerprint"] = (
            dashboard_values_fingerprint
        )
    if (
        st.session_state.get("aircraft_profile_download_fingerprint")
        == dashboard_values_fingerprint
    ):
        st.download_button(
            "Download active profile (JSON)",
            data=st.session_state["aircraft_profile_download_bytes"],
            file_name="aircraft_profile.json",
            mime="application/json",
            key="download_active_aircraft_profile",
        )
        st.download_button(
            "Download active profile (Excel)",
            data=st.session_state["aircraft_profile_excel_download_bytes"],
            file_name="aircraft_profile.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_active_aircraft_profile_excel",
        )
    else:
        st.info("Prepare fresh exports after changing aircraft parameters.")
    if "aircraft_profile_template_excel_bytes" in st.session_state:
        st.download_button(
            "Download editable profile template (Excel)",
            data=st.session_state["aircraft_profile_template_excel_bytes"],
            file_name="uam_aircraft_profile_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_aircraft_profile_template_excel",
        )
    st.download_button(
        "Download editable profile template (YAML)",
        data=template_profile_bytes,
        file_name="uam_profile_template.yaml",
        mime="application/yaml",
        key="download_aircraft_profile_template",
    )
    st.caption(
        "The PX4 integration below exports SITL mapping metadata only. No autopilot connection "
        "or actuator command path is implemented."
    )

with st.sidebar.expander("PX4 SITL mapping (not for upload)"):
    st.warning(
        "This JSON is a mapping scaffold, not a PX4 parameter file. It cannot be flashed "
        "to Pixhawk."
    )
    px4_version = st.text_input(
        "PX4 release",
        value="unknown",
        key="px4_mapping_version",
        help="Enter the exact release when selected; parameter names are not generated here.",
    )
    px4_flight_controller = st.text_input(
        "Flight-controller board",
        value="unknown",
        key="px4_mapping_flight_controller",
    )
    tilt_arrangement_labels = {
        "Unspecified": "UNSPECIFIED",
        "One shared servo drives the tilting rotors": "SHARED_SERVO",
        "Independent servos drive the tilting rotors": "INDEPENDENT_SERVOS",
    }
    tilt_arrangement_label = st.selectbox(
        "Tilt-servo arrangement",
        list(tilt_arrangement_labels),
        key="px4_tilt_servo_arrangement",
    )
    aircraft_profile = export_profile.get("aircraft", {})
    profile_geometry = (
        aircraft_profile.get("geometry", {})
        if isinstance(aircraft_profile, Mapping)
        else {}
    )
    profile_rotors = (
        profile_geometry.get("rotors")
        if isinstance(profile_geometry, Mapping)
        else None
    )
    if not isinstance(profile_rotors, list) or not profile_rotors:
        st.info("Add rotor geometry to the aircraft profile before creating a PX4 mapping.")
    else:
        motor_directions = {}
        st.caption(
            "Rotor spin directions are placeholders until verified against the actual motors."
        )
        for index, rotor in enumerate(profile_rotors):
            rotor_id = (
                str(rotor.get("id", f"rotor_{index + 1}"))
                if isinstance(rotor, Mapping)
                else f"rotor_{index + 1}"
            )
            motor_directions[rotor_id] = st.selectbox(
                f"{rotor_id} rotation",
                ["UNVERIFIED", "CW", "CCW"],
                key=f"px4_motor_direction_{index}_{rotor_id}",
            )
        px4_mapping_options = {
            "px4_version": px4_version,
            "flight_controller": px4_flight_controller,
            "motor_directions": motor_directions,
            "tilt_servo_arrangement": tilt_arrangement_labels[tilt_arrangement_label],
        }
        try:
            px4_mapping = build_px4_sitl_mapping(export_profile, **px4_mapping_options)
            px4_mapping_json = serialize_px4_sitl_mapping(
                export_profile,
                **px4_mapping_options,
            )
        except ValueError as exc:
            st.error(f"Cannot create PX4 SITL mapping: {exc}")
        else:
            st.caption(
                f"{len(px4_mapping['unresolved_before_hardware_configuration'])} unresolved "
                "items are recorded in the export. Physical output assignments remain unassigned."
            )
            st.download_button(
                "Download PX4 SITL mapping JSON (not for upload)",
                data=px4_mapping_json,
                file_name="px4_sitl_mapping_NOT_FOR_UPLOAD.json",
                mime="application/json",
                key="download_px4_sitl_mapping",
            )
            with st.expander("Unresolved hardware and validation items"):
                for item in px4_mapping["unresolved_before_hardware_configuration"]:
                    st.markdown(f"- {item}")
                st.markdown("**PX4 references**")
                st.markdown(
                    f"- [Actuator setup]({px4_mapping['documentation']['px4_actuator_setup']})"
                )
                st.markdown(
                    f"- [Tiltrotor VTOL]({px4_mapping['documentation']['px4_tiltrotor_vtol']})"
                )

with st.sidebar.expander("Simulation report"):
    st.caption(
        "Generate Word and PDF reports with the current simulation, dashboard graphs, "
        "constrained corridor, stability map, and configured Monte Carlo trials. "
        "Generation may take several minutes and requires Chrome or Chromium for Plotly graphs."
    )
    generate_report = st.button("Generate full simulation report", key="generate_full_report")
    report_downloads = st.container()
if generate_report and "full_simulation_report" in st.session_state:
    del st.session_state["full_simulation_report"]
report = SimulationReport(generate_report)

report.text("markdown", st, "---")
report.text("subheader", st, "3D aircraft profile preview")
report.text("caption", st,
    " preview "
)
profile_preview_tilt_deg = st.slider(
    "Preview rotor tilt angle (deg)",
    min_value=0.0,
    max_value=90.0,
    value=0.0,
    step=1.0,
    key="aircraft_profile_preview_tilt_deg",
)
profile_preview_figure = build_aircraft_profile_figure(
    active_profile["metadata"]["name"],
    dashboard_profile_values,
    profile_preview_tilt_deg,
)
report.plotly_chart(st,
    profile_preview_figure,
    width="stretch",
    alt=f"3D profile preview of {active_profile['metadata']['name']} with configured lifting surfaces and rotors",
    key="aircraft_profile_preview",
)

if st.checkbox("Show reference-only transition animation", key="show_transition_reference"):
    st.warning(
        "REFERENCE ONLY: This schematic shows a predefined 0–90° tilt command, "
        "not simulated aircraft motion or evidence of a safe transition. It does "
        "not apply actuator-rate, aerodynamic, or other physical limits. "
        "Simulation results and warnings below remain unchanged."
    )
    st.caption(
        "Play or drag the time slider to inspect the timed reference. The airframe "
        "does not move; only rotors marked as tilting rotate. This animation is "
        "not included in simulation reports."
    )
    st.plotly_chart(
        build_transition_reference_animation(
            active_profile["metadata"]["name"], dashboard_profile_values
        ),
        width="stretch",
        alt="Reference-only schematic animation of a 0 to 90 degree rotor tilt command, not simulated flight",
        key="transition_reference_animation",
    )

profile_compatibility_issues = simulation_compatibility_issues(export_profile)
if profile_compatibility_issues:
    report.text("error", st,
        "This aircraft profile can be inspected and exported, but the current simulation "
        "adapter cannot run it."
    )
    for issue in profile_compatibility_issues:
        report.text("write", st, f"- {issue}")
    st.stop()

# Map gust settings to the wind variables expected by run_simulation
wind_enabled = gust_enabled
wind_speed = gust_mag
wind_direction = gust_start       # (see note below)
wind_turbulence = gust_duration   # (see note below)
surface_roughness = gust_dir      # (see note below)

@st.cache_data
def run_simulation(mass, mass_fuselage, mass_wing, mass_vtail, mass_motor, battery_cells, battery_mah, battery_c_rating, motor_kv, prop_dia, prop_pitch, num_blades, k_thrust, max_omega, wing_area, vtail_area, vtail_arm, root_x, root_c, tip_c, wingspan, sweep, motor_positions, alt_kp, alt_ki, alt_kd, spd_kp, spd_ki, spd_kd, att_kp, att_ki, att_kd, rate_kp, rate_ki, rate_kd, hover_rate_scale, cruise_rate_scale, gust_enabled, gust_start, gust_duration, gust_mag, gust_dir, max_tilt_rate_deg_s, flight_plan, mission_mode, manual_altitude, manual_speed, manual_tilt_deg, waypoint_rows, gust_model="Finite-duration gust", dryden_sigma_mps=(2.5, 2.5, 1.5), dryden_tau_s=1.0, dryden_seed=0, enable_envelope_protection=True, observer_time_constant_s=0.25, max_alpha_deg=15.0, max_pitch_deg=60.0, max_body_rate_deg_s=180.0, speed_margin_mps=0.5, actuator_failure=None, tilt_pivot_xz=(0.1, 0.0)):
    dt, duration = 0.01, 63.0
    
    rotor = RotorModel(k_thrust=k_thrust, max_omega=max_omega)
    vtail = VtailModel(area=vtail_area, arm_length=vtail_arm)
    prop_diameter_m = prop_dia * 0.0254
    aero = AeroModel(wing_area=wing_area, vtail=vtail, prop_diameter_m=prop_diameter_m)
    
    props = MassProperties(cfg_path)
    props.mass_fuselage = mass_fuselage
    props.mass_wing = mass_wing
    props.mass_vtail = mass_vtail
    props.mass_motor = mass_motor
    props.total_mass = mass
    props.vtail_area = vtail_area
    props.vtail_arm = vtail_arm
    props.wingspan = wingspan
    props.root_chord = root_c
    props.tip_chord = tip_c
    props.sweep_deg = sweep
    props.le_root_x = root_x
    props.motor_positions = [list(position) for position in motor_positions]
    props.tilt_pivot_xz = tilt_pivot_xz
    props.mac, props.ac_wing_x = props._calculate_mac_and_ac()
    props.neutral_point_x = props._calculate_neutral_point()
    
    dyn = TiltrotorDynamics(mass=mass, rotor=rotor, mass_props=props, aero=aero)
    maximum_rotor_thrust = rotor.compute_thrust(max_omega, height_above_ground=100.0)
    maximum_lift_coefficient = aero.CL_0 + aero.CL_alpha * aero.stall_angle
    transition_envelope = transition_speed_envelope(
        mass=mass,
        wing_area=wing_area,
        max_thrust_per_rotor=maximum_rotor_thrust,
        rotor_tilts=[bool(position[3]) for position in motor_positions],
        rho=aero.rho,
        cl_max=maximum_lift_coefficient,
        cd_0=aero.CD_0,
        aspect_ratio=aero.AR,
    )
    feasible_envelope = (
        transition_envelope["feasible"]
        & np.isfinite(transition_envelope["minimum_speed_mps"])
        & np.isfinite(transition_envelope["maximum_speed_mps"])
    )
    transition_observer = TransitionStateObserver(
        time_constant_s=observer_time_constant_s
    )
    envelope_protection = (
        EnvelopeProtection(
            transition_envelope,
            max_alpha_rad=np.deg2rad(max_alpha_deg),
            max_pitch_rad=np.deg2rad(max_pitch_deg),
            max_body_rate_rad_s=np.deg2rad(max_body_rate_deg_s),
            speed_margin_mps=speed_margin_mps,
        )
        if enable_envelope_protection and np.any(transition_envelope["feasible"])
        else None
    )
    
    controller = ControlLawArchitecture(
        mass=mass,
        k_thrust=k_thrust,
        max_omega=max_omega,
        motor_positions=motor_positions,
        tilt_pivot_xz=tilt_pivot_xz,
        vtail_model=vtail,
        aero_model=aero,
        altitude_gains={"kp": alt_kp, "ki": alt_ki, "kd": alt_kd},
        speed_gains={"kp": spd_kp, "ki": spd_ki, "kd": spd_kd},
        attitude_gains={"kp": att_kp, "ki": att_ki, "kd": att_kd},
        rate_gains={"kp": rate_kp, "ki": rate_ki, "kd": rate_kd},
        hover_rate_scale=hover_rate_scale,
        cruise_rate_scale=cruise_rate_scale,
        transition_observer=transition_observer,
        envelope_protection=envelope_protection,
    )
    ref = MissionProfile()
    waypoints = [
        Waypoint(
            north_m=row[0],
            east_m=row[1],
            altitude_m=row[2],
            speed_mps=row[3],
            tilt_rad=np.deg2rad(row[4]),
            name=f"WP {index + 1}",
        )
        for index, row in enumerate(waypoint_rows)
    ]
    mission_manager = MissionManager(waypoints)
    
    if gust_enabled and gust_model == "Finite-duration gust":
        peak_wind = gust_mag * GUST_DIRECTION_VECTORS[gust_dir]
        gust = FiniteDurationGust(
            gust_start,
            gust_duration,
            tuple(float(value) for value in peak_wind),
        )
    elif gust_enabled and gust_model == "Dryden turbulence":
        gust = DrydenGust(
            sigma=dryden_sigma_mps,
            airspeed=max(12.0, manual_speed),
            seed=dryden_seed,
            tau=dryden_tau_s,
        )
    else:
        gust = None

    state = AircraftState()
    if mission_mode == "MANUAL":
        initial_altitude = manual_altitude
    elif flight_plan == "Waypoint route" and waypoints:
        initial_altitude = waypoints[0].altitude_m
    else:
        initial_altitude = 100.0
    state.pos = np.array([0.0, 0.0, -initial_altitude])
    if mission_mode == "HOLD":
        mission_manager.set_mode("HOLD", state)
    elif mission_mode == "MANUAL":
        mission_manager.set_mode("MANUAL")
    
    log = {
        "t": [], "alt": [], "ref_alt": [], "pitch": [], "desired_pitch": [], 
        "wind": [], "wind_north": [], "wind_east": [], "wind_down": [],
        "tilt": [], "roll": [], "yaw": [], "ld_ratio": [],
        "current": [], "voltage": [], "ground_effect": [], "u_body": [], 
        "w_body": [], "v_body": [], "V_T": [], "beta": [], 
        "rotor_lift": [], "wing_lift": [], "wash_lift": [], "handoff_ratio": [],
        "tilt_cmd": [], "omega_fl": [], "omega_fr": [], "omega_rl": [], "omega_rr": [],
        "delta_left": [], "delta_right": [], "speed": [], "ref_speed": [],
        "north": [], "east": [], "down": [],
        "vel_north": [], "vel_east": [], "vel_down": [],
        "body_p": [], "body_q": [], "body_r": [],
        "force_x_body": [], "force_y_body": [], "force_z_body": [],
        "moment_l_body": [], "moment_m_body": [], "moment_n_body": [],
        "accel_north": [], "accel_east": [], "accel_down": [],
        "angular_accel_p": [], "angular_accel_q": [], "angular_accel_r": [],
        "mission_index": [], "phase": [],
        "desired_roll": [], "desired_heading": [],
        "desired_p": [], "desired_q": [], "desired_r": [],
        "moment_l_command": [], "moment_m_command": [], "moment_n_command": [],
        "allocation_residual": [], "allocation_saturated": [],
        "tilt_controller_cmd": [], "tilt_limited": [],
        "gain_blend": [], "gain_kp": [], "gain_ki": [], "gain_kd": [],
        "airspeed_mps": [], "guidance_altitude_error": [], "guidance_speed_error": [],
        "guidance_north_error": [], "guidance_east_error": [],
        "guidance_heading_error": [], "guidance_path_distance": [],
        "requested_thrust": [], "vertical_acceleration_command": [], "requested_tilt": [],
        "requested_speed": [], "protected_speed": [], "protected_tilt": [],
        "transition_airspeed": [], "transition_alpha_rad": [], "transition_beta_rad": [],
        "transition_blend": [], "transition_phase": [], "available_control_authority": [],
        "envelope_protected": [], "envelope_within": [], "envelope_reasons": [],
        "failure_active": [],
    }
    
    from src.dynamics.frames import rotation_earth_to_body
    from src.aero.handoff import compute_lift_breakdown
    termination_reason = "Mission duration completed"
    
    for i in range(int(duration/dt)):
        t = i * dt
        if mission_mode == "HOLD":
            r = mission_manager.update(state)["reference"]
            r["phase"] = "HOLD"
        elif mission_mode == "MANUAL":
            r = {
                "altitude": manual_altitude,
                "speed": manual_speed,
                "tilt": np.deg2rad(manual_tilt_deg),
                "phase": "MANUAL",
            }
        elif flight_plan == "Waypoint route":
            mission_status = mission_manager.update(state)
            route_reference = mission_status["reference"]
            if route_reference is None:
                route_reference = waypoints[-1].as_reference()
            r = {**route_reference, "phase": "WAYPOINT"}
        else:
            r = ref.at(t)

        if isinstance(gust, DrydenGust):
            wind_vec = gust.step(dt)
        elif gust is not None:
            wind_vec = gust.get_wind(t, altitude=max(-state.pos[2], 0.0))
        else:
            wind_vec = np.zeros(3, dtype=float)
        tilt_error = r["tilt"] - state.tilt
        tilt_step = np.deg2rad(max_tilt_rate_deg_s) * dt
        actual_tilt = state.tilt + np.clip(tilt_error, -tilt_step, tilt_step)
        cmd = controller.compute(
            state,
            r,
            dt,
            wind_vec,
            actuator_tilt_rad=actual_tilt,
        )
        failure_active = actuator_failure is not None and t >= actuator_failure.start_time_s
        if actuator_failure is not None:
            failed_commands = actuator_failure.apply(
                {name: cmd[name] for name in ACTUATOR_CHANNELS},
                t,
            )
            cmd.update(failed_commands)
        actual_tilt = state.tilt + np.clip(
            cmd["tilt"] - state.tilt,
            -tilt_step,
            tilt_step,
        )
        state = dyn.step(state, wind_vec, cmd["omega_fl"], cmd["omega_fr"], cmd["omega_rl"], cmd["omega_rr"], actual_tilt, cmd["delta_left"], cmd["delta_right"], dt)
        
        if abs(np.rad2deg(state.att[1])) > 60.0:
            termination_reason = "Stopped by the 60° pitch safety limit"
            break
            
        height_above_ground = max(-state.pos[2], 0.01)
        ge_ratio = rotor.get_ground_effect_ratio(height_above_ground)
        
        # --- WIND TRIANGLE LOGGING ---
        vel_air_earth = state.vel - wind_vec
        R_e2b = rotation_earth_to_body(state.att[0], state.att[1], state.att[2])
        vel_body = R_e2b @ vel_air_earth
        
        V_T_current = np.sqrt(vel_body[0]**2 + vel_body[1]**2 + vel_body[2]**2)
        alpha_current = np.arctan2(vel_body[2], vel_body[0])
        beta_current = np.rad2deg(np.arcsin(np.clip(vel_body[1] / max(V_T_current, 0.01), -1.0, 1.0)))
        
        log["u_body"].append(vel_body[0])
        log["v_body"].append(vel_body[1])
        log["w_body"].append(vel_body[2])
        log["V_T"].append(V_T_current)
        log["beta"].append(beta_current)
        if np.any(feasible_envelope):
            envelope_tilt_deg = np.rad2deg(state.tilt)
            minimum_envelope_speed = float(np.interp(
                envelope_tilt_deg,
                transition_envelope["tilt_deg"][feasible_envelope],
                transition_envelope["minimum_speed_mps"][feasible_envelope],
            ))
            maximum_envelope_speed = float(np.interp(
                envelope_tilt_deg,
                transition_envelope["tilt_deg"][feasible_envelope],
                transition_envelope["maximum_speed_mps"][feasible_envelope],
            ))
            inside_envelope = (
                minimum_envelope_speed - speed_margin_mps
                <= V_T_current
                <= maximum_envelope_speed + speed_margin_mps
            )
        else:
            inside_envelope = False
        log["envelope_within"].append(bool(inside_envelope))
        
        # --- LIFT HANDOFF LOGGING ---
        rotor_thrusts = [
            rotor.compute_thrust(omega, height_above_ground)
            for omega in (cmd["omega_fl"], cmd["omega_fr"], cmd["omega_rl"], cmd["omega_rr"])
        ]
        total_thrust = sum(rotor_thrusts)
        sorted_rotors = sorted(motor_positions, key=lambda position: (-position[0], position[1]))
        
        breakdown = compute_lift_breakdown(
            state, state.tilt, V_T_current, alpha_current, total_thrust,
            wing_area, mass, prop_radius=prop_dia * 0.0254 / 2.0,
            rotor_thrusts=rotor_thrusts,
            rotor_tilts=[bool(position[3]) for position in sorted_rotors],
        )
        log["rotor_lift"].append(breakdown["rotor_thrust"])
        log["wing_lift"].append(breakdown["wing_lift"])
        log["wash_lift"].append(breakdown["wash_lift"])
        log["handoff_ratio"].append(breakdown["handoff_ratio"])
        
        # --- MAIN LOGGING ---
        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["ref_alt"].append(r["altitude"])
        log["speed"].append(state.vel[0])
        log["ref_speed"].append(r["speed"])
        log["pitch"].append(np.rad2deg(state.att[1]))
        log["desired_pitch"].append(np.rad2deg(cmd["desired_pitch"]))
        log["wind"].append(np.linalg.norm(wind_vec))
        log["wind_north"].append(wind_vec[0])
        log["wind_east"].append(wind_vec[1])
        log["wind_down"].append(wind_vec[2])
        log["tilt"].append(np.rad2deg(state.tilt))
        log["tilt_cmd"].append(np.rad2deg(r["tilt"]))
        log["tilt_controller_cmd"].append(np.rad2deg(cmd["tilt"]))
        log["tilt_limited"].append(cmd["tilt_limited"])
        log["roll"].append(np.rad2deg(state.att[0]))
        log["yaw"].append(np.rad2deg(state.att[2]))
        for motor_name in ("omega_fl", "omega_fr", "omega_rl", "omega_rr"):
            log[motor_name].append(cmd[motor_name])
        log["delta_left"].append(cmd["delta_left"])
        log["delta_right"].append(cmd["delta_right"])
        log["north"].append(state.pos[0])
        log["east"].append(state.pos[1])
        log["down"].append(state.pos[2])
        log["vel_north"].append(state.vel[0])
        log["vel_east"].append(state.vel[1])
        log["vel_down"].append(state.vel[2])
        log["body_p"].append(state.omega[0])
        log["body_q"].append(state.omega[1])
        log["body_r"].append(state.omega[2])
        log["force_x_body"].append(dyn.last_diagnostics["force_body"][0])
        log["force_y_body"].append(dyn.last_diagnostics["force_body"][1])
        log["force_z_body"].append(dyn.last_diagnostics["force_body"][2])
        log["moment_l_body"].append(dyn.last_diagnostics["moment_body"][0])
        log["moment_m_body"].append(dyn.last_diagnostics["moment_body"][1])
        log["moment_n_body"].append(dyn.last_diagnostics["moment_body"][2])
        log["accel_north"].append(dyn.last_diagnostics["acceleration_ned"][0])
        log["accel_east"].append(dyn.last_diagnostics["acceleration_ned"][1])
        log["accel_down"].append(dyn.last_diagnostics["acceleration_ned"][2])
        log["angular_accel_p"].append(dyn.last_diagnostics["angular_acceleration_body"][0])
        log["angular_accel_q"].append(dyn.last_diagnostics["angular_acceleration_body"][1])
        log["angular_accel_r"].append(dyn.last_diagnostics["angular_acceleration_body"][2])
        log["mission_index"].append(mission_manager.sequencer.index)
        log["phase"].append(r["phase"])
        for key in (
            "desired_roll",
            "desired_heading",
            "desired_p",
            "desired_q",
            "desired_r",
            "moment_l_command",
            "moment_m_command",
            "moment_n_command",
            "gain_blend",
            "gain_kp",
            "gain_ki",
            "gain_kd",
            "airspeed_mps",
            "guidance_altitude_error",
            "guidance_speed_error",
            "guidance_north_error",
            "guidance_east_error",
            "guidance_heading_error",
            "guidance_path_distance",
            "requested_thrust",
            "vertical_acceleration_command",
            "requested_tilt",
            "requested_speed",
            "protected_speed",
            "protected_tilt",
            "available_control_authority",
            "envelope_protected",
        ):
            log[key].append(cmd[key])
        log["transition_airspeed"].append(cmd["transition_airspeed_mps"])
        log["transition_alpha_rad"].append(cmd["transition_alpha_rad"])
        log["transition_beta_rad"].append(cmd["transition_beta_rad"])
        log["transition_blend"].append(cmd["transition_blend"])
        log["transition_phase"].append(cmd["transition_phase"])
        log["envelope_reasons"].append(", ".join(cmd["envelope_reasons"]))
        log["failure_active"].append(failure_active)
        log["allocation_residual"].append(
            float(np.max(np.abs(cmd["allocation_residual_normalized"])))
        )
        log["allocation_saturated"].append(cmd["allocation_saturated"])
        
        if state.V_T >= 0.5 and state.drag > 1e-9:
            ld = state.lift / state.drag
        else:
            ld = np.nan
        log["ld_ratio"].append(ld)
        
        total_current = (cmd["omega_fl"] + cmd["omega_fr"] + cmd["omega_rl"] + cmd["omega_rr"]) / motor_kv
        max_current = (battery_mah / 1000) * battery_c_rating
        voltage_sag = 2.0 if total_current > max_current else 0.0
        log["current"].append(total_current)
        log["voltage"].append(battery_cells * 4.2 - voltage_sag)
        log["ground_effect"].append(ge_ratio)
        
    info = {
        "start": gust_start,
        "dur": gust_duration,
        "mag": gust_mag if gust_enabled else 0,
        "dir": gust_dir,
        "wind_model": gust_model if gust_enabled else "None",
        "gust_sigma_mps": tuple(dryden_sigma_mps),
        "failure": actuator_failure,
        "envelope_protection_active": envelope_protection is not None,
        "mode": mission_mode,
        "plan": flight_plan,
        "waypoints": waypoint_rows,
        "mission_complete": mission_manager.sequencer.complete,
        "termination_reason": termination_reason,
        "elapsed_time": float(log["t"][-1]) if log["t"] else 0.0,
        "allocation_matrix": controller.last_allocation_matrix.copy(),
        "allocation_residual": controller.last_allocation_residual.copy(),
        "allocation_saturation_fraction": float(np.mean(log["allocation_saturated"]))
        if log["allocation_saturated"] else 0.0,
        "tilt_limited_steps": int(np.count_nonzero(log["tilt_limited"])),
        "envelope_protected_fraction": float(np.mean(log["envelope_protected"]))
        if log["envelope_protected"] else 0.0,
        "envelope_outside_fraction": float(1.0 - np.mean(log["envelope_within"]))
        if log["envelope_within"] else 0.0,
        "failure_active_steps": int(np.count_nonzero(log["failure_active"])),
    }
    return {k: np.array(v) for k, v in log.items()}, info

simulation_parameters = {
    "mass": mass,
    "mass_fuselage": mass_fuselage,
    "mass_wing": mass_wing,
    "mass_vtail": mass_vtail,
    "mass_motor": mass_motor,
    "battery_cells": battery_cells,
    "battery_mah": battery_mah,
    "battery_c_rating": battery_c_rating,
    "motor_kv": motor_kv,
    "prop_dia": prop_dia,
    "prop_pitch": prop_pitch,
    "num_blades": num_blades,
    "k_thrust": k_thrust,
    "max_omega": max_omega,
    "wing_area": wing_area,
    "vtail_area": vtail_area,
    "vtail_arm": vtail_arm,
    "root_x": root_x,
    "root_c": root_c,
    "tip_c": tip_c,
    "wingspan": 2.0 * tip_y,
    "sweep": sweep,
    "motor_positions": tuple(tuple(position) for position in motor_positions),
    "tilt_pivot_xz": (tilt_pivot_x, tilt_pivot_z),
    "alt_kp": alt_kp,
    "alt_ki": alt_ki,
    "alt_kd": alt_kd,
    "spd_kp": spd_kp,
    "spd_ki": spd_ki,
    "spd_kd": spd_kd,
    "att_kp": att_kp,
    "att_ki": att_ki,
    "att_kd": att_kd,
    "rate_kp": rate_kp,
    "rate_ki": rate_ki,
    "rate_kd": rate_kd,
    "hover_rate_scale": hover_rate_scale,
    "cruise_rate_scale": cruise_rate_scale,
    "gust_enabled": gust_enabled,
    "gust_start": gust_start,
    "gust_duration": gust_duration,
    "gust_mag": gust_mag,
    "gust_dir": gust_dir,
    "max_tilt_rate_deg_s": max_tilt_rate_deg_s,
    "flight_plan": flight_plan,
    "mission_mode": mission_mode,
    "manual_altitude": manual_altitude,
    "manual_speed": manual_speed,
    "manual_tilt_deg": manual_tilt_deg,
    "waypoint_rows": waypoint_rows,
    "gust_model": gust_model,
    "dryden_sigma_mps": (dryden_sigma, dryden_sigma, dryden_sigma * 0.6),
    "dryden_tau_s": dryden_tau_s,
    "dryden_seed": 42,
    "enable_envelope_protection": enable_envelope_protection,
    "observer_time_constant_s": observer_time_constant_s,
    "max_alpha_deg": max_alpha_deg,
    "max_pitch_deg": max_pitch_deg,
    "max_body_rate_deg_s": max_body_rate_deg_s,
    "speed_margin_mps": speed_margin_mps,
    "actuator_failure": None,
}

with st.spinner("Running simulation..."):
    data, gust_info = run_simulation(**simulation_parameters)
if data["t"].size == 0:
    report.text("error", st, "The simulation ended before producing a valid flight sample.")
    st.stop()

if gust_info["wind_model"] == "Dryden turbulence":
    report.text("info", st,
        "Wind model: seeded Dryden-like turbulence "
        f"(σ={gust_info['gust_sigma_mps'][0]:.1f}/{gust_info['gust_sigma_mps'][1]:.1f}/"
        f"{gust_info['gust_sigma_mps'][2]:.1f} m/s; τ={dryden_tau_s:.1f} s)."
    )
elif wind_enabled and gust_info["mag"] > 0.0:
    report.text("info", st,
        f"Wind event: {gust_info['dir']} at {gust_info['mag']:.1f} m/s, "
        f"starting at t={gust_info['start']:.1f}s for {gust_info['dur']:.1f}s."
    )

col1, col2, col3, col4 = st.columns(4)
max_altitude_deviation = float(np.max(np.maximum(data["ref_alt"] - data["alt"], 0.0)))
report.metric(col1, "Max altitude shortfall", f"{max_altitude_deviation:.2f} m")
report.metric(col2, "Max Pitch", f"{np.max(np.abs(data['pitch'])):.2f} deg")
report.metric(col3, "Peak Gust", f"{np.max(data['wind']):.2f} m/s")
report.metric(col4, "Final Alt", f"{data['alt'][-1]:.1f} m")

mode_col, plan_col, phase_col, route_col, duration_col = st.columns(5)
report.metric(mode_col, "Mission mode", gust_info["mode"])
report.metric(plan_col, "Flight plan", gust_info["plan"])
report.metric(phase_col, "Current phase", str(data["phase"][-1]))
if mission_mode != "AUTO":
    route_status = mission_mode.title()
elif flight_plan == "Waypoint route":
    route_status = "Complete" if gust_info["mission_complete"] else (
        f"Waypoint {int(data['mission_index'][-1]) + 1}"
    )
else:
    route_status = "Timed"
report.metric(route_col, "Route status", route_status)
report.metric(duration_col, "Simulated duration", f"{gust_info['elapsed_time']:.2f} s")
if gust_info["termination_reason"] != "Mission duration completed":
    report.text("warning", st,
        f"Simulation stopped at t={gust_info['elapsed_time']:.2f} s: "
        f"{gust_info['termination_reason']}."
    )

report.text("markdown", st, "---")

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

# Show only the timed AUTO transition segment in the eight mission plots.
mission = MissionProfile()
transition_start_s = mission.hover_time
transition_end_s = mission.total_time
show_transition_window = flight_plan == "Timed transition" and mission_mode == "AUTO"
transition_times = np.asarray(data["t"], dtype=float)[
    np.asarray(data["phase"]) == "TRANSITION"
]
if show_transition_window and transition_times.size:
    transition_start_s = float(transition_times[0])
    transition_end_s = min(mission.total_time, float(data["t"][-1]))
phase_colors = ["#FFB6C1"]
mission_phases = (
    [("TRANSITION", transition_start_s, transition_end_s - transition_start_s)]
    if show_transition_window and transition_times.size
    else []
)

if show_transition_window:
    report.text("subheader", st, "Requested transition target assessment")
    target_rows = assess_transition_targets(
        data,
        transition_start_s=mission.hover_time,
        transition_end_s=mission.total_time,
    )
    report.dataframe(
        st,
        pd.DataFrame(target_rows),
        hide_index=True,
        alt="Requested transition targets and measured simulation outcomes",
    )
    report.text(
        "caption", st,
        "Simulation targets ",
    )
    report.text(
        "warning", st,
        "Static margin is not verified: center of gravity and aerodynamic data are "
        "unconfirmed. The roll and yaw tolerances and gust recovery criteria have not "
        "been specified; no flight-safety claim is made.",
    )
    if not gust_enabled:
        report.text("caption", st, "No-gust run. Gust testing remains available in the Gust tab.")
    else:
        report.text(
            "caption", st,
            "Gusts are enabled; recovery is not qualified without agreed gust and settling criteria.",
        )

def add_phase_background(ax):
    for (name, start, dur, *_), color in zip(mission_phases, phase_colors):
        ax.axvspan(start, start + dur, alpha=0.15, color=color)

# 1. Altitude
axes[0, 0].plot(data["t"], data["alt"], 'b-', linewidth=2, label="Altitude")
axes[0, 0].plot(data["t"], data["ref_alt"], 'k--', linewidth=1.5, label="Reference")
add_phase_background(axes[0, 0])
axes[0, 0].set_title(
    "Altitude (Transition)" if show_transition_window else "Altitude (Mission)"
)
axes[0, 0].set_ylabel("Altitude (m, positive up)")
axes[0, 0].grid(True)
axes[0, 0].legend()

# 2. Pitch
axes[0, 1].plot(data["t"], data["pitch"], 'r-', linewidth=2, label="Actual")
axes[0, 1].plot(data["t"], data["desired_pitch"], 'k--', linewidth=1.5, label="Desired")
axes[0, 1].axhspan(2.0, 2.8, color='green', alpha=0.15, label="Target after settling")
add_phase_background(axes[0, 1])
axes[0, 1].set_title("Pitch")
axes[0, 1].set_ylabel("Pitch angle (deg)")
axes[0, 1].grid(True)
axes[0, 1].legend()

# 3. Roll
axes[1, 0].plot(data["t"], data["roll"], 'orange', linewidth=2)
add_phase_background(axes[1, 0])
axes[1, 0].axhline(0, color='k', linestyle='--')
axes[1, 0].set_title("Roll")
axes[1, 0].set_ylabel("Roll angle (deg)")
axes[1, 0].grid(True)

# 4. Yaw
axes[1, 1].plot(data["t"], data["yaw"], 'purple', linewidth=2)
add_phase_background(axes[1, 1])
axes[1, 1].axhline(0, color='k', linestyle='--')
axes[1, 1].set_title("Yaw")
axes[1, 1].set_ylabel("Yaw angle (deg)")
axes[1, 1].grid(True)

# 5. Wind
axes[2, 0].plot(data["t"], data["wind"], 'g-', linewidth=2)
add_phase_background(axes[2, 0])
axes[2, 0].set_title("Wind Speed")
axes[2, 0].set_ylabel("Wind magnitude (m/s)")
axes[2, 0].grid(True)

# 6. Rotor Tilt
axes[2, 1].plot(data["t"], data["tilt"], 'm-', linewidth=2)
add_phase_background(axes[2, 1])
axes[2, 1].set_title("Rotor Tilt (Mission)")
axes[2, 1].set_ylabel("Nacelle tilt (deg)")
axes[2, 1].grid(True)

# 7. L/D Ratio
axes[3, 0].plot(data["t"], data["ld_ratio"], 'teal', linewidth=2)
add_phase_background(axes[3, 0])
axes[3, 0].set_title("Wing L/D estimate")
axes[3, 0].set_ylabel("Lift-to-drag ratio (dimensionless)")
axes[3, 0].grid(True)

# 8. Motor-speed/KV proxy
axes[3, 1].plot(data["t"], data["current"], 'b-', linewidth=2)
add_phase_background(axes[3, 1])
axes[3, 1].set_title("Motor-Speed/KV Proxy (not electrical current)")
axes[3, 1].set_ylabel("Motor-speed/KV proxy (model units)")
axes[3, 1].grid(True)

# Every panel uses mission time; timed AUTO plots are clipped to transition.
for ax in axes.flat:
    ax.set_xlabel("Mission time (s)")
    if show_transition_window:
        ax.set_xlim(transition_start_s, transition_end_s)

# Mark the transition boundary on the top row.
for ax in axes[0, :]:
    if show_transition_window:
        for boundary in (transition_start_s, transition_end_s):
            ax.axvline(boundary, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
report.pyplot(st,
    fig,
    alt=(
        "Transition altitude, attitude, wind, rotor tilt, lift-to-drag ratio, and "
        "motor-speed/KV proxy"
        if show_transition_window
        else "Mission altitude, attitude, wind, rotor tilt, lift-to-drag ratio, and "
        "motor-speed/KV proxy"
    ),
)
report.text(
    "caption", st,
    "Wing L/D is unassessed when airspeed or drag is too small; it is not forced "
    "to zero in hover or to a maximum at 90° tilt.",
)
if show_transition_window:
    if transition_times.size:
        completion_note = (
            f" The simulation ended before the planned transition completion at "
            f"{mission.total_time:.1f} s."
            if transition_end_s < mission.total_time
            else ""
        )
        report.text("caption", st,
            f"Showing recorded transition data only: {transition_start_s:.1f}–"
            f"{transition_end_s:.1f} s.{completion_note}"
        )
    else:
        report.text("caption", st,
            f"No transition samples were recorded. Showing the planned transition "
            f"window: {mission.hover_time:.1f}–{mission.total_time:.1f} s."
        )
report.text("caption", st,
    "The plotted motor-speed/KV value is a proxy, not an electrical current estimate; "
    "the model does not include motor resistance or battery current dynamics."
)

report.text("subheader", st, "Transition time histories")
report.text(
    "caption", st,
    "These are recorded transition samples and mission targets, not verified safety "
    "corridor boundaries. The simulated transition may end early.",
)
transition_mask = np.asarray(data["phase"]) == "TRANSITION"
if np.any(transition_mask):
    transition_time_frame = pd.DataFrame({
        "time_s": data["t"][transition_mask],
        "Altitude (m)": data["alt"][transition_mask],
        "Altitude target (m)": data["ref_alt"][transition_mask],
        "Horizontal airspeed (m/s)": data["airspeed_mps"][transition_mask],
        "Target speed (m/s)": data["ref_speed"][transition_mask],
        "Actual tilt (deg)": data["tilt"][transition_mask],
        "Tilt command (deg)": data["tilt_cmd"][transition_mask],
    })
    
    report.line_chart(
        st,
        transition_time_frame,
        x="time_s",
        y=["Altitude (m)", "Altitude target (m)"],
        x_label="Mission time (s)",
        y_label="Altitude (m, positive up)",
        alt="Transition altitude and altitude target versus mission time",
    )
    report.line_chart(
        st,
        transition_time_frame,
        x="time_s",
        y=["Horizontal airspeed (m/s)", "Target speed (m/s)"],
        x_label="Mission time (s)",
        y_label="Horizontal airspeed and target speed (m/s)",
        alt="Transition horizontal airspeed and target speed versus mission time",
    )
    report.line_chart(
        st,
        transition_time_frame,
        x="time_s",
        y=["Actual tilt (deg)", "Tilt command (deg)"],
        x_label="Mission time (s)",
        y_label="Nacelle tilt (deg)",
        alt="Transition actual and commanded tilt versus mission time",
    )
    transition_speeds = data["V_T"][transition_mask]
    transition_tilts = data["tilt"][transition_mask]
    transition_sample_times = data["t"][transition_mask]
    tilt_speed_figure = go.Figure()
    tilt_speed_figure.add_trace(go.Scatter(
        x=transition_speeds,
        y=transition_tilts,
        customdata=transition_sample_times,
        mode="lines",
        line=dict(color="darkorange", width=3),
        name="Recorded transition",
        hovertemplate=(
            "True airspeed: %{x:.2f} m/s<br>Tilt: %{y:.2f}°"
            "<br>Time: %{customdata:.2f} s<extra></extra>"
        ),
    ))
    for index, label, color in ((0, "Start", "green"), (-1, "Last recorded sample", "red")):
        tilt_speed_figure.add_trace(go.Scatter(
            x=[transition_speeds[index]],
            y=[transition_tilts[index]],
            customdata=[transition_sample_times[index]],
            mode="markers",
            marker=dict(color=color, size=9),
            name=label,
            hovertemplate=(
                f"{label}<br>True airspeed: %{{x:.2f}} m/s<br>"
                "Tilt: %{y:.2f}°<br>Time: %{customdata:.2f} s<extra></extra>"
            ),
        ))
    tilt_speed_figure.update_layout(
        title="Recorded nacelle tilt versus true airspeed",
        xaxis_title="True airspeed (m/s)",
        yaxis_title="Nacelle tilt (deg)",
        yaxis_range=[0, 90],
        legend=dict(orientation="h"),
        height=440,
    )
    report.plotly_chart(
        st,
        tilt_speed_figure,
        width="stretch",
        alt="Recorded transition nacelle tilt versus true airspeed in time order",
        key="transition_tilt_versus_airspeed",
    )
    report.text(
        "caption", st,
        "This curve connects recorded transition samples in time order, including any "
        "speed reversals. The last marker is not a completion indicator; stopped runs "
        "remain incomplete. True airspeed is relative to the surrounding air. This "
        "curve is not a verified safe transition corridor.",
    )
else:
    report.text("info", st, "No transition-phase samples were recorded in this simulation.")

report.text("markdown", st, "---")
report.text("subheader", st, "6-DOF equations and response")
report.text("caption", st,
    "Position and velocity are NED (north-east-down); forces and angular rates are FRD body-frame quantities."
)
report.text("markdown", st,
    r"""
$$
\dot{\mathbf p}_{NED}=\mathbf v_{NED}, \qquad
m\dot{\mathbf v}_{NED}=R_{B\to N}\mathbf F_{FRD}+m\mathbf g_{NED}
$$
$$
\begin{bmatrix}\dot\phi\\\dot\theta\\\dot\psi\end{bmatrix}
=T(\phi,\theta)\begin{bmatrix}p\\q\\r\end{bmatrix}, \qquad
\mathbf I\dot{\boldsymbol\omega}
=\mathbf M-\boldsymbol\omega\times(\mathbf I\boldsymbol\omega)-\dot{\mathbf I}\boldsymbol\omega
$$
"""
)

kinematics_tab, force_tab, moment_tab = st.tabs(
    ["Kinematics", "Newton force balance", "Euler moment balance"]
)
with kinematics_tab:
    position_col, velocity_col = st.columns(2)
    position_data = pd.DataFrame({
        "time_s": data["t"],
        "North (m)": data["north"],
        "East (m)": data["east"],
        "Down (m)": data["down"],
    })
    report.line_chart(position_col,
        position_data,
        x="time_s",
        y=["North (m)", "East (m)", "Down (m)"],
        x_label="Time (s)",
        y_label="NED position (m)",
        alt="Integrated north, east, and down position in the NED frame",
    )
    velocity_data = pd.DataFrame({
        "time_s": data["t"],
        "North (m/s)": data["vel_north"],
        "East (m/s)": data["vel_east"],
        "Down (m/s)": data["vel_down"],
    })
    report.line_chart(velocity_col,
        velocity_data,
        x="time_s",
        y=["North (m/s)", "East (m/s)", "Down (m/s)"],
        x_label="Time (s)",
        y_label="NED velocity (m/s)",
        alt="Integrated north, east, and down velocity in the NED frame",
    )

with force_tab:
    force_col, acceleration_col = st.columns(2)
    force_data = pd.DataFrame({
        "time_s": data["t"],
        "Forward X (N)": data["force_x_body"],
        "Right Y (N)": data["force_y_body"],
        "Down Z (N)": data["force_z_body"],
    })
    report.line_chart(force_col,
        force_data,
        x="time_s",
        y=["Forward X (N)", "Right Y (N)", "Down Z (N)"],
        x_label="Time (s)",
        y_label="Body force (N)",
        alt="Resultant rotor and aerodynamic force resolved in the FRD body frame",
    )
    acceleration_data = pd.DataFrame({
        "time_s": data["t"],
        "North (m/s²)": data["accel_north"],
        "East (m/s²)": data["accel_east"],
        "Down (m/s²)": data["accel_down"],
    })
    report.line_chart(acceleration_col,
        acceleration_data,
        x="time_s",
        y=["North (m/s²)", "East (m/s²)", "Down (m/s²)"],
        x_label="Time (s)",
        y_label="NED acceleration (m/s²)",
        alt="NED acceleration from body force, attitude transformation, and gravity",
    )

with moment_tab:
    rate_col, moment_col = st.columns(2)
    rate_data = pd.DataFrame({
        "time_s": data["t"],
        "p (rad/s)": data["body_p"],
        "q (rad/s)": data["body_q"],
        "r (rad/s)": data["body_r"],
    })
    report.line_chart(rate_col,
        rate_data,
        x="time_s",
        y=["p (rad/s)", "q (rad/s)", "r (rad/s)"],
        x_label="Time (s)",
        y_label="Body angular rate (rad/s)",
        alt="Body roll, pitch, and yaw rates p, q, and r",
    )
    moment_data = pd.DataFrame({
        "time_s": data["t"],
        "Roll L (N·m)": data["moment_l_body"],
        "Pitch M (N·m)": data["moment_m_body"],
        "Yaw N (N·m)": data["moment_n_body"],
    })
    report.line_chart(moment_col,
        moment_data,
        x="time_s",
        y=["Roll L (N·m)", "Pitch M (N·m)", "Yaw N (N·m)"],
        x_label="Time (s)",
        y_label="Body moment (N·m)",
        alt="Applied roll, pitch, and yaw moments in the FRD body frame",
    )
    angular_acceleration_data = pd.DataFrame({
        "time_s": data["t"],
        "p-dot (rad/s²)": data["angular_accel_p"],
        "q-dot (rad/s²)": data["angular_accel_q"],
        "r-dot (rad/s²)": data["angular_accel_r"],
    })
    report.line_chart(st,
        angular_acceleration_data,
        x="time_s",
        y=["p-dot (rad/s²)", "q-dot (rad/s²)", "r-dot (rad/s²)"],
        x_label="Time (s)",
        y_label="Body angular acceleration (rad/s²)",
        alt="Angular acceleration produced by applied moments and Euler gyroscopic coupling",
    )

# Mission phase legend
if mission_phases:
    report.text("markdown", st, "**Timed mission phases**")
    cols = st.columns(len(mission_phases))
    for i, (name, start, dur) in enumerate(mission_phases):
        report.metric(cols[i], name.title(), f"{start:.0f}–{start + dur:.0f} s")

report.text("markdown", st, "---")
report.text("subheader", st, "Guidance, control allocation, and mission")
report.text("caption", st,
    "The charts show the guidance references and the four motor/V-tail commands applied by the cascade controller."
)

tracking_col, mission_col = st.columns(2)
with tracking_col:
    report.text("markdown", st, "**Flight-path tracking**")
    altitude_data = pd.DataFrame({
        "time_s": data["t"],
        "Altitude (m)": data["alt"],
        "Altitude target (m)": data["ref_alt"],
    })
    report.line_chart(st,
        altitude_data,
        x="time_s",
        y=["Altitude (m)", "Altitude target (m)"],
        x_label="Time (s)",
        y_label="Altitude (m, positive up)",
        alt="Aircraft altitude compared with the mission altitude target",
    )
    speed_data = pd.DataFrame({
        "time_s": data["t"],
        "Horizontal airspeed (m/s)": data["airspeed_mps"],
        "Target speed (m/s)": data["ref_speed"],
    })
    report.line_chart(st,
        speed_data,
        x="time_s",
        y=["Horizontal airspeed (m/s)", "Target speed (m/s)"],
        x_label="Time (s)",
        y_label="Horizontal airspeed / target speed (m/s)",
        alt="Horizontal airspeed compared with the mission speed target",
    )

with mission_col:
    report.text("markdown", st, "**Mission route and current position**")
    if flight_plan == "Waypoint route":
        route_frame = pd.DataFrame(
            waypoint_rows,
            columns=["North (m)", "East (m)", "Altitude (m)", "Speed (m/s)", "Tilt (deg)"],
        )
        report.dataframe(st, route_frame, hide_index=True, alt="Configured waypoint route")
        report.text("caption", st, "The current guidance model tracks altitude and forward speed; lateral waypoint tracking is not yet modeled.")
        route_plot = pd.DataFrame({
            "North (m)": data["north"],
            "East (m)": data["east"],
            "Time (s)": data["t"],
        })
        report.scatter_chart(st,
            route_plot,
            x="North (m)",
            y="East (m)",
            color="Time (s)",
            x_label="North position (m)",
            y_label="East position (m)",
            alt="Aircraft position through the local north-east flight path",
        )
    else:
        report.text("write", st,
            f"Mode: **{mission_mode}**  \n"
            f"Profile: **{flight_plan}**  \n"
            f"Tilt actuator limit: **{max_tilt_rate_deg_s:.1f} deg/s**"
        )

tilt_col, motor_col, surface_col = st.columns(3)
with tilt_col:
    report.text("markdown", st, "**Tilt actuator tracking**")
    tilt_data = pd.DataFrame({
        "time_s": data["t"],
        "Actual tilt (deg)": data["tilt"],
        "Tilt command (deg)": data["tilt_cmd"],
    })
    report.line_chart(st,
        tilt_data,
        x="time_s",
        y=["Actual tilt (deg)", "Tilt command (deg)"],
        x_label="Time (s)",
        y_label="Nacelle tilt (deg)",
        alt="Actual nacelle tilt tracking its commanded angle",
    )

with motor_col:
    report.text("markdown", st, "**Per-motor command (RPM)**")
    rad_s_to_rpm = 60.0 / (2.0 * np.pi)
    motor_data = pd.DataFrame({
        "time_s": data["t"],
        "Front left (RPM)": data["omega_fl"] * rad_s_to_rpm,
        "Front right (RPM)": data["omega_fr"] * rad_s_to_rpm,
        "Rear left (RPM)": data["omega_rl"] * rad_s_to_rpm,
        "Rear right (RPM)": data["omega_rr"] * rad_s_to_rpm,
    })
    report.line_chart(st,
        motor_data,
        x="time_s",
        y=[
            "Front left (RPM)",
            "Front right (RPM)",
            "Rear left (RPM)",
            "Rear right (RPM)",
        ],
        x_label="Time (s)",
        y_label="Motor speed (RPM)",
        alt="Commanded RPM for each of the four motors",
    )

with surface_col:
    report.text("markdown", st, "**V-tail command (deg)**")
    surface_data = pd.DataFrame({
        "time_s": data["t"],
        "Left surface": np.rad2deg(data["delta_left"]),
        "Right surface": np.rad2deg(data["delta_right"]),
    })
    report.line_chart(st,
        surface_data,
        x="time_s",
        y=["Left surface", "Right surface"],
        x_label="Time (s)",
        y_label="V-tail deflection (deg)",
        alt="Left and right V-tail control surface commands",
    )

if np.any(transition_mask):
    fig_rpm_tilt, ax_rpm_tilt = plt.subplots(figsize=(8, 4))
    for channel, label in (
        ("omega_fl", "Front left"),
        ("omega_fr", "Front right"),
        ("omega_rl", "Rear left"),
        ("omega_rr", "Rear right"),
    ):
        ax_rpm_tilt.plot(
            data["tilt"][transition_mask],
            data[channel][transition_mask] * rad_s_to_rpm,
            label=label,
            linewidth=1,
        )
    ax_rpm_tilt.set_xlabel("Nacelle tilt of tilting rotors (deg)")
    ax_rpm_tilt.set_ylabel("Commanded motor speed (RPM)")
    ax_rpm_tilt.grid(True)
    ax_rpm_tilt.legend()
    report.pyplot(st, fig_rpm_tilt, alt="Commanded motor speeds against nacelle tilt")
    report.text(
        "caption", st,
        "These are commanded speeds, not measured motor responses. Each pair can "
        "differ when the allocator requests moments.",
    )
    

# Instantiate for the plots
props_plot = MassProperties(cfg_path)
props_plot.mass_fuselage = mass_fuselage
props_plot.mass_wing = mass_wing
props_plot.mass_vtail = mass_vtail
props_plot.mass_motor = mass_motor
props_plot.total_mass = mass

# ==========================================================
# WIND TRIANGLE VISUALIZATION
# ==========================================================
report.text("markdown", st, "---")
report.text("subheader", st, " Wind Triangle (Live)")

c1, c2 = st.columns(2)

with c1:
    # Wind Triangle vector plot
    fig_wt, ax_wt = plt.subplots(figsize=(8, 6))
    
    # Get the most recent values
    u_airspeed = data["u_body"][-1]
    w_airspeed = data["w_body"][-1]
    
    # Draw the airspeed vector in body frame
    ax_wt.arrow(0, 0, u_airspeed, -w_airspeed, 
                head_width=0.3, head_length=0.5, fc='blue', ec='blue', 
                linewidth=3, label=f"True Airspeed V_T = {data['V_T'][-1]:.2f} m/s")
    
    ax_wt.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax_wt.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax_wt.set_xlabel("Forward velocity u (m/s)")
    ax_wt.set_ylabel("Vertical component −w (m/s; up positive)")
    ax_wt.set_title(f"Airspeed Vector in Body Frame\n(α = {np.rad2deg(np.arctan2(w_airspeed, u_airspeed)):.1f}°)")
    ax_wt.grid(True)
    ax_wt.legend()
    ax_wt.set_aspect('equal', adjustable='datalim')
    report.pyplot(st, fig_wt, alt="Body-frame airspeed vector showing forward and vertical airspeed")

with c2:
    # Angle of Attack (alpha) over time
    alpha_history = np.rad2deg(np.arctan2(np.array(data["w_body"]), np.array(data["u_body"])))
    
    fig_alpha, ax_alpha = plt.subplots(figsize=(8, 6))
    ax_alpha.plot(data["t"], alpha_history, 'b-', linewidth=2)
    ax_alpha.axhline(15, color='r', linestyle='--', alpha=0.5, label="Stall angle (15°)")
    ax_alpha.axhline(-15, color='r', linestyle='--', alpha=0.5)
    ax_alpha.axhspan(5.0, 12.0, color='green', alpha=0.15, label="Requested target")
    ax_alpha.set_xlabel("Mission time (s)")
    ax_alpha.set_ylabel("Angle of Attack α (deg)")
    ax_alpha.set_title("Angle of Attack during Flight")
    ax_alpha.grid(True)
    ax_alpha.legend()
    report.pyplot(st, fig_alpha, alt="Angle of attack through the flight with stall-angle references")
    report.text("caption", st, "The 5–12° target is assessed only at airspeed ≥15 m/s.")

# Show the instantaneous values
report.text("markdown", st, "**Instantaneous Wind Triangle**")
c1, c2, c3, c4 = st.columns(4)
report.metric(c1, "True Airspeed", f"{data['V_T'][-1]:.2f} m/s")
report.metric(c2, "u (Forward Airspeed)", f"{data['u_body'][-1]:.2f} m/s")
report.metric(c3, "w (Downward Airspeed)", f"{data['w_body'][-1]:.2f} m/s")
alpha_now = np.rad2deg(np.arctan2(data['w_body'][-1], data['u_body'][-1])) if data['u_body'][-1] != 0 else 0
report.metric(c4, "Angle of Attack α", f"{alpha_now:.1f}°")

# GEOMETRY
report.text("markdown", st, "---")
report.text("subheader", st, " Aircraft Geometry & Mass Properties")
col_geo1, col_geo2 = st.columns(2)

# 1. Instantiate MassProperties
props = MassProperties(cfg_path)
# 2. Override with the live dashboard variables from the Aero tab
# Calculate wingspan from the XFLR5 coordinates
calculated_wingspan = tip_y * 2.0
props.mass_fuselage = mass_fuselage
props.mass_wing = mass_wing
props.mass_vtail = mass_vtail
props.mass_motor = mass_motor
props.wingspan = calculated_wingspan
props.root_chord = root_c
props.tip_chord = tip_c
props.sweep_deg = sweep
props.le_root_x = root_x
props.motor_positions = motor_positions
props.tilt_pivot_xz = (tilt_pivot_x, tilt_pivot_z)

props.vtail_area = vtail_area
props.vtail_arm = vtail_arm
props.mac, props.ac_wing_x = props._calculate_mac_and_ac()
props.neutral_point_x = props._calculate_neutral_point()
props.total_mass = mass
props.cg = np.array([0.0, 0.0, 0.0])

with col_geo1:
    fig_side, ax_side = plt.subplots(figsize=(8, 4))
    ax_side.axhline(0, color='brown', linestyle='--', linewidth=1, alpha=0.5)
    for pos in motor_positions:
        ax_side.plot(pos[0], pos[2], 'ko', markersize=10)
    
    # Plot CG, Wing AC, and Neutral Point
    ax_side.plot(props.cg[0], props.cg[2], 'bo', markersize=12, label="CG")
    ax_side.plot(props.ac_wing_x, props.ac_z, 'go', markersize=10, label="Wing AC")
    ax_side.plot(props.neutral_point_x, props.ac_z, 'r', marker='X', markersize=12, label="Neutral Point")
    
    # Plot Wing and Tail
    ax_side.plot([props.le_root_x, props.le_root_x + props.root_chord], [0, 0], 'k-', linewidth=4, label="Wing (Root)")
    ax_side.plot(
        [-props.cg[0] - props.vtail_arm, -props.cg[0] - props.vtail_arm],
        [0, 0.2],
        'purple',
        linewidth=4,
        label="V-Tail",
    )
    
    ax_side.set_xlim(-1.0, 1.0); ax_side.set_ylim(-0.5, 0.5)
    ax_side.invert_yaxis()
    ax_side.set_xlabel("Longitudinal position X (m; forward positive)")
    ax_side.set_ylabel("Vertical position Z (m; down positive)")
    ax_side.set_title("Side Profile: Neutral Point & AC")
    ax_side.grid(True, alpha=0.3)
    ax_side.legend()
    report.pyplot(st, fig_side, alt="Aircraft side profile showing the center of gravity, aerodynamic center, and neutral point")

with col_geo2:
    fig_top, ax_top = plt.subplots(figsize=(8, 4))
    for pos in motor_positions:
        color = 'tab:blue' if pos[3] else 'gray'
        ax_top.plot(pos[0], pos[1], 'o', color=color, markersize=12)
    ax_top.plot([props.ac_wing_x, props.ac_wing_x], [-calculated_wingspan/2, calculated_wingspan/2], 'k-', linewidth=4, label="Wing")
    ax_top.plot(props.cg[0], props.cg[1], 'bo', markersize=12, label="CG")
    ax_top.set_xlim(-1.0, 1.0)
    ax_top.set_ylim(-calculated_wingspan, calculated_wingspan)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Longitudinal position X (m; forward positive)")
    ax_top.set_ylabel("Lateral position Y (m; right positive)")
    ax_top.set_title("Top Profile (X-Y)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()
    report.pyplot(st, fig_top, alt="Aircraft top profile showing wing span, center of gravity, and motor positions")
    # ==========================================================
# LIFT HANDOFF & CONTROL AUTHORITY
# ==========================================================
report.text("markdown", st, "---")
report.text("subheader", st, "Lift Handoff & Control Authority")
report.text(
    "caption", st,
    " ",
)

c1, c2 = st.columns(2)

with c1:
    # Lift breakdown over time
    fig_lift, ax_lift = plt.subplots(figsize=(8, 5))
    ax_lift.stackplot(data["t"], data["rotor_lift"], data["wing_lift"], data["wash_lift"],
                     labels=["Rotor Vertical Thrust", "Wing Lift (Free Stream)", "Rotor Wash Lift"],
                     colors=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.8)
    ax_lift.plot(data["t"], [props.total_mass * 9.81] * len(data["t"]),
                 'k--', linewidth=2, label="Weight (mg)")
    ax_lift.set_xlabel("Time (s)")
    ax_lift.set_ylabel("Vertical Force (N)")
    ax_lift.set_title("Lift Sources During Transition")
    ax_lift.grid(True, alpha=0.3)
    ax_lift.legend(loc='upper right')
    report.pyplot(st, fig_lift, alt="Rotor, wing, and rotor-wash contributions to vertical lift through transition")

with c2:
    # Handoff ratio
    fig_ratio, ax_ratio = plt.subplots(figsize=(8, 5))
    ax_ratio.plot(data["t"], data["handoff_ratio"], 'b-', linewidth=2)
    ax_ratio.axhline(0, color='r', linestyle='--', alpha=0.5, label="100% Rotor")
    ax_ratio.axhline(1, color='g', linestyle='--', alpha=0.5, label="100% Wing")
    ax_ratio.fill_between(data["t"], 0, data["handoff_ratio"], alpha=0.2, color='blue')
    ax_ratio.set_xlabel("Time (s)")
    ax_ratio.set_ylabel("Handoff Ratio (0=rotor, 1=wing)")
    ax_ratio.set_title("Lift Handoff Ratio")
    ax_ratio.set_ylim(-0.1, 1.1)
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend()
    report.pyplot(st, fig_ratio, alt="Lift handoff ratio from rotor-dominated to wing-dominated support")

# Control authority at current instant
report.text("markdown", st, "### Simplified Control Authority Estimate")
report.text("caption", st, "This pitch-authority blend is a heuristic; the model-based control-effectiveness map is in Phase 5 below.")
current_tilt = data["tilt"][-1]
current_V_T = data["V_T"][-1]
if current_V_T < 0.5:
    current_V_T = 0.5

from src.aero.handoff import control_authority, control_blend
auth = control_authority(np.deg2rad(current_tilt), current_V_T, 12.0)
blend = control_blend(np.deg2rad(current_tilt))

c1, c2, c3 = st.columns(3)
report.metric(c1, "Rotor Pitch Authority", f"{auth['rotor_authority_frac']*100:.0f}%")
report.metric(c2, "V-Tail Pitch Authority", f"{auth['vtail_authority_frac']*100:.0f}%")
report.metric(c3, "Controller Blend", f"{blend*100:.0f}% (0=rotor, 100=wing)")

# Control authority vs tilt angle (the "handoff map")
report.text("markdown", st, "### Heuristic Pitch Authority Trend")
report.text("caption", st, "Illustrative tilt trend evaluated at a fixed 12 m/s; it is not the finite-difference actuator map.")
tilt_scan = np.linspace(0, 90, 50)
rotor_fracs = []
vtail_fracs = []
blends = []

for t_deg in tilt_scan:
    t_rad = np.deg2rad(t_deg)
    auth_scan = control_authority(t_rad, 12.0, 12.0)
    rotor_fracs.append(auth_scan['rotor_authority_frac'] * 100)
    vtail_fracs.append(auth_scan['vtail_authority_frac'] * 100)
    blends.append(control_blend(t_rad) * 100)

fig_map, ax_map = plt.subplots(figsize=(12, 5))
ax_map.plot(tilt_scan, rotor_fracs, 'r-', linewidth=3, label="Rotor Pitch Authority (%)")
ax_map.plot(tilt_scan, vtail_fracs, 'g-', linewidth=3, label="V-Tail Pitch Authority (%)")
ax_map.plot(tilt_scan, blends, 'b--', linewidth=2, label="Control Blend Law (%)")
ax_map.axvline(current_tilt, color='purple', linestyle=':', linewidth=2, label=f"Current Tilt: {current_tilt:.0f}°")
ax_map.set_xlabel("Rotor Tilt Angle (deg)")
ax_map.set_ylabel("Authority / Blend (%)")
ax_map.set_title("Control Authority Handoff Map")
ax_map.grid(True, alpha=0.3)
ax_map.legend()
report.pyplot(st, fig_map, alt="Rotor and V-tail control authority and blend versus nacelle tilt")


# Calculate static margin percentage
sm_pct = props.get_static_margin_pct()

c1, c2, c3, c4 = st.columns(4)
report.metric(c1, "Total Mass", f"{props.total_mass:.2f} kg")
report.metric(c2, "Wing Loading", f"{props.total_mass / wing_area:.1f} kg/m²")
report.metric(c3, "Static Margin", f"{sm_pct:.1f} %")

# The neutral-point calculation is a geometry estimate relative to an assumed CG origin.
if sm_pct <= 0.0:
    c4.error("Non-positive estimate")
    report.text("error", st,
        "The simplified neutral-point estimate is non-positive relative to the assumed "
        "CG origin. This can indicate a longitudinal static-stability concern; verify the "
        "actual center of gravity and aerodynamic derivatives before drawing conclusions."
    )
elif sm_pct > 20.0:
    c4.warning("Above the dashboard range")
    report.text("warning", st,
        "The geometry-based static-margin estimate is above 20%. This simplified "
        "estimate alone does not establish that the aircraft is over-stable or unsafe."
    )
else:
    report.metric(c4, "Positive estimate", f"{sm_pct:.1f}%")
    report.text("info", st,
        "A positive geometry-based static-margin estimate does not demonstrate static "
        "or dynamic stability. It assumes the CG is at the model origin; use the "
        "transition derivative map and aircraft-specific test data for further analysis."
    )

report.text("markdown", st, "---")
report.text("subheader", st, "3x3 Inertia Tensor & Mass Properties")
report.text("markdown", st, "This matrix describes the aircraft's resistance to rotation. Cross-coupling terms ($I_{xz}$) appear when the mass is asymmetric.")

# Calculate the current Inertia Matrix at hover (0 deg tilt)
current_I = props.calculate_inertia(0.0)

# Convert to a displayable format
I_display = np.round(current_I, 4)

# Display as columns
c1, c2, c3 = st.columns(3)
report.metric(c1, "Ixx (Roll)", f"{I_display[0,0]:.4f} kg·m²")
report.metric(c2, "Iyy (Pitch)", f"{I_display[1,1]:.4f} kg·m²")
report.metric(c3, "Izz (Yaw)", f"{I_display[2,2]:.4f} kg·m²")

report.text("write", st, "Full 3x3 Matrix:")
report.text("write", st, I_display)

# Dynamic Inertia Visualization
report.text("markdown", st, "**Dynamic Inertia during Tilt Transition**")
tilt_scan = np.linspace(0, 90, 20)
ixx_scan = []
iyy_scan = []
izz_scan = []

for t_deg in tilt_scan:
    I_dyn = props.calculate_inertia(t_deg)
    ixx_scan.append(I_dyn[0,0])
    iyy_scan.append(I_dyn[1,1])
    izz_scan.append(I_dyn[2,2])

fig_I, ax_I = plt.subplots(figsize=(10, 4))
ax_I.plot(tilt_scan, ixx_scan, label="Ixx (Roll)")
ax_I.plot(tilt_scan, iyy_scan, label="Iyy (Pitch)")
ax_I.plot(tilt_scan, izz_scan, label="Izz (Yaw)")
ax_I.set_xlabel("Rotor Tilt Angle (deg)")
ax_I.set_ylabel("Inertia (kg·m²)")
ax_I.set_title("Inertia changes as motors tilt forward")
ax_I.grid(True)
ax_I.legend()
report.pyplot(st, fig_I, alt="Roll, pitch, and yaw inertia values as nacelles tilt")

# ==========================================================
# PHASE 5: TRANSITION DYNAMICS
# ==========================================================
report.text("markdown", st, "---")
report.text("subheader", st, "Phase 5: Transition dynamics — the bridge")
report.text("caption", st,
    " "
)

analysis_rotor = RotorModel(k_thrust=k_thrust, max_omega=max_omega)
analysis_vtail = VtailModel(area=vtail_area, arm_length=vtail_arm)
analysis_aero = AeroModel(
    wing_area=wing_area,
    vtail=analysis_vtail,
    prop_diameter_m=prop_dia * 0.0254,
)
analysis_dynamics = TiltrotorDynamics(
    mass=mass,
    rotor=analysis_rotor,
    mass_props=props,
    aero=analysis_aero,
)

render_transition_analyses(analysis_dynamics, max_tilt_rate_deg_s)

analysis_speed_col, analysis_tilt_col = st.columns(2)
analysis_speed = analysis_speed_col.slider(
    "Analysis true airspeed (m/s)",
    min_value=0.0,
    max_value=30.0,
    value=float(np.clip(max(data["V_T"][-1], 8.0), 0.0, 30.0)),
    step=0.5,
    key="transition_analysis_speed",
)
analysis_tilt_deg = analysis_tilt_col.slider(
    "Analysis nacelle tilt (deg)",
    min_value=0,
    max_value=90,
    value=45,
    step=1,
    key="transition_analysis_tilt",
)
analysis_tilt_rad = np.deg2rad(analysis_tilt_deg)
tail_limit_rad = 0.2
trim = solve_longitudinal_trim(
    analysis_dynamics,
    airspeed_mps=analysis_speed,
    tilt_rad=analysis_tilt_rad,
    tail_limit_rad=tail_limit_rad,
)

corridor_tab, effectiveness_tab, trim_tab, stability_tab = st.tabs(
    [
        "5.1 Transition corridor",
        "5.2 Control effectiveness",
        "5.3 Trim analysis",
        "5.4 Stability map",
    ]
)

with corridor_tab:
    max_rotor_thrust = analysis_rotor.compute_thrust(max_omega, height_above_ground=100.0)
    cl_max = analysis_aero.CL_0 + analysis_aero.CL_alpha * analysis_aero.stall_angle
    envelope = transition_speed_envelope(
        mass=mass,
        wing_area=wing_area,
        max_thrust_per_rotor=max_rotor_thrust,
        rotor_tilts=[bool(position[3]) for position in motor_positions],
        rho=analysis_aero.rho,
        cl_max=cl_max,
        cd_0=analysis_aero.CD_0,
        aspect_ratio=analysis_aero.AR,
    )
    transition_duration_s = st.slider(
        "Available transition time (s)",
        min_value=1.0,
        max_value=60.0,
        value=20.0,
        step=1.0,
        help="Used to check reachable nacelle tilt and average motor RPM change from hover.",
        key="corridor_transition_duration_s",
    )
    if gust_enabled and gust_model == "Finite-duration gust":
        gust_margin_mps = float(gust_mag)
    elif gust_enabled and gust_model == "Dryden turbulence":
        gust_margin_mps = float(
            3.0 * np.linalg.norm([dryden_sigma, dryden_sigma, 0.6 * dryden_sigma])
        )
    else:
        gust_margin_mps = 0.0

    profile_aircraft = active_profile.get("aircraft", {})
    profile_propulsion = (
        profile_aircraft.get("propulsion", {})
        if isinstance(profile_aircraft, Mapping)
        else {}
    )
    profile_battery = profile_propulsion.get("battery", {})
    profile_motor = profile_propulsion.get("motor", {})
    profile_propeller = profile_propulsion.get("propeller", {})
    profile_esc = profile_propulsion.get("esc", {})
    profile_motor_dynamics = profile_propulsion.get("motor_dynamics", {})
    electrical_parameters = {
        "motor_kv_rpm_per_volt": float(motor_kv),
        "motor_resistance_ohm": profile_motor.get("resistance_ohm"),
        "motor_no_load_current_a": profile_motor.get("no_load_current_a"),
        "motor_continuous_current_a": profile_motor.get("continuous_current_a"),
        "propeller_torque_coefficient_nm_s2_rad2": profile_propeller.get(
            "torque_coefficient_nm_s2_rad2"
        ),
        "battery_nominal_voltage_v": float(battery_v_nom),
        "battery_internal_resistance_ohm": profile_battery.get(
            "internal_resistance_ohm"
        ),
        "battery_capacity_ah": float(battery_mah) / 1000.0,
        "battery_c_rating": float(battery_c_rating),
        "battery_continuous_current_a": profile_battery.get(
            "continuous_current_a"
        ),
        "esc_efficiency": profile_esc.get("efficiency"),
        "esc_continuous_current_a": profile_esc.get("continuous_current_a"),
        "motor_slew_rate_rpm_s": profile_motor_dynamics.get("slew_rate_rpm_s"),
    }
    corridor_signature = hashlib.sha256(
        repr((
            "independent_pair_trim_v1",
            dashboard_values_fingerprint,
            gust_margin_mps,
            max_tilt_rate_deg_s,
            transition_duration_s,
            tuple(sorted(electrical_parameters.items())),
        )).encode("utf-8")
    ).hexdigest()
    if generate_report or st.button(
        "Compute constrained corridor",
        key="compute_constrained_transition_corridor",
    ):
        with st.spinner(
            "Evaluating trim, pitch moment, static stability, gust reserve, and actuator limits..."
        ):
            corridor_result = constrained_transition_corridor(
                analysis_dynamics,
                max_thrust_per_rotor=max_rotor_thrust,
                rotor_tilts=[bool(position[3]) for position in motor_positions],
                airspeeds_mps=np.arange(0.0, 31.0, 1.0),
                tilt_angles_deg=np.linspace(0.0, 90.0, 7),
                max_tilt_rate_deg_s=float(max_tilt_rate_deg_s),
                transition_duration_s=float(transition_duration_s),
                gust_speed_margin_mps=gust_margin_mps,
                electrical_parameters=electrical_parameters,
                tail_limit_rad=tail_limit_rad,
                cl_max=cl_max,
                cd_0=analysis_aero.CD_0,
                aspect_ratio=analysis_aero.AR,
            )
        st.session_state["constrained_transition_corridor_result"] = {
            "signature": corridor_signature,
            "result": corridor_result,
        }

    corridor_record = st.session_state.get("constrained_transition_corridor_result")
    current_corridor_result = None
    if isinstance(corridor_record, dict):
        if corridor_record.get("signature") == corridor_signature:
            current_corridor_result = corridor_record["result"]
        else:
            report.text("info", st,
                "Corridor inputs changed. Recompute the constrained corridor to refresh its filters."
            )

    if current_corridor_result is None:
        report.text("info", st,
            "."
        )
    elif not np.any(current_corridor_result["known_constraints_mask"]):
        report.text("warning", st,
            "No grid cells pass the known corridor filters. Gray crosses, if shown, "
            "pass longitudinal trim only; other constraints have failed or remain "
            "unassessed. They are not a safe corridor."
        )

    feasible = envelope["feasible"]
    fig_corridor = go.Figure()
    if np.any(feasible):
        corridor_tilt = envelope["tilt_deg"][feasible]
        minimum_speed = envelope["minimum_speed_mps"][feasible]
        maximum_speed = envelope["maximum_speed_mps"][feasible]
        fig_corridor.add_trace(go.Scatter(
            x=np.concatenate([maximum_speed, minimum_speed[::-1]]),
            y=np.concatenate([corridor_tilt, corridor_tilt[::-1]]),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.18)",
            line=dict(color="rgba(52, 152, 219, 0.25)"),
            name="Quasi-steady feasible envelope",
            hoverinfo="skip",
        ))
        fig_corridor.add_trace(go.Scatter(
            x=minimum_speed,
            y=corridor_tilt,
            mode="lines",
            line=dict(color="royalblue", dash="dash"),
            name="Minimum speed",
        ))
        fig_corridor.add_trace(go.Scatter(
            x=maximum_speed,
            y=corridor_tilt,
            mode="lines",
            line=dict(color="royalblue"),
            name="Maximum speed",
        ))
    if np.any(transition_mask):
        fig_corridor.add_trace(go.Scatter(
            x=data["V_T"][transition_mask],
            y=data["tilt"][transition_mask],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="darkorange", width=2),
            name="Recorded transition",
        ))
    if current_corridor_result is not None:
        known_mask = np.asarray(
            current_corridor_result["known_constraints_mask"],
            dtype=bool,
        )
        qualified_mask = np.asarray(
            current_corridor_result["qualified_mask"],
            dtype=bool,
        )
        corridor_speeds = np.asarray(
            current_corridor_result["airspeeds_mps"],
            dtype=float,
        )
        corridor_tilts = np.asarray(
            current_corridor_result["tilt_angles_deg"],
            dtype=float,
        )
        trim_only_mask = (
            np.asarray(current_corridor_result["analytic_mask"], dtype=bool)
            & np.asarray(current_corridor_result["trim_mask"], dtype=bool)
            & np.asarray(current_corridor_result["rotor_thrust_mask"], dtype=bool)
            & ~known_mask
        )
        trim_speed_indices, trim_tilt_indices = np.where(trim_only_mask)
        if trim_speed_indices.size:
            trim_stability = current_corridor_result["stability_map"]
            pair_rpm = np.column_stack((
                np.asarray(trim_stability["trim_front_motor_rpm"], dtype=float)[trim_only_mask],
                np.asarray(trim_stability["trim_rear_motor_rpm"], dtype=float)[trim_only_mask],
            ))
            fig_corridor.add_trace(go.Scatter(
                x=corridor_speeds[trim_speed_indices],
                y=corridor_tilts[trim_tilt_indices],
                customdata=pair_rpm,
                mode="markers",
                marker=dict(color="gray", size=8, symbol="x"),
                name="Trim only; not qualified",
                hovertemplate=(
                    "Airspeed: %{x:.1f} m/s<br>Tilt: %{y:.0f}°"
                    "<br>Front pair: %{customdata[0]:.0f} RPM"
                    "<br>Rear pair: %{customdata[1]:.0f} RPM"
                    "<br>Other constraints failed or unassessed<extra></extra>"
                ),
            ))
        known_speed_indices, known_tilt_indices = np.where(
            known_mask & ~qualified_mask
        )
        if known_speed_indices.size:
            fig_corridor.add_trace(go.Scatter(
                x=corridor_speeds[known_speed_indices],
                y=corridor_tilts[known_tilt_indices],
                mode="markers",
                marker=dict(color="purple", size=6, symbol="circle-open"),
                name="Known filters pass; incomplete",
                hovertemplate=(
                    "Airspeed: %{x:.1f} m/s<br>Tilt: %{y:.0f}°"
                    "<br>Cl_beta, electrical or motor-slew assessment may be unresolved"
                    "<extra></extra>"
                ),
            ))
        qualified_speed_indices, qualified_tilt_indices = np.where(qualified_mask)
        if qualified_speed_indices.size:
            fig_corridor.add_trace(go.Scatter(
                x=corridor_speeds[qualified_speed_indices],
                y=corridor_tilts[qualified_tilt_indices],
                mode="markers",
                marker=dict(color="green", size=7, symbol="diamond"),
                name="All configured filters pass",
                hovertemplate="Airspeed: %{x:.1f} m/s<br>Tilt: %{y:.0f}°<extra></extra>",
            ))
    fig_corridor.update_layout(
        xaxis_title="True airspeed (m/s)",
        yaxis_title="Nacelle tilt (deg)",
        yaxis_range=[0, 90],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=30, b=20),
        height=440,
    )
    report.plotly_chart(st,
        fig_corridor,
        width="stretch",
        alt="Transition corridor with airspeed on the horizontal axis and nacelle tilt on the vertical axis",
    )
    report.text("caption", st,
        " "
    )
    report.text("caption", st,
        " "
    )
    report.text("markdown", st,
        " "
    )
    if current_corridor_result is not None:
        report.text("markdown", st, "**Recorded transition histories**")
        if np.any(transition_mask):
            transition_time = data["t"][transition_mask] - data["t"][transition_mask][0]
            transition_alpha = np.rad2deg(np.arctan2(
                data["w_body"][transition_mask], data["u_body"][transition_mask]
            ))
            alpha_assessed = (
                (data["V_T"][transition_mask] >= 15.0)
                & (np.hypot(data["u_body"][transition_mask], data["w_body"][transition_mask]) > 1e-6)
            )
            histories = make_subplots(
                rows=2, cols=3,
                subplot_titles=("True airspeed", "Pitch angle", "Nacelle tilt",
                                "Angle of attack", "Commanded motor speeds", "V-tail commands"),
                vertical_spacing=0.16,
            )
            for name, values, row, col, color in (
                ("True airspeed", data["V_T"][transition_mask], 1, 1, "royalblue"),
                ("Pitch", data["pitch"][transition_mask], 1, 2, "firebrick"),
                ("Actual tilt", data["tilt"][transition_mask], 1, 3, "darkorange"),
                ("Mission tilt request", data["tilt_cmd"][transition_mask], 1, 3, "gray"),
                ("Front left", data["omega_fl"][transition_mask] * rad_s_to_rpm, 2, 2, "royalblue"),
                ("Front right", data["omega_fr"][transition_mask] * rad_s_to_rpm, 2, 2, "deepskyblue"),
                ("Rear left", data["omega_rl"][transition_mask] * rad_s_to_rpm, 2, 2, "firebrick"),
                ("Rear right", data["omega_rr"][transition_mask] * rad_s_to_rpm, 2, 2, "darkorange"),
                ("Left V-tail", np.rad2deg(data["delta_left"][transition_mask]), 2, 3, "royalblue"),
                ("Right V-tail", np.rad2deg(data["delta_right"][transition_mask]), 2, 3, "firebrick"),
            ):
                histories.add_trace(go.Scatter(
                    x=transition_time, y=values, mode="lines", name=name,
                    line=dict(color=color, dash="dash" if name == "Mission tilt request" else "solid"),
                ), row=row, col=col)
            for name, mask, color in (
                ("AoA: target assessed", alpha_assessed, "royalblue"),
                ("AoA: target unassessed", ~alpha_assessed, "gray"),
            ):
                histories.add_trace(go.Scatter(
                    x=transition_time, y=np.where(mask, transition_alpha, np.nan),
                    mode="lines", name=name, line=dict(color=color),
                    connectgaps=False,
                ), row=2, col=1)
            histories.add_hrect(
                y0=5.0, y1=12.0, fillcolor="green", opacity=0.15,
                line_width=0, row=2, col=1,
            )
            for row, col, title in (
                (1, 1, "Airspeed (m/s)"), (1, 2, "Pitch (deg)"),
                (1, 3, "Tilt (deg)"), (2, 1, "Angle of attack (deg)"),
                (2, 2, "Commanded RPM"), (2, 3, "Deflection command (deg)"),
            ):
                histories.update_xaxes(title_text="Transition time (s)", row=row, col=col)
                histories.update_yaxes(title_text=title, row=row, col=col)
            histories.update_layout(height=720, legend=dict(orientation="h"), margin=dict(t=60, b=70))
            report.plotly_chart(
                st, histories, width="stretch",
                alt="Recorded transition histories for airspeed, pitch, tilt, angle of attack, motor commands and V-tail commands",
            )
            report.text("caption", st,
                "The green band is the requested 5–12° angle-of-attack target, assessed "
                "only at true airspeed ≥15 m/s. Gray segments are unassessed, not clipped "
                "or corrected; large angles can indicate reversed or low-speed flow. "
                "Motor speeds and V-tail deflections are commands, not measured responses. "
                "These histories stop at the final recorded sample and are not an optimized trajectory."
            )
            if not np.any(alpha_assessed):
                report.text("warning", st,
                    "The angle-of-attack target is unassessed: this transition has no "
                    "samples at or above 15 m/s. The displayed angles have not been "
                    "forced into the target range."
                )
            elif np.any((transition_alpha[alpha_assessed] < 5.0) | (transition_alpha[alpha_assessed] > 12.0)):
                report.text("warning", st,
                    "The recorded angle of attack is outside the 5–12° target at one or "
                    "more assessed samples. This plot does not correct the simulation."
                )
        else:
            report.text("info", st, "No transition-phase samples were recorded for the histories.")
        analytic_count = int(np.count_nonzero(current_corridor_result["analytic_mask"]))
        known_count = int(np.count_nonzero(current_corridor_result["known_constraints_mask"]))
        qualified_count = int(np.count_nonzero(current_corridor_result["qualified_mask"]))
        analytic_mask = np.asarray(current_corridor_result["analytic_mask"], dtype=bool)
        electrical_assessed_count = int(
            np.count_nonzero(
                analytic_mask
                & np.asarray(current_corridor_result["electrical_assessed_mask"], dtype=bool)
            )
        )
        electrical_pass_count = int(
            np.count_nonzero(
                analytic_mask
                & np.asarray(current_corridor_result["electrical_feasible_mask"], dtype=bool)
            )
        )
        slew_assessed_count = int(
            np.count_nonzero(
                analytic_mask
                & np.asarray(current_corridor_result["motor_slew_assessed_mask"], dtype=bool)
            )
        )
        slew_pass_count = int(
            np.count_nonzero(
                analytic_mask
                & np.asarray(current_corridor_result["motor_slew_feasible_mask"], dtype=bool)
            )
        )
        status_columns = st.columns(5)
        report.metric(status_columns[0], "Analytical grid cells", str(analytic_count))
        report.metric(status_columns[1], "Known constraints pass", str(known_count))
        report.metric(status_columns[2], "Electrical cells assessed", str(electrical_assessed_count))
        report.metric(status_columns[3], "Motor slew cells assessed", str(slew_assessed_count))
        report.metric(status_columns[4], "All configured filters pass", str(qualified_count))
        point_statuses = np.asarray(
            current_corridor_result["point_screen_status"],
            dtype=object,
        )
        point_status_counts = {
            status: int(np.count_nonzero(point_statuses == status))
            for status in ("pass", "fail", "incomplete")
        }
        report.text("caption", st,
            "Steady point screen: "
            f"{point_status_counts['pass']} pass, "
            f"{point_status_counts['fail']} fail, "
            f"{point_status_counts['incomplete']} incomplete. "
            "Path-level tilt-rate and motor-slew constraints are reported separately."
        )
        trim_mask = np.asarray(current_corridor_result["trim_mask"], dtype=bool)
        pitch_mask = np.asarray(current_corridor_result["pitch_moment_mask"], dtype=bool)
        stability_mask = np.asarray(
            current_corridor_result["static_stability_mask"],
            dtype=bool,
        )
        stability_assessed = np.asarray(
            current_corridor_result["static_stability_assessed_mask"],
            dtype=bool,
        )
        known_stability_mask = np.asarray(
            current_corridor_result["known_static_stability_mask"],
            dtype=bool,
        )
        gust_mask = np.asarray(current_corridor_result["gust_mask"], dtype=bool)
        actuator_mask = np.asarray(
            current_corridor_result["actuator_rate_mask"],
            dtype=bool,
        )
        constraint_rows = [
            ("Analytical envelope", analytic_count, analytic_count),
            (
                "Longitudinal trim / force balance",
                int(np.count_nonzero(analytic_mask & trim_mask)),
                analytic_count,
            ),
            (
                "Pitch-moment residual",
                int(np.count_nonzero(analytic_mask & pitch_mask)),
                analytic_count,
            ),
            (
                "Modeled pitch/yaw derivative signs",
                int(np.count_nonzero(analytic_mask & known_stability_mask)),
                int(np.count_nonzero(
                    analytic_mask
                    & current_corridor_result["stability_map"]["assessment_mask"]
                )),
            ),
            (
                "Full static derivative signs (including Cl_beta)",
                int(np.count_nonzero(analytic_mask & stability_mask)),
                int(np.count_nonzero(analytic_mask & stability_assessed)),
            ),
            (
                "Configured gust airspeed reserve",
                int(np.count_nonzero(analytic_mask & gust_mask)),
                analytic_count,
            ),
            (
                "Nacelle-rate reachable tilt",
                int(np.count_nonzero(analytic_mask & actuator_mask)),
                analytic_count,
            ),
            ("Electrical current / voltage", electrical_pass_count, electrical_assessed_count),
            ("Motor RPM slew", slew_pass_count, slew_assessed_count),
            ("All configured constraints", qualified_count, analytic_count),
        ]
        report.dataframe(st,
            pd.DataFrame(
                constraint_rows,
                columns=["Constraint", "Passing cells", "Assessed cells"],
            ),
            hide_index=True,
            width="stretch",
        )
        missing_electrical = tuple(
            current_corridor_result["electrical_missing_parameters"]
        )
        cl_beta_assessed = np.asarray(
            current_corridor_result["stability_map"]["cl_beta_assessed"],
            dtype=bool,
        )
        if not np.any(analytic_mask & cl_beta_assessed):
            report.text("warning", st,
                "Lateral roll stability (Cl_beta) is unassessed because the current "
                "aerodynamic model has no sideslip-to-roll moment response. This does "
                "not prove instability; full corridor qualification remains blocked."
            )
        if missing_electrical:
            report.text("warning", st,
                "Electrical feasibility is not assessed; fill these profile fields with "
                "datasheet or bench values: "
                + ", ".join(missing_electrical)
                + "."
            )
        if slew_assessed_count == 0:
            report.text("warning", st,
                "Motor RPM slew is not assessed: set propulsion.motor_dynamics.slew_rate_rpm_s "
                "in the uploaded profile. A time constant is not treated as an equivalent hard slew limit."
            )
        if qualified_count == 0:
            report.text("warning", st,
                "No grid cells currently pass every configured filter. Unassessed electrical "
                "or motor-slew limits are intentionally not counted as feasible."
            )
        report.text("caption", st,
            f"Configured gust airspeed reserve: {gust_margin_mps:.2f} m/s; "
            f"available transition time: {transition_duration_s:.1f} s; "
            f"maximum tilt rate: {max_tilt_rate_deg_s:.1f} deg/s. "
            "Motor RPM slew uses the largest pair's average change from hover trim over the full duration, "
            "not a time-domain actuator simulation. The electrical estimate assumes "
            "ESC current is motor-side current and uses continuous-current limits."
        )
        report.text("warning", st,
            "This is a simulation-only screening corridor, not a certified or flight-ready "
            "safe envelope. Static derivative signs are not dynamic-stability proof, and "
            "the aerodynamic model remains simplified."
        )
    if not np.any(feasible):
        report.text("warning", st, "No quasi-steady corridor is feasible with the current mass, wing, and propulsion settings.")

with effectiveness_tab:
    effectiveness_reference = "trim solution" if trim["feasible"] else "best-fit point"
    rotor_labels, moment_effectiveness = control_effectiveness_map(
        analysis_dynamics,
        airspeed_mps=analysis_speed,
        tilt_rad=analysis_tilt_rad,
        alpha_rad=np.deg2rad(float(trim["alpha_deg"])),
        collective_omega=float(trim["motor_rpm"]) * 2.0 * np.pi / 60.0,
        pair_omegas=(
            float(trim["front_motor_rpm"]) * 2.0 * np.pi / 60.0,
            float(trim["rear_motor_rpm"]) * 2.0 * np.pi / 60.0,
        ),
        tail_deflection_rad=np.deg2rad(float(trim["tail_deflection_deg"])),
        tail_limit_rad=tail_limit_rad,
    )
    fig_effectiveness = go.Figure(go.Heatmap(
        z=moment_effectiveness,
        x=rotor_labels,
        y=["Roll L", "Pitch M", "Yaw N"],
        colorscale="RdBu",
        zmid=0.0,
        colorbar=dict(title="Moment change (N·m)"),
        hovertemplate="%{y}<br>%{x}<br>%{z:.4f} N·m<extra></extra>",
    ))
    fig_effectiveness.update_xaxes(
        tickmode="array",
        tickvals=rotor_labels,
        ticktext=rotor_labels,
        tickangle=-30,
        automargin=True,
    )
    fig_effectiveness.update_layout(
        xaxis_title="Actuator perturbation pattern",
        yaxis_title="Body moment axis",
        margin=dict(l=20, r=20, t=30, b=130),
        height=440,
    )
    report.plotly_chart(st,
        fig_effectiveness,
        width="stretch",
        alt="Finite-difference mapping from differential motor RPM and V-tail deflections to roll, pitch, and yaw moments",
    )
    report.text("caption", st,
        f"Local moment changes are evaluated around the {effectiveness_reference} "
        f"({analysis_speed:.1f} m/s, {analysis_tilt_deg}°). Each rotor channel applies "
        "a ±50 RPM differential pattern; each V-tail channel applies a ±5° perturbation. "
        "Perturbations are clipped at actuator bounds."
    )

with trim_tab:
    trim_columns = st.columns(5)
    report.metric(trim_columns[0], "Longitudinal trim", "Feasible" if trim["feasible"] else "Not feasible")
    report.metric(trim_columns[1], "Angle of attack", f"{float(trim['alpha_deg']):.2f}°")
    report.metric(trim_columns[2], "Front pair motor speed", f"{float(trim['front_motor_rpm']):.0f} RPM")
    report.metric(trim_columns[3], "Rear pair motor speed", f"{float(trim['rear_motor_rpm']):.0f} RPM")
    report.metric(trim_columns[4], "Symmetric V-tail", f"{float(trim['tail_deflection_deg']):.2f}°")
    trim_acceleration = np.asarray(trim["acceleration_ned"], dtype=float)
    trim_moment = np.asarray(trim["moment_body"], dtype=float)
    report.text("caption", st,
        f"Steady, level, zero-wind trim at {analysis_speed:.1f} m/s and {analysis_tilt_deg}° tilt; "
        "each front/rear pair shares a speed; "
        f"bounds: α ±{np.rad2deg(analysis_aero.stall_angle):.1f}°, "
        f"motor 0–{max_rpm:.0f} RPM, V-tail ±{np.rad2deg(tail_limit_rad):.1f}°. "
        f"Residuals: aN={trim_acceleration[0]:.3f} m/s², "
        f"aD={trim_acceleration[2]:.3f} m/s², "
        f"My={trim_moment[1]:.3f} N·m, "
        f"max normalized residual={float(trim['max_normalized_residual']):.4f}."
    )
    if trim["feasible"]:
        report.text("success", st, "Force and pitch-moment residuals are within the configured trim tolerance.")
    else:
        report.text("warning", st,
            "No feasible trim was found within the angle-of-attack, motor, and V-tail limits. "
            f"Optimizer: {trim['solver_message']}"
        )

with stability_tab:
    report.text("caption", st,
        "The map finite-differences body-moment coefficients around the longitudinal "
        "trim solution (or the optimizer's best-fit point) across airspeed and nacelle tilt."
    )
    report.text("info", st,
        "Static sign checks from the supplied reference: longitudinal C_mα < 0, "
        "lateral C_lβ < 0, and directional C_nβ > 0. These local derivatives do not "
        "establish dynamic stability."
    )
    report.text("warning", st,
        "Simulation-only model check: the aerodynamic model uses built-in simplified "
        "coefficients rather than measured aircraft data, and the uploaded profile's "
        "aerodynamic coefficient fields are not yet wired into that model. A failed "
        "longitudinal trim means that cell's derivatives are exploratory, not stability "
        "evidence. This analysis does not issue or change actuator commands."
    )
    report.text("caption", st,
        "The fixed grid uses 0–30 m/s in 5 m/s steps and 0–90° tilt in 15° steps. "
        "The calculation is on demand and is invalidated when the active aircraft settings change."
    )

    stability_signature = (
        "independent_pair_trim_v1",
        dashboard_values_fingerprint,
        float(analysis_aero.CL_alpha),
        float(analysis_aero.CL_0),
        float(analysis_aero.CD_0),
        float(analysis_aero.AR),
        float(analysis_aero.stall_angle),
        float(analysis_aero.Cm_0),
        float(analysis_aero.Cm_alpha),
        float(analysis_aero.chord),
        float(analysis_aero.x_ac),
        float(analysis_vtail.CL_alpha_tail),
        float(analysis_vtail.CD_0_tail),
        float(analysis_vtail.AR_tail),
        float(analysis_vtail.dihedral),
    )
    if generate_report or st.button("Compute stability map", key="compute_transition_stability_map"):
        with st.spinner("Solving trim and evaluating local derivatives across the transition grid..."):
            stability_result = transition_stability_map(
                analysis_dynamics,
                airspeeds_mps=np.arange(0.0, 31.0, 5.0),
                tilt_angles_rad=np.deg2rad(np.arange(0.0, 91.0, 15.0)),
                tail_limit_rad=tail_limit_rad,
            )
        st.session_state["transition_stability_map_result"] = {
            "signature": stability_signature,
            "result": stability_result,
        }

    map_record = st.session_state.get("transition_stability_map_result")
    if not isinstance(map_record, dict):
        report.text("info", st, "Compute the map to inspect trim quality, static derivatives, and actuator response.")
    elif map_record.get("signature") != stability_signature:
        report.text("info", st, "Aircraft settings changed. Recompute the map before using these results.")
    else:
        stability_result = map_record["result"]
        assessed = np.asarray(stability_result["assessment_mask"], dtype=bool)
        trim_feasible = np.asarray(stability_result["trim_feasible"], dtype=bool)
        assessed_count = int(np.count_nonzero(assessed))
        total_points = int(trim_feasible.size)
        stability_columns = st.columns(2)
        report.metric(stability_columns[0],
            "Longitudinal trim-feasible points",
            f"{int(np.count_nonzero(trim_feasible))}/{total_points}",
        )
        report.metric(stability_columns[1],
            "Static-derivative assessment points",
            f"{assessed_count}/{total_points}",
        )
        if assessed_count == 0:
            report.text("warning", st,
                "No grid point has both feasible longitudinal trim and nonzero modeled "
                "aerodynamic pressure. Static stability is therefore not established "
                "for this grid."
            )

        derivative_specs = (
            ("Longitudinal C_mα (1/rad)", "cm_alpha_per_rad", -1.0),
            ("Lateral C_lβ (1/rad)", "cl_beta_per_rad", -1.0),
            ("Directional C_nβ (1/rad)", "cn_beta_per_rad", 1.0),
        )
        derivative_columns = st.columns(3)
        airspeeds = np.asarray(stability_result["airspeeds_mps"], dtype=float)
        tilts_deg = np.asarray(stability_result["tilt_angles_deg"], dtype=float)
        trim_residuals = np.asarray(stability_result["trim_residual"], dtype=float)
        trim_hover = trim_feasible.astype(float)
        derivative_customdata = np.stack((trim_hover, trim_residuals), axis=-1)
        sign_failures = False

        for column, (title, key, stable_sign) in zip(
            derivative_columns,
            derivative_specs,
        ):
            derivative_values = np.asarray(stability_result[key], dtype=float)
            if assessed_count:
                assessed_values = derivative_values[assessed]
                sign_consistent_count = int(
                    np.count_nonzero(assessed_values * stable_sign > 0.0)
                )
                report.metric(column,
                    f"{title} sign-consistent",
                    f"{sign_consistent_count}/{assessed_count}",
                )
                sign_failures |= sign_consistent_count < assessed_count
            else:
                report.metric(column, f"{title} sign check", "Not assessed")

            figure = go.Figure(go.Heatmap(
                z=derivative_values,
                x=tilts_deg,
                y=airspeeds,
                customdata=derivative_customdata,
                colorscale="RdBu",
                zmid=0.0,
                colorbar=dict(title="1/rad"),
                hovertemplate=(
                    "Airspeed: %{y:.1f} m/s<br>"
                    "Tilt: %{x:.0f}°<br>"
                    "Derivative: %{z:.5g} 1/rad<br>"
                    "Longitudinal trim feasible: %{customdata[0]:.0f}<br>"
                    "Trim residual: %{customdata[1]:.4f}"
                    "<extra></extra>"
                ),
            ))
            figure.update_layout(
                title=title,
                xaxis_title="Nacelle tilt (deg)",
                yaxis_title="Airspeed (m/s)",
                margin=dict(l=20, r=20, t=55, b=20),
                height=340,
            )
            report.plotly_chart(column,
                figure,
                width="stretch",
                alt=f"{title} across airspeed and nacelle tilt; hover values include trim status",
                key=f"transition_{key}_heatmap",
            )

        if sign_failures:
            report.text("warning", st,
                "At least one trim-qualified grid point does not meet a reference static "
                "derivative sign check. Review the model and aircraft data; this is not "
                "a flight-control command or certification result."
            )

        trim_figure = go.Figure(go.Heatmap(
            z=trim_hover,
            x=tilts_deg,
            y=airspeeds,
            customdata=trim_residuals,
            zmin=0.0,
            zmax=1.0,
            colorscale="RdYlGn",
            colorbar=dict(
                title="Longitudinal trim",
                tickmode="array",
                tickvals=[0.0, 1.0],
                ticktext=["Not feasible", "Feasible"],
            ),
            hovertemplate=(
                "Airspeed: %{y:.1f} m/s<br>"
                "Tilt: %{x:.0f}°<br>"
                "Trim feasible: %{z:.0f}<br>"
                "Maximum normalized residual: %{customdata:.4f}"
                "<extra></extra>"
            ),
        ))
        trim_figure.update_layout(
            title="Longitudinal trim-feasibility mask",
            xaxis_title="Nacelle tilt (deg)",
            yaxis_title="Airspeed (m/s)",
            margin=dict(l=20, r=20, t=55, b=20),
            height=390,
        )
        report.plotly_chart(st,
            trim_figure,
            width="stretch",
            alt="Longitudinal trim feasibility and residual across airspeed and nacelle tilt",
            key="transition_trim_feasibility_heatmap",
        )

        effectiveness = np.asarray(
            stability_result["control_effectiveness_nm"],
            dtype=float,
        )
        effectiveness_labels = tuple(
            stability_result["control_effectiveness_labels"]
        )
        peak_axis_response = np.max(np.abs(effectiveness), axis=-1)
        report.text("markdown", st, "#### Actuator response across the transition grid")
        report.text("caption", st,
            "Heatmaps show the peak absolute moment change for each axis under the "
            "existing standard rotor/V-tail test perturbations. This is a local response "
            "comparison, not a full actuator-margin or dynamic-controllability assessment. "
            f"Test patterns: {', '.join(effectiveness_labels)}."
        )
        control_specs = (
            ("Roll moment response (N·m)", 0),
            ("Pitch moment response (N·m)", 1),
            ("Yaw moment response (N·m)", 2),
        )
        control_columns = st.columns(3)
        for column, (title, axis_index) in zip(control_columns, control_specs):
            figure = go.Figure(go.Heatmap(
                z=peak_axis_response[:, :, axis_index],
                x=tilts_deg,
                y=airspeeds,
                colorscale="Viridis",
                colorbar=dict(title="N·m"),
                hovertemplate=(
                    "Airspeed: %{y:.1f} m/s<br>"
                    "Tilt: %{x:.0f}°<br>"
                    "Peak absolute response: %{z:.4g} N·m"
                    "<extra></extra>"
                ),
            ))
            figure.update_layout(
                title=title,
                xaxis_title="Nacelle tilt (deg)",
                yaxis_title="Airspeed (m/s)",
                margin=dict(l=20, r=20, t=55, b=20),
                height=340,
            )
            report.plotly_chart(column,
                figure,
                width="stretch",
                alt=f"{title} across airspeed and nacelle tilt",
                key=f"transition_control_response_{axis_index}_heatmap",
            )

# ==========================================================
# PHASE 6: CONTROL LAW ARCHITECTURE
# ==========================================================
report.text("markdown", st, "---")
report.text("subheader", st, "Phase 6: Control law architecture — the brain")
report.text("caption", st,
    "The simulation now runs guidance → attitude → body-rate control → scheduled gains → "
    "bounded actuator allocation."
)
allocation_tab, rate_tab, attitude_tab, guidance_tab, schedule_tab = st.tabs([
    "6.1 Control allocation",
    "6.2 Inner loop: rate control",
    "6.3 Outer loop: attitude control",
    "6.4 Guidance: path following",
    "6.5 Gain scheduling",
])

with allocation_tab:
    allocation_metrics = st.columns(3)
    report.metric(allocation_metrics[0],
        "Time steps with allocation saturation",
        f"{100.0 * gust_info['allocation_saturation_fraction']:.1f}%",
    )
    report.metric(allocation_metrics[1],
        "Final normalized allocation residual",
        f"{float(data['allocation_residual'][-1]):.3f}",
    )
    report.metric(allocation_metrics[2],
        "Time steps with tilt protection",
        str(gust_info["tilt_limited_steps"]),
    )
    tilt_protection_frame = pd.DataFrame({
        "time_s": data["t"],
        "Mission tilt request (deg)": data["tilt_cmd"],
        "Controller tilt command (deg)": data["tilt_controller_cmd"],
        "Actual tilt (deg)": data["tilt"],
    })
    report.line_chart(st,
        tilt_protection_frame,
        x="time_s",
        y=[
            "Mission tilt request (deg)",
            "Controller tilt command (deg)",
            "Actual tilt (deg)",
        ],
        x_label="Time (s)",
        y_label="Nacelle tilt (deg)",
        alt="Requested, protected, and actual nacelle tilt during the mission",
    )
    allocation_frame = pd.DataFrame(
        gust_info["allocation_matrix"],
        index=ControlAllocator.output_labels,
        columns=ControlAllocator.actuator_labels,
    )
    report.dataframe(st,
        allocation_frame,
        alt="Local control allocation matrix from six actuators to collective thrust and body moments",
    )
    report.text("caption", st,
        "Derivatives per newton of rotor thrust; V-tail columns are "
        "moment derivatives per radian. Allocation is bounded by motor speed and ±11.5° "
        "tail travel; the reported residual exposes requests the actuators cannot meet."
    )
    if gust_info["allocation_saturation_fraction"] > 0.05:
        report.text("warning", st,
            "Actuator limits or the available control axes prevented at least 5% of "
            "simulation-step requests from being allocated within tolerance."
        )
    if gust_info["tilt_limited_steps"] > 0:
        report.text("warning", st,
            "Nacelle tilt was held or reduced when pitch allocation was short or "
            "pitch attitude/rate exceeded the controller's protection limits."
        )

with rate_tab:
    rate_data = pd.DataFrame({
        "time_s": data["t"],
        "p actual (deg/s)": np.rad2deg(data["body_p"]),
        "p target (deg/s)": np.rad2deg(data["desired_p"]),
        "q actual (deg/s)": np.rad2deg(data["body_q"]),
        "q target (deg/s)": np.rad2deg(data["desired_q"]),
        "r actual (deg/s)": np.rad2deg(data["body_r"]),
        "r target (deg/s)": np.rad2deg(data["desired_r"]),
    })
    report.line_chart(st,
        rate_data,
        x="time_s",
        y=[
            "p actual (deg/s)", "p target (deg/s)",
            "q actual (deg/s)", "q target (deg/s)",
            "r actual (deg/s)", "r target (deg/s)",
        ],
        x_label="Time (s)",
        y_label="Body angular rate (deg/s)",
        alt="Actual and commanded roll, pitch, and yaw rates",
    )
    report.text("caption", st,
        "The inner loop compares body rates p, q, r with the outer-loop rate targets "
        "and requests body moments, subject to the allocator and actuator limits."
    )

with attitude_tab:
    attitude_data = pd.DataFrame({
        "time_s": data["t"],
        "Roll actual (deg)": data["roll"],
        "Roll target (deg)": np.rad2deg(data["desired_roll"]),
        "Pitch actual (deg)": data["pitch"],
        "Pitch target (deg)": data["desired_pitch"],
        "Heading actual (deg)": data["yaw"],
        "Heading target (deg)": np.rad2deg(data["desired_heading"]),
    })
    report.line_chart(st,
        attitude_data,
        x="time_s",
        y=[
            "Roll actual (deg)", "Roll target (deg)",
            "Pitch actual (deg)", "Pitch target (deg)",
            "Heading actual (deg)", "Heading target (deg)",
        ],
        x_label="Time (s)",
        y_label="Attitude angle (deg)",
        alt="Actual roll, pitch, and heading compared with guidance targets",
    )
    report.text("caption", st,
        "The outer loop converts roll, pitch, and wrapped heading errors into bounded "
        "body-rate targets before the inner loop acts."
    )

with guidance_tab:
    path_figure = go.Figure()
    path_figure.add_trace(go.Scatter(
        x=data["east"],
        y=data["north"],
        mode="lines",
        name="Aircraft path",
    ))
    if gust_info["waypoints"]:
        path_figure.add_trace(go.Scatter(
            x=[waypoint[1] for waypoint in gust_info["waypoints"]],
            y=[waypoint[0] for waypoint in gust_info["waypoints"]],
            mode="markers+lines",
            marker=dict(size=9, symbol="x"),
            name="Mission waypoints",
        ))
    path_figure.update_layout(
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=360,
    )
    report.plotly_chart(st,
        path_figure,
        width="stretch",
        alt="Simulated north-east flight path and commanded waypoints",
    )
    altitude_tracking, speed_tracking = st.columns(2)
    with altitude_tracking:
        report.line_chart(st,
            pd.DataFrame({
                "time_s": data["t"],
                "Altitude (m)": data["alt"],
                "Target altitude (m)": data["ref_alt"],
            }),
            x="time_s",
            y=["Altitude (m)", "Target altitude (m)"],
            x_label="Time (s)",
            y_label="Altitude (m, positive up)",
            alt="Guidance altitude tracking",
        )
    with speed_tracking:
        report.line_chart(st,
            pd.DataFrame({
                "time_s": data["t"],
                "Horizontal airspeed (m/s)": data["airspeed_mps"],
                "Target speed (m/s)": data["ref_speed"],
            }),
            x="time_s",
            y=["Horizontal airspeed (m/s)", "Target speed (m/s)"],
            x_label="Time (s)",
            y_label="Horizontal airspeed / target speed (m/s)",
            alt="Guidance airspeed tracking",
        )
    guidance_metrics = st.columns(3)
    report.metric(guidance_metrics[0],
        "Final path distance",
        f"{float(data['guidance_path_distance'][-1]):.1f} m",
    )
    report.metric(guidance_metrics[1],
        "Final altitude error",
        f"{float(data['guidance_altitude_error'][-1]):.2f} m",
    )
    report.metric(guidance_metrics[2],
        "Final heading error",
        f"{np.rad2deg(float(data['guidance_heading_error'][-1])):.1f}°",
    )
    report.text("caption", st,
        "Position guidance steers toward the active north/east waypoint; altitude and "
        "airspeed references feed the vertical-thrust and pitch commands."
    )

with schedule_tab:
    blend_data = pd.DataFrame({
        "time_s": data["t"],
        "Transition blend (0 hover – 1 cruise)": data["gain_blend"],
        "Horizontal airspeed (m/s)": data["airspeed_mps"],
        "Nacelle tilt (deg)": data["tilt"],
    })
    blend_column, airspeed_column, tilt_column = st.columns(3)
    report.line_chart(blend_column,
        blend_data,
        x="time_s",
        y=["Transition blend (0 hover – 1 cruise)"],
        x_label="Time (s)",
        y_label="Gain-schedule blend (0–1)",
        alt="Transition gain-scheduling blend through time",
    )
    report.line_chart(airspeed_column,
        blend_data,
        x="time_s",
        y=["Horizontal airspeed (m/s)"],
        x_label="Time (s)",
        y_label="Horizontal airspeed (m/s)",
        alt="Airspeed through time",
    )
    report.line_chart(tilt_column,
        blend_data,
        x="time_s",
        y=["Nacelle tilt (deg)"],
        x_label="Time (s)",
        y_label="Nacelle tilt (deg)",
        alt="Nacelle tilt through time",
    )
    scheduled_gain_data = pd.DataFrame({
        "time_s": data["t"],
        "Rate Kp": data["gain_kp"],
        "Rate Ki": data["gain_ki"],
        "Rate Kd": data["gain_kd"],
    })
    gain_columns = st.columns(3)
    for column, (gain_key, gain_name) in zip(
        gain_columns,
        (
            ("Rate Kp", "Proportional gain Kp"),
            ("Rate Ki", "Integral gain Ki"),
            ("Rate Kd", "Derivative gain Kd"),
        ),
    ):
        report.line_chart(column,
            scheduled_gain_data,
            x="time_s",
            y=[gain_key],
            x_label="Time (s)",
            y_label=f"{gain_key} (configured units)",
            alt=f"{gain_name} scheduled through transition",
        )
    report.text("caption", st,
        "The schedule linearly blends hover and cruise rate gains using the mean of "
        "normalized airspeed (0–20 m/s) and nacelle tilt (0–90°). Hover and cruise "
        "gain scales are adjustable in the PID sidebar."
    )

# ==========================================================
# PHASE 7: DISTURBANCE REJECTION AND SAFETY
# ==========================================================
report.text("markdown", st, "---")
report.text("subheader", st, "Phase 7: Disturbance rejection and safety — the shield")
report.text("caption", st,
    " "
)
gust_tab, observer_tab, envelope_tab, monte_carlo_tab = st.tabs([
    "7.1 Gust model",
    "7.2 Transition state observer",
    "7.3 Envelope protection",
    "7.4 Monte Carlo validation",
])

with gust_tab:
    wind_metrics = st.columns(3)
    report.metric(wind_metrics[0], "Wind model", gust_info["wind_model"])
    report.metric(wind_metrics[1], "Peak wind", f"{float(np.max(data['wind'])):.2f} m/s")
    report.metric(wind_metrics[2], "Seed", str(simulation_parameters["dryden_seed"]))
    wind_frame = pd.DataFrame({
        "time_s": data["t"],
        "North gust (m/s)": data["wind_north"],
        "East gust (m/s)": data["wind_east"],
        "Down gust (m/s)": data["wind_down"],
        "Wind magnitude (m/s)": data["wind"],
    })
    report.line_chart(st,
        wind_frame,
        x="time_s",
        y=[
            "North gust (m/s)",
            "East gust (m/s)",
            "Down gust (m/s)",
            "Wind magnitude (m/s)",
        ],
        x_label="Time (s)",
        y_label="Wind component or magnitude (m/s)",
        alt="North, east, down, and total gust wind applied during the simulation",
    )
    if gust_info["wind_model"] == "Dryden turbulence":
        report.text("caption", st,
            "Dryden-like turbulence uses a seeded, correlated first-order process. "
            f"Configured σ: {gust_info['gust_sigma_mps']} m/s; "
            f"correlation time: {dryden_tau_s:.2f} s."
        )
    elif gust_info["wind_model"] == "Finite-duration gust":
        report.text("caption", st,
            f"Half-cosine gust: {gust_info['dir']}, peak {gust_info['mag']:.1f} m/s, "
            f"from {gust_info['start']:.1f} s for {gust_info['dur']:.1f} s."
        )
    else:
        report.text("info", st, "Wind disturbance is disabled for this simulation.")

with observer_tab:
    observer_metrics = st.columns(4)
    report.metric(observer_metrics[0],
        "Estimated airspeed",
        f"{float(data['transition_airspeed'][-1]):.2f} m/s",
    )
    report.metric(observer_metrics[1],
        "Estimated angle of attack",
        f"{np.rad2deg(float(data['transition_alpha_rad'][-1])):.2f}°",
    )
    report.metric(observer_metrics[2],
        "Estimated sideslip",
        f"{np.rad2deg(float(data['transition_beta_rad'][-1])):.2f}°",
    )
    report.metric(observer_metrics[3],
        "Transition phase",
        str(data["transition_phase"][-1]).title(),
    )
    observer_speed_col, observer_angle_col = st.columns(2)
    report.line_chart(observer_speed_col,
        pd.DataFrame({
            "time_s": data["t"],
            "Observer airspeed (m/s)": data["transition_airspeed"],
            "Dynamics true airspeed (m/s)": data["V_T"],
            "Reference speed (m/s)": data["ref_speed"],
        }),
        x="time_s",
        y=[
            "Observer airspeed (m/s)",
            "Dynamics true airspeed (m/s)",
            "Reference speed (m/s)",
        ],
        x_label="Time (s)",
        y_label="Airspeed (m/s)",
        alt="Observer airspeed compared with dynamics true airspeed and commanded speed",
    )
    report.line_chart(observer_angle_col,
        pd.DataFrame({
            "time_s": data["t"],
            "Angle of attack (deg)": np.rad2deg(data["transition_alpha_rad"]),
            "Sideslip (deg)": np.rad2deg(data["transition_beta_rad"]),
        }),
        x="time_s",
        y=["Angle of attack (deg)", "Sideslip (deg)"],
        x_label="Time (s)",
        y_label="Aerodynamic angle (deg)",
        alt="Filtered angle-of-attack and sideslip estimates during transition",
    )
    report.line_chart(st,
        pd.DataFrame({
            "time_s": data["t"],
            "Transition blend (0 hover – 1 cruise)": data["transition_blend"],
            "Gain-schedule blend": data["gain_blend"],
        }),
        x="time_s",
        y=["Transition blend (0 hover – 1 cruise)", "Gain-schedule blend"],
        x_label="Time (s)",
        y_label="Transition progress (0–1)",
        alt="Observer transition progress compared with the controller gain-schedule blend",
    )
    report.text("caption", st,
        f"Observer filter time constant: {observer_time_constant_s:.2f} s. "
        "Airspeed is computed from body-frame velocity relative to the simulated wind."
    )

with envelope_tab:
    if gust_info["envelope_protection_active"]:
        protection_status = "Active"
    elif enable_envelope_protection:
        protection_status = "Unavailable"
        report.text("warning", st,
            "Envelope protection was requested, but the current aircraft configuration "
            "has no feasible transition-corridor samples, so no envelope filter was applied."
        )
    else:
        protection_status = "Disabled"

    envelope_metrics = st.columns(4)
    report.metric(envelope_metrics[0], "Protection", protection_status)
    report.metric(envelope_metrics[1],
        "Steps with protected commands",
        f"{100.0 * gust_info['envelope_protected_fraction']:.1f}%",
    )
    report.metric(envelope_metrics[2],
        "Steps outside corridor",
        f"{100.0 * gust_info['envelope_outside_fraction']:.1f}%",
    )
    report.metric(envelope_metrics[3],
        "Mean available control authority",
        f"{100.0 * float(np.mean(data['available_control_authority'])):.1f}%",
    )

    envelope_figure = go.Figure()
    feasible = envelope["feasible"]
    if np.any(feasible):
        corridor_tilt = envelope["tilt_deg"][feasible]
        minimum_speed = envelope["minimum_speed_mps"][feasible]
        maximum_speed = envelope["maximum_speed_mps"][feasible]
        envelope_figure.add_trace(go.Scatter(
            x=np.concatenate([corridor_tilt, corridor_tilt[::-1]]),
            y=np.concatenate([maximum_speed, minimum_speed[::-1]]),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.16)",
            line=dict(color="rgba(52, 152, 219, 0.25)"),
            name="Feasible corridor",
            hoverinfo="skip",
        ))
        envelope_figure.add_trace(go.Scatter(
            x=corridor_tilt,
            y=minimum_speed,
            mode="lines",
            name="Minimum speed",
        ))
        envelope_figure.add_trace(go.Scatter(
            x=corridor_tilt,
            y=maximum_speed,
            mode="lines",
            name="Maximum speed",
        ))
    envelope_figure.add_trace(go.Scatter(
        x=data["tilt"],
        y=data["transition_airspeed"],
        mode="lines",
        name="Observed flight",
        line=dict(color="darkorange", width=2),
    ))
    envelope_figure.update_layout(
        xaxis_title="Nacelle tilt (deg)",
        yaxis_title="Airspeed (m/s)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
    )
    report.plotly_chart(st,
        envelope_figure,
        width="stretch",
        alt="Observed transition against the feasible airspeed and nacelle-tilt corridor",
    )

    protection_frame = pd.DataFrame({
        "time_s": data["t"],
        "Requested tilt (deg)": data["requested_tilt"] * 180.0 / np.pi,
        "Protected tilt target (deg)": data["protected_tilt"] * 180.0 / np.pi,
        "Actual tilt (deg)": data["tilt"],
        "Available control authority (%)": 100.0 * data["available_control_authority"],
    })
    protection_tilt_col, protection_authority_col = st.columns(2)
    report.line_chart(protection_tilt_col,
        protection_frame,
        x="time_s",
        y=[
            "Requested tilt (deg)",
            "Protected tilt target (deg)",
            "Actual tilt (deg)",
        ],
        x_label="Time (s)",
        y_label="Nacelle tilt (deg)",
        alt="Requested, protected, and actual nacelle tilt",
    )
    report.line_chart(protection_authority_col,
        protection_frame,
        x="time_s",
        y=["Available control authority (%)"],
        x_label="Time (s)",
        y_label="Available control authority (%)",
        alt="Estimated available control authority through time",
    )
    reason_series = pd.Series(data["envelope_reasons"])
    reason_counts = (
        reason_series[reason_series != ""]
        .str.split(", ")
        .explode()
        .value_counts()
        .rename_axis("Protection reason")
        .reset_index(name="Time steps")
    )
    if reason_counts.empty:
        report.text("info", st, "No Phase 7 envelope-protection limit was triggered in this simulation.")
    else:
        report.dataframe(st,
            reason_counts,
            hide_index=True,
            alt="Counts of each envelope-protection reason during the simulation",
        )
    report.text("caption", st,
        f"Protection limits: α ≤ {max_alpha_deg:.1f}°, pitch ≤ {max_pitch_deg:.1f}°, "
        f"body rates ≤ {max_body_rate_deg_s:.0f}°/s, and a {speed_margin_mps:.1f} m/s "
        "margin around the quasi-steady corridor. These are simulation safeguards, not certification."
    )

with monte_carlo_tab:
    report.text("caption", st,
        "Each trial runs the configured mission with a reproducible Dryden-like gust and, "
        "according to the selected probability, one degraded actuator."
    )
    if generate_report or st.button(
        "Run Monte Carlo validation",
        key="run_phase7_monte_carlo",
        type="primary",
        icon=":material/science:",
    ):
        validator = MonteCarloValidator(
            trials=int(monte_carlo_trials),
            seed=int(monte_carlo_seed),
            gust_sigma_max_mps=(
                monte_carlo_gust_sigma,
                monte_carlo_gust_sigma,
                monte_carlo_gust_sigma * 0.6,
            ),
            failure_probability=monte_carlo_failure_probability,
            failure_time_range_s=(5.0, 45.0),
        )

        def simulate_monte_carlo_trial(scenario):
            trial_parameters = dict(simulation_parameters)
            trial_parameters.update({
                "gust_enabled": True,
                "gust_model": "Dryden turbulence",
                "gust_start": 0.0,
                "gust_duration": 63.0,
                "gust_mag": 0.0,
                "gust_dir": "Dryden turbulence",
                "dryden_sigma_mps": scenario.gust_sigma_mps,
                "dryden_seed": scenario.seed,
                "actuator_failure": scenario.actuator_failure,
            })
            trial_data, trial_info = run_simulation(**trial_parameters)
            mission_completed = (
                trial_info["termination_reason"] == "Mission duration completed"
                and not (
                    flight_plan == "Waypoint route"
                    and mission_mode == "AUTO"
                    and not trial_info["mission_complete"]
                )
            )
            return {
                "completed": mission_completed,
                "max_abs_pitch_rad": float(np.deg2rad(np.max(np.abs(trial_data["pitch"])))),
                "max_abs_alpha_rad": float(np.max(np.abs(trial_data["transition_alpha_rad"]))),
                "min_altitude_m": float(np.min(trial_data["alt"])),
                "max_allocation_residual": float(np.max(trial_data["allocation_residual"])),
            }

        with st.spinner(f"Running {int(monte_carlo_trials)} seeded simulation trials..."):
            st.session_state["phase7_monte_carlo_report"] = validator.run(
                simulate_monte_carlo_trial
            )

    monte_carlo_report = st.session_state.get("phase7_monte_carlo_report")
    if monte_carlo_report is None:
        report.text("info", st, "Run validation to evaluate gust and actuator-failure scenarios.")
    else:
        failed_trials = [trial for trial in monte_carlo_report.trials if not trial.passed]
        mc_metrics = st.columns(3)
        report.metric(mc_metrics[0], "Pass rate", f"{100.0 * monte_carlo_report.success_rate:.1f}%")
        report.metric(mc_metrics[1], "Passed trials", f"{len(monte_carlo_report.trials) - len(failed_trials)}")
        report.metric(mc_metrics[2], "Failed trials", str(len(failed_trials)))

        trial_rows = []
        for trial in monte_carlo_report.trials:
            metrics = trial.metrics
            failure = trial.scenario.actuator_failure
            trial_rows.append({
                "Trial": trial.scenario.trial_index + 1,
                "Seed": trial.scenario.seed,
                "Gust σ (N/E/D, m/s)": " / ".join(
                    f"{value:.2f}" for value in trial.scenario.gust_sigma_mps
                ),
                "Failed actuator": failure.channel if failure is not None else "None",
                "Effectiveness left": (
                    f"{100.0 * failure.remaining_effectiveness:.0f}%"
                    if failure is not None else "—"
                ),
                "Completed": metrics["completed"],
                "Peak |pitch| (deg)": np.rad2deg(metrics["max_abs_pitch_rad"]),
                "Peak |α| (deg)": np.rad2deg(metrics["max_abs_alpha_rad"]),
                "Minimum altitude (m)": metrics["min_altitude_m"],
                "Result": "PASS" if trial.passed else "FAIL",
                "Failure reasons": ", ".join(trial.failure_reasons),
            })
        trial_frame = pd.DataFrame(trial_rows)
        report.dataframe(st,
            trial_frame,
            hide_index=True,
            alt="Per-trial Monte Carlo gust, actuator-failure, and safety results",
        )

        pitch_chart_col, alpha_chart_col = st.columns(2)
        pitch_figure = go.Figure(go.Histogram(
            x=trial_frame["Peak |pitch| (deg)"],
            name="Peak pitch",
            nbinsx=12,
        ))
        pitch_figure.update_layout(
            xaxis_title="Peak absolute pitch angle (deg)",
            yaxis_title="Number of trials",
        )
        report.plotly_chart(pitch_chart_col,
            pitch_figure,
            width="stretch",
            alt="Distribution of peak absolute pitch across Monte Carlo trials",
        )
        alpha_figure = go.Figure(go.Histogram(
            x=trial_frame["Peak |α| (deg)"],
            name="Peak angle of attack",
            nbinsx=12,
        ))
        alpha_figure.update_layout(
            xaxis_title="Peak absolute angle of attack (deg)",
            yaxis_title="Number of trials",
        )
        report.plotly_chart(alpha_chart_col,
            alpha_figure,
            width="stretch",
            alt="Distribution of peak absolute angle of attack across Monte Carlo trials",
        )
        report.text("caption", st,
            f"Seed {monte_carlo_report.seed}; safety limits are |pitch| ≤ 60°, "
            "|α| ≤ 25°, minimum altitude ≥ 0 m, and mission completion. "
            "This Monte Carlo batch is a model validation aid, not flight certification."
        )

# ==========================================================
# PHASE 8: AIRCRAFT PROFILE AND BENCH HIL READINESS
# ==========================================================
report.text("markdown", st, "---")
report.text("subheader", st, "Aircraft profile and PX4 bench-HIL readiness")
report.text("caption", st,
    "Load and version aircraft data independently of this example model. Custom fields are "
    "preserved in profile files; only the currently supported four-rotor tiltrotor subset "
    "is used by the flight simulation."
)
profile_tab, hil_tab = st.tabs(["8.1 Aircraft profile", "8.2 PX4 bench-HIL boundary"])

with profile_tab:
    aircraft_data = export_profile.get("aircraft", {})
    if st.session_state.get("_aircraft_summary_fingerprint") != profile_fingerprint:
        profile_summary = {
            "parameter_count": count_profile_values(aircraft_data),
            "extension_count": count_profile_values(export_profile.get("extensions", {})),
            "subsystems": [
                {
                    "Subsystem": name.replace("_", " ").title(),
                    "Configured values": count_profile_values(value),
                    "Status": "Present" if value else "Not defined",
                }
                for name, value in aircraft_data.items()
            ] if isinstance(aircraft_data, dict) else [],
        }
        st.session_state["_aircraft_summary"] = profile_summary
        st.session_state["_aircraft_summary_fingerprint"] = profile_fingerprint
    profile_summary = st.session_state["_aircraft_summary"]
    profile_metrics = st.columns(4)
    report.metric(profile_metrics[0], "Profile", active_profile["metadata"]["name"])
    report.metric(profile_metrics[1], "Schema version", str(active_profile["schema_version"]))
    report.metric(profile_metrics[2],
        "Aircraft data values", f"{profile_summary['parameter_count']:,}"
    )
    report.metric(profile_metrics[3],
        "Custom extension values",
        f"{profile_summary['extension_count']:,}",
    )
    report.dataframe(st,
        pd.DataFrame(profile_summary["subsystems"]),
        hide_index=True,
        alt="Aircraft profile subsystem coverage and configured value counts",
    )
    report.text("info", st,
        " "
    )

with hil_tab:
    hardware_profile = export_profile.get("hardware", {})
    hil_metrics = st.columns(3)
    report.metric(hil_metrics[0], "Autopilot", str(hardware_profile.get("autopilot", "Not specified")))
    report.metric(hil_metrics[1], "Protocol", str(hardware_profile.get("protocol", "Not specified")))
    report.metric(hil_metrics[2],
        "Execution mode",
        str(hardware_profile.get("execution_mode", "simulation")),
    )
    report.text("warning", st,
        " "
    )
    report.text("markdown", st,
        " "
    )

# ==========================================================
# 3D COORDINATE FRAME & CAD MODEL VISUALIZATION
# ==========================================================
import trimesh

report.text("markdown", st, "---")
report.text("subheader", st, "3D Aircraft & Coordinate Systems")
report.text("markdown", st, "The Earth frame is fixed (Red/Green/Blue). The UAM rotates based on the flight telemetry.")

time_idx = st.slider("Scrub through flight time (s)", 0.0, float(data["t"][-1]), 5.0, 0.1)
idx = np.argmin(np.abs(data["t"] - time_idx))

current_pitch = data["pitch"][idx]
current_roll = data["roll"][idx]
current_yaw = data["yaw"][idx]

# Rotation matrix at this exact moment
R = rotation_body_to_earth(np.deg2rad(current_roll), np.deg2rad(current_pitch), np.deg2rad(current_yaw))

fig_3d = go.Figure()

# --- 1. Inertial NED Frame ---
fig_3d.add_trace(go.Scatter3d(x=[0, 3], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=4), name='Earth X (North)'))
fig_3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, 3], z=[0, 0], mode='lines', line=dict(color='green', width=4), name='Earth Y (East)'))
fig_3d.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 3], mode='lines', line=dict(color='blue', width=4), name='Earth Z (Down)'))

# --- 2. Body Frame Axes ---
bx = R @ np.array([1.5, 0, 0])
fig_3d.add_trace(go.Scatter3d(x=[0, bx[0]], y=[0, bx[1]], z=[0, bx[2]], mode='lines', line=dict(color='orange', width=6), name='Body X'))
by = R @ np.array([0, 1.5, 0])
fig_3d.add_trace(go.Scatter3d(x=[0, by[0]], y=[0, by[1]], z=[0, by[2]], mode='lines', line=dict(color='purple', width=6), name='Body Y'))
bz = R @ np.array([0, 0, 1.5])
fig_3d.add_trace(go.Scatter3d(x=[0, bz[0]], y=[0, bz[1]], z=[0, bz[2]], mode='lines', line=dict(color='cyan', width=6), name='Body Z'))

# --- 3. The CAD STL Model (Auto-Centered & Scaled) ---
stl_path = os.path.join(base_dir, "configs", "uam_model.stl")

if os.path.exists(stl_path):
    try:
        mesh = trimesh.load(stl_path)
        
        # 1. Center the model at origin (0,0,0)
        vertices = mesh.vertices - mesh.centroid
        
        # 2. Scale the model to fit the view (max dimension = 2.0 meters)
        scale_factor = 2.0 / np.max(np.abs(vertices))
        vertices = vertices * scale_factor
        
        faces = mesh.faces
        
        # 3. Rotate using the DCM
        rotated_vertices = (R @ vertices.T).T
        
        # 4. Plot
        fig_3d.add_trace(go.Mesh3d(
            x=rotated_vertices[:, 0], 
            y=rotated_vertices[:, 1], 
            z=rotated_vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='steelblue',
            opacity=1.0,
            name='UAM CAD Model'
        ))
        report.text("success", st, "✅ CAD Model loaded and centered successfully!")
    except Exception as e:
        report.text("error", st, f"Error loading STL: {e}")
else:
    report.text("warning", st, f"⚠️ CAD model not found at {stl_path}")

# --- Layout ---
fig_3d.update_layout(
    scene=dict(
        xaxis_title='North X (m)',
        yaxis_title='East Y (m)',
        zaxis_title='Down Z (m)',
        xaxis=dict(range=[-2, 2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        yaxis=dict(range=[-2, 2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        zaxis=dict(range=[-2, 2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

report.plotly_chart(st, fig_3d, width="stretch", alt="Three-dimensional aircraft model with the earth and body coordinate frames")
# ==========================================================
# ==========================================================
# 3D CAD SKELETON WITH CUSTOM AIRFOILS & CONTROL SURFACES
# ==========================================================
from src.aero.airfoil import generate_naca4, parse_airfoil_file

report.text("markdown", st, "---")
report.text("subheader", st, " 3D CAD Geometry & Control Surfaces")

# --- AIRFOIL SELECTION ---
report.text("markdown", st, "### 1. Airfoil Selection")
c1, c2 = st.columns(2)

with c1:
    report.text("markdown", st, "**Main Wing**")
    wing_af_type = st.selectbox("Type", ["NACA 6412", "Custom Upload"], key="w_type")
    if wing_af_type == "NACA 6412":
        
        w_coords = generate_naca4(m=0.06, p=0.4, t=0.12)
    else:
        w_file = st.file_uploader("Upload Wing .dat", type=["dat", "txt"], key="w_up")
        if w_file is not None:
            w_coords = parse_airfoil_file(w_file.getvalue())
        else:
            w_coords = generate_naca4(m=0.06, p=0.4, t=0.12)

with c2:
    report.text("markdown", st, "**V-Tail**")
    vt_af_type = st.selectbox("Type", ["NACA 6412", "Custom Upload"], key="vt_type")
    if vt_af_type == "NACA 6412":
        vt_coords = generate_naca4(m=0.06, p=0.4, t=0.12)
    else:
        vt_file = st.file_uploader("Upload V-Tail .dat", type=["dat", "txt"], key="vt_up")
        if vt_file is not None:
            vt_coords = parse_airfoil_file(vt_file.getvalue())
        else:
            vt_coords = generate_naca4(m=0.06, p=0.4, t=0.12)

# --- CONTROL SURFACES ---
report.text("markdown", st, "### 2. Control Surface Parameters")
c3, c4 = st.columns(2)
with c3:
    cs_chord = st.slider("Control Surface Chord (% of Wing Chord)", 0.1, 0.5, 0.20, 0.05)
with c4:
    cs_span = st.slider("Control Surface Span (% of Wingspan)", 0.1, 1.0, 0.80, 0.05)

tilt_deg = st.slider("Rotor Tilt Angle (deg)", 0.0, 90.0, 0.0, 1.0)

# --- GENERATE MESHES ---

tilted_motors = cad.calculate_tilted_motors(tilt_deg)
fig_cad = go.Figure()

# Wing Meshes
for side in ["wing_right", "wing_left"]:
    main_verts, cs_verts = cad.generate_wing_mesh(side, w_coords, cs_chord, cs_span)
    
    # Plot Main Wing (Blue)
    fig_cad.add_trace(go.Mesh3d(x=main_verts[:,0], y=main_verts[:,1], z=-main_verts[:,2], color='steelblue', opacity=0.9, name="Main Wing"))
    
    # Plot Control Surface (Cyan - Aileron/Flap)
    fig_cad.add_trace(go.Mesh3d(x=cs_verts[:,0], y=cs_verts[:,1], z=-cs_verts[:,2], color='cyan', opacity=1.0, name="Control Surface"))

# V-Tail Meshes
vt_left, vt_right = cad.generate_vtail_mesh(vt_coords)
fig_cad.add_trace(go.Mesh3d(x=vt_left[:,0], y=vt_left[:,1], z=-vt_left[:,2], color='darkred', opacity=0.9, name="V-Tail Left"))
fig_cad.add_trace(go.Mesh3d(x=vt_right[:,0], y=vt_right[:,1], z=-vt_right[:,2], color='darkred', opacity=0.9, name="V-Tail Right"))

# Motors & Thrust
for m in tilted_motors:
    color = 'blue' if m["tilts"] else 'orange'
    pos = m["pos"]
    t_dir = m["thrust_dir"]
    fig_cad.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[-pos[2]], mode='markers', marker=dict(size=8, color=color), name=f"M{m['id']}"))
    
    arrow_len = 0.3
    end_x, end_y, end_z = pos[0] + t_dir[0]*arrow_len, pos[1] + t_dir[1]*arrow_len, -(pos[2] + t_dir[2]*arrow_len)
    fig_cad.add_trace(go.Scatter3d(x=[pos[0], end_x], y=[pos[1], end_y], z=[-pos[2], end_z], mode='lines', line=dict(color=color, width=4, dash='dash'), showlegend=False))

fig_cad.update_layout(
    scene=dict(
        xaxis_title='X forward (m)',
        yaxis_title='Y right (m)',
        zaxis_title='Z up (m)',
        xaxis=dict(range=[-1.0, 1.0], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        yaxis=dict(range=[-1.5, 1.5], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        zaxis=dict(range=[-0.5, 0.5], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        aspectmode='manual', aspectratio=dict(x=1, y=1.5, z=0.5),
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)) 
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=600
)

report.plotly_chart(st, fig_cad, width="stretch", alt="Three-dimensional wing, V-tail, control surfaces, motors, and thrust directions")

report_signature = hashlib.sha256(
    repr((
        dashboard_values_fingerprint,
        simulation_parameters,
        corridor_signature,
        stability_signature,
        analysis_speed,
        analysis_tilt_deg,
        monte_carlo_trials,
        monte_carlo_seed,
        monte_carlo_gust_sigma,
        monte_carlo_failure_probability,
        profile_preview_tilt_deg,
        time_idx,
        cs_chord,
        cs_span,
        tilt_deg,
        np.asarray(w_coords).tolist(),
        np.asarray(vt_coords).tolist(),
    )).encode("utf-8")
).hexdigest()

if generate_report:
    report.items.insert(
        0,
        ReportItem(
            "text",
            "",
            f"Simulation termination: {gust_info['termination_reason']}; "
            f"recorded duration: {gust_info['elapsed_time']:.2f} s. "
            f"Simulation inputs: {simulation_parameters}",
        ),
    )
    with st.spinner("Rendering all report graphs and creating Word and PDF files..."):
        try:
            report_word, report_pdf = report.render(
                active_profile["metadata"]["name"],
                serialize_aircraft_profile(export_profile).decode("utf-8"),
            )
        except (RuntimeError, ValueError, OSError) as exc:
            report_downloads.error(f"Could not generate the full report: {exc}")
        else:
            st.session_state["full_simulation_report"] = {
                "signature": report_signature,
                "word": report_word,
                "pdf": report_pdf,
            }

saved_report = st.session_state.get("full_simulation_report")
if isinstance(saved_report, dict) and saved_report.get("signature") == report_signature:
    report_downloads.download_button(
        "Download report (Word)",
        data=saved_report["word"],
        file_name="aircraft_transition_simulation_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        on_click="ignore",
        key="download_simulation_report_word",
    )
    report_downloads.download_button(
        "Download report (PDF)",
        data=saved_report["pdf"],
        file_name="aircraft_transition_simulation_report.pdf",
        mime="application/pdf",
        on_click="ignore",
        key="download_simulation_report_pdf",
    )
elif isinstance(saved_report, dict):
    report_downloads.info("Inputs changed. Generate a fresh report for the current configuration.")

