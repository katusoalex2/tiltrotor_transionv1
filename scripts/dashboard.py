
# Cache Buster: 2026-11-11-00-37-15
"""KADA UAM Aircraft Transition FlightConfigurator"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import sys
import os
import plotly.graph_objects as go

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
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.trajectory import ReferenceTrajectory
from src.control.pid import PIDController
from src.dynamics.mass_properties import MassProperties
from src.dynamics.frames import rotation_body_to_earth
from src.control.mission import MissionManager, MissionProfile, Waypoint


# ==========================================================
# 3. LOAD CONFIG
# ==========================================================
cfg_path = os.path.join(base_dir, "configs", "blue_uam.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
# Initialize the CAD Geometry engine

geo_path = os.path.join(base_dir, "configs", "geometry_offsets.yaml")
cad = CadGeometry(geo_path)

class DiscreteGust:
    def __init__(self, start_time, duration, magnitude, direction):
        self.start = start_time
        self.duration = duration
        self.mag = magnitude
        self.dir = direction
        self.end = start_time + duration

    def get_wind(self, t):
        if t < self.start or t > self.end:
            return np.array([0.0, 0.0, 0.0])
        frac = (t - self.start) / self.duration
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * frac))
        current_mag = self.mag * shape
        if self.dir == "Front": return np.array([current_mag, 0.0, 0.0])
        if self.dir == "Back": return np.array([-current_mag, 0.0, 0.0])
        if self.dir == "Right": return np.array([0.0, current_mag, 0.0])
        if self.dir == "Left": return np.array([0.0, -current_mag, 0.0])
        if self.dir == "Up": return np.array([0.0, 0.0, -current_mag])
        if self.dir == "Down": return np.array([0.0, 0.0, current_mag])
        if self.dir == "Diagonal": 
            m = current_mag / np.sqrt(3)
            return np.array([m, m, -m])
        return np.array([0.0, 0.0, 0.0])

class FullCascadePID:
    def __init__(self, mass, k_thrust, gains):
        self.mass = mass
        self.k_thrust = k_thrust
        self.alt_pid = PIDController(kp=gains['alt_pid']['kp'], ki=gains['alt_pid']['ki'], kd=gains['alt_pid']['kd'], limit=500.0)
        self.speed_pid = PIDController(kp=gains['speed_pid']['kp'], ki=gains['speed_pid']['ki'], kd=gains['speed_pid']['kd'], limit=0.35)
        self.att_pid = PIDController(kp=gains['att_pid']['kp'], ki=gains['att_pid']['ki'], kd=gains['att_pid']['kd'], limit=2.0)
        self.rate_pid = PIDController(kp=gains['rate_pid']['kp'], ki=gains['rate_pid']['ki'], kd=gains['rate_pid']['kd'], limit=0.2)
        self.roll_pid = PIDController(kp=1.0, ki=0.1, kd=0.2, limit=1.0)
        self.yaw_pid = PIDController(kp=1.0, ki=0.1, kd=0.2, limit=1.0)

    def compute(self, state, ref_alt, ref_speed, ref_tilt, dt):
        current_alt = -state.pos[2]
        current_speed = state.vel[0]
        current_pitch = state.att[1]
        current_pitch_rate = state.omega[1]
        current_roll = state.att[0]
        current_yaw = state.att[2]
        
        alt_error = ref_alt - current_alt
        thrust_correction = self.alt_pid.compute(0, -alt_error, dt)
        base_thrust = (self.mass * 9.81) + thrust_correction
        omega_base = np.sqrt(max(base_thrust, 0.0) / (4 * self.k_thrust))
        
        roll_cmd = self.roll_pid.compute(0, -current_roll, dt)
        yaw_cmd = self.yaw_pid.compute(0, -current_yaw, dt)
        
        omega_fl = np.clip(omega_base + roll_cmd + yaw_cmd, 0, 3000)
        omega_fr = np.clip(omega_base - roll_cmd - yaw_cmd, 0, 3000)
        omega_rl = np.clip(omega_base + roll_cmd - yaw_cmd, 0, 3000)
        omega_rr = np.clip(omega_base - roll_cmd + yaw_cmd, 0, 3000)

        speed_error = ref_speed - current_speed
        pitch_from_speed = -self.speed_pid.compute(0, -speed_error, dt)
        pitch_from_alt = alt_error * 0.05 
        desired_pitch = np.clip(pitch_from_alt + pitch_from_speed, -0.26, 0.26)
        
        pitch_error = desired_pitch - current_pitch
        desired_pitch_rate = self.att_pid.compute(0, -pitch_error, dt)
        
        rate_error = desired_pitch_rate - current_pitch_rate
        vtail_cmd = -self.rate_pid.compute(0, -rate_error, dt)
        
        return {"omega_fl": omega_fl, "omega_fr": omega_fr, "omega_rl": omega_rl, "omega_rr": omega_rr, 
                "tilt": ref_tilt, "delta_left": vtail_cmd, "delta_right": vtail_cmd, 
                "desired_pitch": desired_pitch}

st.set_page_config(page_title="Tilt-rotor systems dashboard", layout="wide")
st.title("Tilt-rotor systems dashboard")
st.caption(
    "Aircraft configuration · aerodynamics · propulsion and tilt · guidance and control · mission management"
)

st.sidebar.title("Aircraft and mission setup")
side_tabs = st.sidebar.tabs(["Body", "Prop", "Geom", "Aero", "PID", "Gust", "Mission"])

with side_tabs[0]:
    st.markdown("**Component Masses (kg)**")
    mass_fuselage = st.number_input("Fuselage Mass", value=3.0, step=0.1)
    mass_wing = st.number_input("Wing Mass", value=1.0, step=0.1)
    mass_vtail = st.number_input("V-Tail Mass", value=0.5, step=0.1)
    mass_motor = st.number_input("Single Motor Mass", value=0.125, step=0.01)
    
    # Calculate total mass automatically
    mass = mass_fuselage + mass_wing + mass_vtail + (4 * mass_motor)
    st.info(f"Total Mass: **{mass:.2f} kg**")
    
with side_tabs[1]:
    st.markdown("**Battery**")
    c1, c2 = st.columns(2)
    battery_cells = c1.number_input("Cells (S)", value=6, step=1)
    battery_mah = c2.number_input("Capacity (mAh)", value=5000, step=100)
    
    c1, c2 = st.columns(2)
    battery_c_rating = c1.number_input("C-Rating", value=30, step=5)
    battery_v_nom = battery_cells * 3.7
    battery_v_max = battery_cells * 4.2
    
    c2.metric("Max Voltage", f"{battery_v_max:.1f} V")
    st.info(f"Max Discharge Current: **{(battery_mah/1000)*battery_c_rating:.1f} A**")
    
    st.markdown("**Motor**")
    motor_kv = st.number_input("Motor KV (RPM/V)", value=920, step=50)
    
    st.markdown("**Propeller**")
    c1, c2 = st.columns(2)
    prop_dia = c1.number_input("Diameter (in)", value=12.0, step=0.5)
    prop_pitch = c2.number_input("Pitch (in)", value=5.5, step=0.5)
    
    num_blades = st.number_input("Number of Blades", value=2, step=1, min_value=2, max_value=5)
    max_tilt_rate_deg_s = st.slider(
        "Maximum nacelle tilt rate (deg/s)", 1.0, 90.0, 30.0, 1.0, key="max_tilt_rate_deg_s"
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
    st.markdown("**Wing Planform (XFLR5 Style)**")
    c1, c2 = st.columns(2)
    root_x = c1.number_input("Root LE X (m)", value=0.10, step=0.05, key="geo_root_x")
    root_y = c2.number_input("Root Y (m)", value=0.0, step=0.05, key="geo_root_y")
    
    tip_x = c1.number_input("Tip LE X (m)", value=0.05, step=0.05, key="geo_tip_x")
    tip_y = c2.number_input("Tip Y (m)", value=1.00, step=0.05, key="geo_tip_y")
    
    root_c = c1.number_input("Root Chord (m)", value=0.40, step=0.05, key="geo_root_c")
    tip_c = c2.number_input("Tip Chord (m)", value=0.20, step=0.05, key="geo_tip_c")
    
    sweep = st.number_input("Sweep Angle (deg)", value=0.0, step=5.0, key="geo_sweep")
    
    st.markdown("**V-Tail**")
    vt_root_x = st.number_input("V-Tail Root X (m)", value=-0.6, step=0.05, key="geo_vt_x")
    vt_tip_y = st.number_input("V-Tail Tip Y (m)", value=0.3, step=0.05, key="geo_vt_y")
    vt_tip_z = st.number_input("V-Tail Tip Z (m)", value=-0.15, step=0.05, key="geo_vt_z")
    vtail_area = st.number_input("V-Tail Area", value=0.1, step=0.01, key="geo_vt_area")
    vtail_arm = st.number_input("V-Tail Arm (m)", value=0.5, step=0.05, key="geo_vt_arm")
    
    st.markdown("**Motors**")
    def_motors = cfg.get('motor_positions', [[0.3, 0.4, 0.0, True], [0.3, -0.4, 0.0, True], [-0.3, 0.4, 0.0, False], [-0.3, -0.4, 0.0, False]])
    motor_positions = []
    for i in range(4):
        c1, c2, c3, c4 = st.columns(4)
        mx = c1.number_input(f"X{i+1}", value=float(def_motors[i][0]), step=0.05, key=f"geo_mx{i}")
        my = c2.number_input(f"Y{i+1}", value=float(def_motors[i][1]), step=0.05, key=f"geo_my{i}")
        mz = c3.number_input(f"Z{i+1}", value=float(def_motors[i][2]), step=0.01, key=f"geo_mz{i}")
        tilts = c4.checkbox(f"Tilt{i+1}", value=bool(def_motors[i][3]), key=f"geo_tilt{i}")
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
    alt_kp = st.number_input("Alt Kp", value=1.5, step=0.1)
    alt_ki = st.number_input("Alt Ki", value=0.05, step=0.01)
    alt_kd = st.number_input("Alt Kd", value=0.1, step=0.01)
    spd_kp = st.number_input("Spd Kp", value=0.1, step=0.01)
    spd_ki = st.number_input("Spd Ki", value=0.0, step=0.01)
    spd_kd = st.number_input("Spd Kd", value=0.01, step=0.01)
    att_kp = st.number_input("Att Kp", value=1.0, step=0.1)
    att_ki = st.number_input("Att Ki", value=0.0, step=0.1)
    att_kd = st.number_input("Att Kd", value=0.05, step=0.01)
    rate_kp = st.number_input("Rate Kp", value=0.5, step=0.05)
    rate_ki = st.number_input("Rate Ki", value=0.0, step=0.01)
    rate_kd = st.number_input("Rate Kd", value=0.01, step=0.01)

with side_tabs[5]:
    gust_enabled = st.checkbox("Enable Wind Gust", value=True)
    random_gust = st.checkbox("Randomize (Monte Carlo)", value=False)
    if not random_gust:
        gust_start = st.slider("Start Time (s)", 0.0, 15.0, 5.0, 0.5)
        gust_duration = st.slider("Duration (s)", 0.5, 5.0, 2.0, 0.5)
        gust_mag = st.slider("Magnitude (m/s)", 0.0, 15.0, 6.0, 0.5)
        gust_dir = st.selectbox("Direction", ["Front", "Back", "Left", "Right", "Up", "Down", "Diagonal"])
    else:
        gust_start, gust_duration, gust_mag, gust_dir = 0, 0, 0, "Front"

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

# Map gust settings to the wind variables expected by run_simulation
wind_enabled = gust_enabled
wind_speed = gust_mag
wind_direction = gust_start       # (see note below)
wind_turbulence = gust_duration   # (see note below)
surface_roughness = gust_dir      # (see note below)

@st.cache_data
def run_simulation(mass, mass_fuselage, mass_wing, mass_vtail, mass_motor, battery_cells, battery_mah, battery_c_rating, motor_kv, prop_dia, prop_pitch, num_blades, k_thrust, max_omega, wing_area, vtail_area, vtail_arm, root_x, root_c, tip_c, wingspan, sweep, motor_positions, alt_kp, alt_ki, alt_kd, spd_kp, spd_ki, spd_kd, att_kp, att_ki, att_kd, rate_kp, rate_ki, rate_kd, gust_enabled, gust_start, gust_duration, gust_mag, gust_dir, max_tilt_rate_deg_s, flight_plan, mission_mode, manual_altitude, manual_speed, manual_tilt_deg, waypoint_rows):
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
    props.wingspan = wingspan
    props.root_chord = root_c
    props.tip_chord = tip_c
    props.sweep_deg = sweep
    props.le_root_x = root_x
    props.motor_positions = [list(position) for position in motor_positions]
    props.mac, props.ac_wing_x = props._calculate_mac_and_ac()
    props.neutral_point_x = props._calculate_neutral_point()
    
    dyn = TiltrotorDynamics(mass=mass, rotor=rotor, mass_props=props, aero=aero)
    
    gains = {
        'alt_pid': {'kp': alt_kp, 'ki': alt_ki, 'kd': alt_kd},
        'speed_pid': {'kp': spd_kp, 'ki': spd_ki, 'kd': spd_kd},
        'att_pid': {'kp': att_kp, 'ki': att_ki, 'kd': att_kd},
        'rate_pid': {'kp': rate_kp, 'ki': rate_ki, 'kd': rate_kd}
    }
    controller = FullCascadePID(mass, k_thrust, gains)
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
    
    if gust_enabled:
        gust = DiscreteGust(gust_start, gust_duration, gust_mag, gust_dir)
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
        "wind": [], "tilt": [], "roll": [], "yaw": [], "ld_ratio": [], 
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

        cmd = controller.compute(state, r["altitude"], r["speed"], r["tilt"], dt)
        
        current_alt = -state.pos[2]
        wind_vec = gust.get_wind(t) if gust is not None else np.array([0.0, 0.0, 0.0])
        tilt_error = r["tilt"] - state.tilt
        tilt_step = np.deg2rad(max_tilt_rate_deg_s) * dt
        actual_tilt = state.tilt + np.clip(tilt_error, -tilt_step, tilt_step)
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
        
        # --- LIFT HANDOFF LOGGING ---
        total_thrust = sum(
            rotor.compute_thrust(omega, height_above_ground)
            for omega in (cmd["omega_fl"], cmd["omega_fr"], cmd["omega_rl"], cmd["omega_rr"])
        )
        
        breakdown = compute_lift_breakdown(
            state, state.tilt, V_T_current, alpha_current, total_thrust,
            wing_area, mass, prop_radius=prop_dia * 0.0254 / 2.0
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
        log["tilt"].append(np.rad2deg(state.tilt))
        log["tilt_cmd"].append(np.rad2deg(r["tilt"]))
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
        
        if state.vel[0] > 2.0 and state.drag > 0.1:
            ld = state.lift / state.drag
        else:
            ld = 0.0
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
        "mode": mission_mode,
        "plan": flight_plan,
        "waypoints": waypoint_rows,
        "mission_complete": mission_manager.sequencer.complete,
        "termination_reason": termination_reason,
        "elapsed_time": float(log["t"][-1]) if log["t"] else 0.0,
    }
    return {k: np.array(v) for k, v in log.items()}, info

with st.spinner("Running simulation..."):
    data, gust_info = run_simulation(
        mass, mass_fuselage, mass_wing, mass_vtail, mass_motor,
        battery_cells, battery_mah, battery_c_rating, motor_kv,
        prop_dia, prop_pitch, num_blades, k_thrust, max_omega,
        wing_area, vtail_area, vtail_arm, root_x, root_c, tip_c, 2.0 * tip_y,
        sweep, tuple(tuple(position) for position in motor_positions),
        alt_kp, alt_ki, alt_kd, spd_kp, spd_ki, spd_kd,
        att_kp, att_ki, att_kd, rate_kp, rate_ki, rate_kd,
        gust_enabled, gust_start, gust_duration, gust_mag, gust_dir,
        max_tilt_rate_deg_s, flight_plan, mission_mode, manual_altitude,
        manual_speed, manual_tilt_deg, waypoint_rows
    )
if data["t"].size == 0:
    st.error("The simulation ended before producing a valid flight sample.")
    st.stop()

if wind_enabled and gust_info["mag"] > 0.0:
    st.info(
        f"Wind event: {gust_info['dir']} at {gust_info['mag']:.1f} m/s, "
        f"starting at t={gust_info['start']:.1f}s for {gust_info['dur']:.1f}s."
    )

col1, col2, col3, col4 = st.columns(4)
max_altitude_deviation = float(np.max(np.maximum(data["ref_alt"] - data["alt"], 0.0)))
col1.metric("Max altitude shortfall", f"{max_altitude_deviation:.2f} m")
col2.metric("Max Pitch", f"{np.max(np.abs(data['pitch'])):.2f} deg")
col3.metric("Peak Gust", f"{np.max(data['wind']):.2f} m/s")
col4.metric("Final Alt", f"{data['alt'][-1]:.1f} m")

mode_col, plan_col, phase_col, route_col, duration_col = st.columns(5)
mode_col.metric("Mission mode", gust_info["mode"])
plan_col.metric("Flight plan", gust_info["plan"])
phase_col.metric("Current phase", str(data["phase"][-1]))
if mission_mode != "AUTO":
    route_status = mission_mode.title()
elif flight_plan == "Waypoint route":
    route_status = "Complete" if gust_info["mission_complete"] else (
        f"Waypoint {int(data['mission_index'][-1]) + 1}"
    )
else:
    route_status = "Timed"
route_col.metric("Route status", route_status)
duration_col.metric("Simulated duration", f"{gust_info['elapsed_time']:.2f} s")
if gust_info["termination_reason"] != "Mission duration completed":
    st.warning(
        f"Simulation stopped at t={gust_info['elapsed_time']:.2f} s: "
        f"{gust_info['termination_reason']}."
    )

st.markdown("---")

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

# Phase boundaries for visualization
mission = MissionProfile()
phase_colors = ['#B0E0E6', '#FFB6C1', '#FFD700']
mission_phases = (
    [("HOVER", 0.0, mission.hover_time),
     ("TRANSITION", mission.hover_time, mission.transition_duration),
     ("CRUISE", mission.total_time, 63.0 - mission.total_time)]
    if flight_plan == "Timed transition" and mission_mode == "AUTO"
    else []
)

def add_phase_background(ax):
    for (name, start, dur, *_), color in zip(mission_phases, phase_colors):
        ax.axvspan(start, start + dur, alpha=0.15, color=color)

# 1. Altitude
axes[0, 0].plot(data["t"], data["alt"], 'b-', linewidth=2, label="Altitude")
axes[0, 0].plot(data["t"], data["ref_alt"], 'k--', linewidth=1.5, label="Reference")
add_phase_background(axes[0, 0])
axes[0, 0].set_title("Altitude (Full Mission)")
axes[0, 0].set_ylabel("m")
axes[0, 0].grid(True)
axes[0, 0].legend()

# 2. Pitch
axes[0, 1].plot(data["t"], data["pitch"], 'r-', linewidth=2, label="Actual")
axes[0, 1].plot(data["t"], data["desired_pitch"], 'k--', linewidth=1.5, label="Desired")
add_phase_background(axes[0, 1])
axes[0, 1].set_title("Pitch")
axes[0, 1].set_ylabel("deg")
axes[0, 1].grid(True)
axes[0, 1].legend()

# 3. Roll
axes[1, 0].plot(data["t"], data["roll"], 'orange', linewidth=2)
add_phase_background(axes[1, 0])
axes[1, 0].axhline(0, color='k', linestyle='--')
axes[1, 0].set_title("Roll")
axes[1, 0].set_ylabel("deg")
axes[1, 0].grid(True)

# 4. Yaw
axes[1, 1].plot(data["t"], data["yaw"], 'purple', linewidth=2)
add_phase_background(axes[1, 1])
axes[1, 1].axhline(0, color='k', linestyle='--')
axes[1, 1].set_title("Yaw")
axes[1, 1].set_ylabel("deg")
axes[1, 1].grid(True)

# 5. Wind
axes[2, 0].plot(data["t"], data["wind"], 'g-', linewidth=2)
add_phase_background(axes[2, 0])
axes[2, 0].set_title("Wind Gust")
axes[2, 0].set_ylabel("m/s")
axes[2, 0].grid(True)

# 6. Rotor Tilt
axes[2, 1].plot(data["t"], data["tilt"], 'm-', linewidth=2)
add_phase_background(axes[2, 1])
axes[2, 1].set_title("Rotor Tilt (Mission)")
axes[2, 1].set_ylabel("deg")
axes[2, 1].grid(True)

# 7. L/D Ratio
axes[3, 0].plot(data["t"], data["ld_ratio"], 'teal', linewidth=2)
add_phase_background(axes[3, 0])
axes[3, 0].set_title("L/D Ratio")
axes[3, 0].set_ylabel("L/D")
axes[3, 0].grid(True)

# 8. Battery Current
axes[3, 1].plot(data["t"], data["current"], 'b-', linewidth=2, label="Current (A)")
axes[3, 1].axhline(y=(battery_mah/1000)*battery_c_rating, color='r', linestyle='--', label="Max C-Rating")
add_phase_background(axes[3, 1])
axes[3, 1].set_title("Battery Current Draw")
axes[3, 1].set_ylabel("Amps")
axes[3, 1].grid(True)
axes[3, 1].legend(loc='upper left')

# Add phase labels to the top
for ax in axes[0, :]:
    for name, start, dur, *_ in mission_phases:
        ax.axvline(start, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
st.pyplot(fig, alt="Full-flight altitude, attitude, wind, rotor tilt, efficiency, and battery telemetry")

st.markdown("---")
st.subheader("6-DOF equations and response")
st.caption(
    "Position and velocity are NED (north-east-down); forces and angular rates are FRD body-frame quantities."
)
st.markdown(
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
    position_col.line_chart(
        position_data,
        x="time_s",
        y=["North (m)", "East (m)", "Down (m)"],
        alt="Integrated north, east, and down position in the NED frame",
    )
    velocity_data = pd.DataFrame({
        "time_s": data["t"],
        "North (m/s)": data["vel_north"],
        "East (m/s)": data["vel_east"],
        "Down (m/s)": data["vel_down"],
    })
    velocity_col.line_chart(
        velocity_data,
        x="time_s",
        y=["North (m/s)", "East (m/s)", "Down (m/s)"],
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
    force_col.line_chart(
        force_data,
        x="time_s",
        y=["Forward X (N)", "Right Y (N)", "Down Z (N)"],
        alt="Resultant rotor and aerodynamic force resolved in the FRD body frame",
    )
    acceleration_data = pd.DataFrame({
        "time_s": data["t"],
        "North (m/s²)": data["accel_north"],
        "East (m/s²)": data["accel_east"],
        "Down (m/s²)": data["accel_down"],
    })
    acceleration_col.line_chart(
        acceleration_data,
        x="time_s",
        y=["North (m/s²)", "East (m/s²)", "Down (m/s²)"],
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
    rate_col.line_chart(
        rate_data,
        x="time_s",
        y=["p (rad/s)", "q (rad/s)", "r (rad/s)"],
        alt="Body roll, pitch, and yaw rates p, q, and r",
    )
    moment_data = pd.DataFrame({
        "time_s": data["t"],
        "Roll L (N·m)": data["moment_l_body"],
        "Pitch M (N·m)": data["moment_m_body"],
        "Yaw N (N·m)": data["moment_n_body"],
    })
    moment_col.line_chart(
        moment_data,
        x="time_s",
        y=["Roll L (N·m)", "Pitch M (N·m)", "Yaw N (N·m)"],
        alt="Applied roll, pitch, and yaw moments in the FRD body frame",
    )
    angular_acceleration_data = pd.DataFrame({
        "time_s": data["t"],
        "p-dot (rad/s²)": data["angular_accel_p"],
        "q-dot (rad/s²)": data["angular_accel_q"],
        "r-dot (rad/s²)": data["angular_accel_r"],
    })
    st.line_chart(
        angular_acceleration_data,
        x="time_s",
        y=["p-dot (rad/s²)", "q-dot (rad/s²)", "r-dot (rad/s²)"],
        alt="Angular acceleration produced by applied moments and Euler gyroscopic coupling",
    )

# Mission phase legend
if mission_phases:
    st.markdown("**Timed mission phases**")
    cols = st.columns(len(mission_phases))
    for i, (name, start, dur) in enumerate(mission_phases):
        cols[i].metric(name.title(), f"{start:.0f}–{start + dur:.0f} s")

st.markdown("---")
st.subheader("Guidance, control allocation, and mission")
st.caption(
    "The charts show the guidance references and the four motor/V-tail commands applied by the cascade controller."
)

tracking_col, mission_col = st.columns(2)
with tracking_col:
    st.markdown("**Flight-path tracking**")
    altitude_data = pd.DataFrame({
        "time_s": data["t"],
        "Altitude (m)": data["alt"],
        "Altitude target (m)": data["ref_alt"],
    })
    st.line_chart(
        altitude_data,
        x="time_s",
        y=["Altitude (m)", "Altitude target (m)"],
        alt="Aircraft altitude compared with the mission altitude target",
    )
    speed_data = pd.DataFrame({
        "time_s": data["t"],
        "North velocity (m/s)": data["speed"],
        "North velocity target (m/s)": data["ref_speed"],
    })
    st.line_chart(
        speed_data,
        x="time_s",
        y=["North velocity (m/s)", "North velocity target (m/s)"],
        alt="Northward ground velocity compared with the mission velocity target",
    )

with mission_col:
    st.markdown("**Mission route and current position**")
    if flight_plan == "Waypoint route":
        route_frame = pd.DataFrame(
            waypoint_rows,
            columns=["North (m)", "East (m)", "Altitude (m)", "Speed (m/s)", "Tilt (deg)"],
        )
        st.dataframe(route_frame, hide_index=True, alt="Configured waypoint route")
        st.caption("The current guidance model tracks altitude and forward speed; lateral waypoint tracking is not yet modeled.")
        route_plot = pd.DataFrame({
            "North (m)": data["north"],
            "East (m)": data["east"],
            "Time (s)": data["t"],
        })
        st.scatter_chart(
            route_plot,
            x="North (m)",
            y="East (m)",
            color="Time (s)",
            alt="Aircraft position through the local north-east flight path",
        )
    else:
        st.write(
            f"Mode: **{mission_mode}**  \n"
            f"Profile: **{flight_plan}**  \n"
            f"Tilt actuator limit: **{max_tilt_rate_deg_s:.1f} deg/s**"
        )

tilt_col, motor_col, surface_col = st.columns(3)
with tilt_col:
    st.markdown("**Tilt actuator tracking**")
    tilt_data = pd.DataFrame({
        "time_s": data["t"],
        "Actual tilt (deg)": data["tilt"],
        "Tilt command (deg)": data["tilt_cmd"],
    })
    st.line_chart(
        tilt_data,
        x="time_s",
        y=["Actual tilt (deg)", "Tilt command (deg)"],
        alt="Actual nacelle tilt tracking its commanded angle",
    )

with motor_col:
    st.markdown("**Per-motor command (RPM)**")
    motor_data = pd.DataFrame({
        "time_s": data["t"],
        "Front left": data["omega_fl"],
        "Front right": data["omega_fr"],
        "Rear left": data["omega_rl"],
        "Rear right": data["omega_rr"],
    })
    st.line_chart(
        motor_data,
        x="time_s",
        y=["Front left", "Front right", "Rear left", "Rear right"],
        alt="Commanded RPM for each of the four motors",
    )

with surface_col:
    st.markdown("**V-tail command (deg)**")
    surface_data = pd.DataFrame({
        "time_s": data["t"],
        "Left surface": np.rad2deg(data["delta_left"]),
        "Right surface": np.rad2deg(data["delta_right"]),
    })
    st.line_chart(
        surface_data,
        x="time_s",
        y=["Left surface", "Right surface"],
        alt="Left and right V-tail control surface commands",
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
st.markdown("---")
st.subheader(" Wind Triangle (Live)")

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
    ax_wt.set_ylabel("Downward velocity w (m/s)")
    ax_wt.set_title(f"Airspeed Vector in Body Frame\n(α = {np.rad2deg(np.arctan2(w_airspeed, u_airspeed)):.1f}°)")
    ax_wt.grid(True)
    ax_wt.legend()
    ax_wt.set_aspect('equal', adjustable='datalim')
    st.pyplot(fig_wt, alt="Body-frame airspeed vector showing forward and vertical airspeed")

with c2:
    # Angle of Attack (alpha) over time
    alpha_history = np.rad2deg(np.arctan2(np.array(data["w_body"]), np.array(data["u_body"])))
    
    fig_alpha, ax_alpha = plt.subplots(figsize=(8, 6))
    ax_alpha.plot(data["t"], alpha_history, 'b-', linewidth=2)
    ax_alpha.axhline(15, color='r', linestyle='--', alpha=0.5, label="Stall angle (15°)")
    ax_alpha.axhline(-15, color='r', linestyle='--', alpha=0.5)
    ax_alpha.set_xlabel("Time (s)")
    ax_alpha.set_ylabel("Angle of Attack α (deg)")
    ax_alpha.set_title("Angle of Attack during Flight")
    ax_alpha.grid(True)
    ax_alpha.legend()
    st.pyplot(fig_alpha, alt="Angle of attack through the flight with stall-angle references")

# Show the instantaneous values
st.markdown("**Instantaneous Wind Triangle**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("True Airspeed", f"{data['V_T'][-1]:.2f} m/s")
c2.metric("u (Forward Airspeed)", f"{data['u_body'][-1]:.2f} m/s")
c3.metric("w (Downward Airspeed)", f"{data['w_body'][-1]:.2f} m/s")
alpha_now = np.rad2deg(np.arctan2(data['w_body'][-1], data['u_body'][-1])) if data['u_body'][-1] != 0 else 0
c4.metric("Angle of Attack α", f"{alpha_now:.1f}°")

# GEOMETRY
st.markdown("---")
st.subheader(" Aircraft Geometry & Mass Properties")
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
        ax_side.plot(pos[0], -pos[2], 'ko', markersize=10)
    
    # Plot CG, Wing AC, and Neutral Point
    ax_side.plot(props.cg[0], -props.cg[2], 'bo', markersize=12, label="CG")
    ax_side.plot(props.ac_wing_x, -props.ac_z, 'go', markersize=10, label="Wing AC")
    ax_side.plot(props.neutral_point_x, -props.ac_z, 'r', marker='X', markersize=12, label="Neutral Point")
    
    # Plot Wing and Tail
    ax_side.plot([props.le_root_x, props.le_root_x + props.root_chord], [0, 0], 'k-', linewidth=4, label="Wing (Root)")
    ax_side.plot([-props.cg[0] - props.vtail_arm, -props.cg[0] - props.vtail_arm], [0, -0.2], 'purple', linewidth=4, label="V-Tail")
    
    ax_side.set_xlim(-1.0, 1.0); ax_side.set_ylim(-0.5, 0.5)
    ax_side.invert_yaxis(); ax_side.set_title("Side Profile: Neutral Point & AC"); ax_side.grid(True, alpha=0.3); ax_side.legend()
    st.pyplot(fig_side, alt="Aircraft side profile showing the center of gravity, aerodynamic center, and neutral point")

with col_geo2:
    fig_top, ax_top = plt.subplots(figsize=(8, 4))
    for pos in motor_positions:
        color = 'tab:blue' if pos[3] else 'gray'
        ax_top.plot(pos[0], pos[1], 'o', color=color, markersize=12)
    ax_top.plot([props.ac_wing_x, props.ac_wing_x], [-calculated_wingspan/2, calculated_wingspan/2], 'k-', linewidth=4, label="Wing")
    ax_top.plot(props.cg[0], props.cg[1], 'bo', markersize=12, label="CG")
    ax_top.set_xlim(-1.0, 1.0)
    ax_top.set_ylim(-calculated_wingspan, calculated_wingspan)
    ax_top.set_title("Top Profile (X-Y)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend()
    st.pyplot(fig_top, alt="Aircraft top profile showing wing span, center of gravity, and motor positions")
    # ==========================================================
# LIFT HANDOFF & CONTROL AUTHORITY
# ==========================================================
st.markdown("---")
st.subheader("🎯 Lift Handoff & Control Authority")

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
    st.pyplot(fig_lift, alt="Rotor, wing, and rotor-wash contributions to vertical lift through transition")

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
    st.pyplot(fig_ratio, alt="Lift handoff ratio from rotor-dominated to wing-dominated support")

# Control authority at current instant
st.markdown("### Control Authority at Current Instant")
current_tilt = data["tilt"][-1]
current_V_T = data["V_T"][-1]
if current_V_T < 0.5:
    current_V_T = 0.5

from src.aero.handoff import control_authority, control_blend
auth = control_authority(np.deg2rad(current_tilt), current_V_T, 12.0)
blend = control_blend(np.deg2rad(current_tilt))

c1, c2, c3 = st.columns(3)
c1.metric("Rotor Pitch Authority", f"{auth['rotor_authority_frac']*100:.0f}%")
c2.metric("V-Tail Pitch Authority", f"{auth['vtail_authority_frac']*100:.0f}%")
c3.metric("Controller Blend", f"{blend*100:.0f}% (0=rotor, 100=wing)")

# Control authority vs tilt angle (the "handoff map")
st.markdown("### Control Authority Map")
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
st.pyplot(fig_map, alt="Rotor and V-tail control authority and blend versus nacelle tilt")


# Calculate static margin percentage
sm_pct = props.get_static_margin_pct()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Mass", f"{props.total_mass:.2f} kg")
c2.metric("Wing Loading", f"{props.total_mass / wing_area:.1f} kg/m²")
c3.metric("Static Margin", f"{sm_pct:.1f} %")

# Condition logic based on aerospace standards
if sm_pct < 0.0:
    c4.error("🔴 UNSTABLE: Pitch divergence")
    st.error("🚨 **CRITICAL WARNING:** Static Margin is negative. The Aerodynamic Center is *ahead* of the CG. The drone will flip backwards during transition.")
elif sm_pct > 20.0:
    c4.warning("🟡 OVER-STABLE: Too sluggish")
    st.warning("⚠️ **WARNING:** Static Margin > 20%. The aircraft is too stable. It will refuse to pitch and require massive V-tail deflection to transition. This will cause aerodynamic drag and power loss.")
else:
    c4.success(f"🟢 STABLE ({sm_pct:.1f}%)")
    st.success("✅ **SAFE:** Static Margin is within the optimal 0-20% range. Pitch response will be predictable during transition.")

st.markdown("---")
st.subheader("3x3 Inertia Tensor & Mass Properties")
st.markdown("This matrix describes the aircraft's resistance to rotation. Cross-coupling terms ($I_{xz}$) appear when the mass is asymmetric.")

# Calculate the current Inertia Matrix at hover (0 deg tilt)
current_I = props.calculate_inertia(0.0)

# Convert to a displayable format
I_display = np.round(current_I, 4)

# Display as columns
c1, c2, c3 = st.columns(3)
c1.metric("Ixx (Roll)", f"{I_display[0,0]:.4f} kg·m²")
c2.metric("Iyy (Pitch)", f"{I_display[1,1]:.4f} kg·m²")
c3.metric("Izz (Yaw)", f"{I_display[2,2]:.4f} kg·m²")

st.write("Full 3x3 Matrix:")
st.write(I_display)

# Dynamic Inertia Visualization
st.markdown("**Dynamic Inertia during Tilt Transition**")
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
st.pyplot(fig_I, alt="Roll, pitch, and yaw inertia values as nacelles tilt")

# ==========================================================
# 3D COORDINATE FRAME & CAD MODEL VISUALIZATION
# ==========================================================
import trimesh

st.markdown("---")
st.subheader("3D Aircraft & Coordinate Systems")
st.markdown("The Earth frame is fixed (Red/Green/Blue). The UAM rotates based on the flight telemetry.")

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
        st.success("✅ CAD Model loaded and centered successfully!")
    except Exception as e:
        st.error(f"Error loading STL: {e}")
else:
    st.warning(f"⚠️ CAD model not found at {stl_path}")

# --- Layout ---
fig_3d.update_layout(
    scene=dict(
        xaxis_title='North (X)',
        yaxis_title='East (Y)',
        zaxis_title='Down (Z)',
        xaxis=dict(range=[-2, 2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        yaxis=dict(range=[-2, 2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        zaxis=dict(range=[-2, 2], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig_3d, width="stretch", alt="Three-dimensional aircraft model with the earth and body coordinate frames")
# ==========================================================
# ==========================================================
# 3D CAD SKELETON WITH CUSTOM AIRFOILS & CONTROL SURFACES
# ==========================================================
from src.aero.airfoil import generate_naca4, parse_airfoil_file

st.markdown("---")
st.subheader(" 3D CAD Geometry & Control Surfaces")

# --- AIRFOIL SELECTION ---
st.markdown("### 1. Airfoil Selection")
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Main Wing**")
    wing_af_type = st.selectbox("Type", ["NACA 0012", "Custom Upload"], key="w_type")
    if wing_af_type == "NACA 0012":
        w_coords = generate_naca4(m=0.0, p=0.4, t=0.12)
    else:
        w_file = st.file_uploader("Upload Wing .dat", type=["dat", "txt"], key="w_up")
        if w_file is not None:
            w_coords = parse_airfoil_file(w_file.getvalue())
        else:
            w_coords = generate_naca4(m=0.0, p=0.4, t=0.12)

with c2:
    st.markdown("**V-Tail**")
    vt_af_type = st.selectbox("Type", ["NACA 0012", "Custom Upload"], key="vt_type")
    if vt_af_type == "NACA 0012":
        vt_coords = generate_naca4(m=0.0, p=0.4, t=0.12)
    else:
        vt_file = st.file_uploader("Upload V-Tail .dat", type=["dat", "txt"], key="vt_up")
        if vt_file is not None:
            vt_coords = parse_airfoil_file(vt_file.getvalue())
        else:
            vt_coords = generate_naca4(m=0.0, p=0.4, t=0.12)

# --- CONTROL SURFACES ---
st.markdown("### 2. Control Surface Parameters")
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
        xaxis_title='X (Fwd)', yaxis_title='Y (Right)', zaxis_title='Z (Up)',
        xaxis=dict(range=[-1.0, 1.0], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        yaxis=dict(range=[-1.5, 1.5], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        zaxis=dict(range=[-0.5, 0.5], backgroundcolor="rgba(240, 240, 240, 0.5)"),
        aspectmode='manual', aspectratio=dict(x=1, y=1.5, z=0.5),
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)) 
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=600
)

st.plotly_chart(fig_cad, width="stretch", alt="Three-dimensional wing, V-tail, control surfaces, motors, and thrust directions")

