
# Cache Buster: 2026-11-11-00-37-15
"""KADA UAM Aircraft Transition FlightConfigurator"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

st.set_page_config(page_title="Ultimate Transition phase Configurator", layout="wide")
st.title(" KADA UAM Aircraft Transition phase Configurator")

st.sidebar.title(" Parameters")
side_tabs = st.sidebar.tabs(["Body", "Prop", "Geom", "Aero", "PID", "Gust"])

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
    
    st.info(f"Calculated \(k_{{thrust}}\): **{k_thrust:.2e}**")

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
@st.cache_data
def run_simulation(mass, mass_fuselage, mass_wing, mass_vtail, mass_motor,
                   battery_cells, battery_mah, battery_c_rating, motor_kv, prop_dia, prop_pitch, num_blades,
                   k_thrust, max_omega, wing_area, vtail_area, vtail_arm, 
                   alt_kp, alt_ki, alt_kd, spd_kp, spd_ki, spd_kd, att_kp, att_ki, att_kd, rate_kp, rate_ki, rate_kd,
                   gust_enabled, random_gust, g_start, g_dur, g_mag, g_dir):
    dt, duration = 0.01, 15.0
    
    rotor = RotorModel(k_thrust=k_thrust, max_omega=max_omega)
    vtail = VtailModel(area=vtail_area, arm_length=vtail_arm)
    
    # Pass the physical prop size into the aerodynamics engine for rotor wash
    prop_diameter_m = prop_dia * 0.0254
    aero = AeroModel(wing_area=wing_area, vtail=vtail, prop_diameter_m=prop_diameter_m)
    
    props = MassProperties(cfg_path)
    props.mass_fuselage = mass_fuselage
    props.mass_wing = mass_wing
    props.mass_vtail = mass_vtail
    props.mass_motor = mass_motor
    props.total_mass = mass
    
    dyn = TiltrotorDynamics(mass=mass, rotor=rotor, mass_props=props, aero=aero)
    
    gains = {
        'alt_pid': {'kp': alt_kp, 'ki': alt_ki, 'kd': alt_kd},
        'speed_pid': {'kp': spd_kp, 'ki': spd_ki, 'kd': spd_kd},
        'att_pid': {'kp': att_kp, 'ki': att_ki, 'kd': att_kd},
        'rate_pid': {'kp': rate_kp, 'ki': rate_ki, 'kd': rate_kd}
    }
    controller = FullCascadePID(mass, k_thrust, gains)
    ref = ReferenceTrajectory("configs/robust_corridor.csv", speed_max=12.0)
    
    if random_gust:
        g_start = np.random.uniform(2.0, 12.0)
        g_dur = np.random.uniform(1.0, 4.0)
        g_mag = np.random.uniform(2.0, 10.0)
        g_dir = np.random.choice(["Front", "Back", "Left", "Right", "Up", "Down", "Diagonal"])
    gust = DiscreteGust(g_start, g_dur, g_mag, g_dir) if gust_enabled else DiscreteGust(0, 0.01, 0.0, "Front")

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    
    log = {"t": [], "alt": [], "ref_alt": [], "pitch": [], "desired_pitch": [], "wind": [], "tilt": [], "roll": [], "yaw": [], "ld_ratio": [], "current": [], "voltage": []}
    
    battery_v_max = battery_cells * 4.2
    max_current = (battery_mah / 1000) * battery_c_rating
    
    for i in range(int(duration/dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(state, r["altitude"], r["speed"], r["tilt"], dt)
        
        wind_vec = gust.get_wind(t)
        state = dyn.step(state, wind_vec, cmd["omega_fl"], cmd["omega_fr"], cmd["omega_rl"], cmd["omega_rr"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
        
        if abs(np.rad2deg(state.att[1])) > 60.0:
            break
            
        # Battery Calculation
        total_current = cmd["omega_fl"] / motor_kv + cmd["omega_fr"] / motor_kv + cmd["omega_rl"] / motor_kv + cmd["omega_rr"] / motor_kv
        voltage_sag = 2.0 if total_current > max_current else 0.0
        current_voltage = battery_v_max - voltage_sag
        
        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["ref_alt"].append(r["altitude"])
        log["pitch"].append(np.rad2deg(state.att[1]))
        log["desired_pitch"].append(np.rad2deg(cmd["desired_pitch"]))
        log["wind"].append(np.linalg.norm(wind_vec))
        log["tilt"].append(np.rad2deg(r["tilt"]))
        log["roll"].append(np.rad2deg(state.att[0]))
        log["yaw"].append(np.rad2deg(state.att[2]))
        
        if state.vel[0] > 2.0 and state.drag > 0.1:
            ld = state.lift / state.drag
        else:
            ld = 0.0
        log["ld_ratio"].append(ld)
        log["current"].append(total_current)
        log["voltage"].append(current_voltage)
        
    info = {"start": g_start, "dur": g_dur, "mag": g_mag, "dir": g_dir}
    return {k: np.array(v) for k, v in log.items()}, info

with st.spinner("Running Ultimate simulation..."):
       data, gust_info = run_simulation(mass, mass_fuselage, mass_wing, mass_vtail, mass_motor,
                                 battery_cells, battery_mah, battery_c_rating, motor_kv, prop_dia, prop_pitch, num_blades,
                                 k_thrust, max_omega, wing_area, vtail_area, vtail_arm, 
                                 alt_kp, alt_ki, alt_kd, spd_kp, spd_ki, spd_kd, att_kp, att_ki, att_kd, rate_kp, rate_ki, rate_kd,
                                 gust_enabled, random_gust, gust_start, gust_duration, gust_mag, gust_dir)
if gust_enabled:
    st.info(f"**Gust Event:** {gust_info['dir']} at {gust_info['mag']:.1f} m/s, starting at t={gust_info['start']:.1f}s for {gust_info['dur']:.1f}s.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Max Alt Loss", f"{100 - data['alt'].min():.2f} m")
col2.metric("Max Pitch", f"{np.max(np.abs(data['pitch'])):.2f} deg")
col3.metric("Peak Gust", f"{np.max(data['wind']):.2f} m/s")
col4.metric("Final Alt", f"{data['alt'][-1]:.1f} m")

st.markdown("---")

fig, axes = plt.subplots(4, 2, figsize=(16, 18))

axes[0, 0].plot(data["t"], data["alt"], 'b-', linewidth=2); axes[0, 0].plot(data["t"], data["ref_alt"], 'k--'); axes[0, 0].set_title("Altitude"); axes[0, 0].grid(True)
axes[0, 1].plot(data["t"], data["pitch"], 'r-', linewidth=2); axes[0, 1].plot(data["t"], data["desired_pitch"], 'k--'); axes[0, 1].set_title("Pitch"); axes[0, 1].grid(True)
axes[1, 0].plot(data["t"], data["roll"], 'orange', linewidth=2); axes[1, 0].axhline(0, color='k', linestyle='--'); axes[1, 0].set_title("Roll"); axes[1, 0].grid(True)
axes[1, 1].plot(data["t"], data["yaw"], 'purple', linewidth=2); axes[1, 1].axhline(0, color='k', linestyle='--'); axes[1, 1].set_title("Yaw"); axes[1, 1].grid(True)
axes[2, 0].plot(data["t"], data["wind"], 'g-', linewidth=2); axes[2, 0].set_title("Wind Gust"); axes[2, 0].grid(True)
axes[2, 1].plot(data["t"], data["tilt"], 'm-', linewidth=2); axes[2, 1].set_title("Rotor Tilt"); axes[2, 1].grid(True)
axes[3, 0].plot(data["t"], data["ld_ratio"], 'teal', linewidth=2); axes[3, 0].set_title("L/D Ratio"); axes[3, 0].grid(True)
# Replace axes[3, 1].axis('off') with:
ax_batt = axes[3, 1]
ax_batt.plot(data["t"], data["current"], 'b-', label="Current (A)")
ax_batt.axhline(y=(battery_mah/1000)*battery_c_rating, color='r', linestyle='--', label="Max C-Rating")
ax_batt.set_title("Battery Current Draw")
ax_batt.set_ylabel("Amps")
ax_batt.grid(True)
ax_batt.legend(loc='upper left')

plt.tight_layout()
st.pyplot(fig)

# Instantiate for the plots
props_plot = MassProperties(cfg_path)
props_plot.mass_fuselage = mass_fuselage
props_plot.mass_wing = mass_wing
props_plot.mass_vtail = mass_vtail
props_plot.mass_motor = mass_motor
props_plot.total_mass = mass

# GEOMETRY
st.markdown("---")
st.subheader(" Aircraft Geometry & Mass Properties")
col_geo1, col_geo2 = st.columns(2)

# 1. Instantiate MassProperties
props = MassProperties(cfg_path)

# 2. Override with the live dashboard variables from the Aero tab
# Calculate wingspan from the XFLR5 coordinates
calculated_wingspan = tip_y * 2.0
props.wingspan = calculated_wingspan
props.root_chord = root_c
props.tip_chord = tip_c
props.sweep_deg = sweep

props.vtail_area = vtail_area
props.vtail_arm = vtail_arm
props.mac, props.ac_wing_x = props._calculate_mac_and_ac()
props.neutral_point_x = props._calculate_neutral_point()
props.mac, props.ac_wing_x = props._calculate_mac_and_ac()
props.total_mass = mass + (4 * 0.15)
props.cg = np.array([0.0, 0.0, 0.0]) 

# 4. Set the mass and CG
props.total_mass = mass + (4 * 0.15)
props.cg = np.array([0.0, 0.0, 0.0]) 

with col_geo1:
    fig_side, ax_side = plt.subplots(figsize=(8, 4))
    ax_side.axhline(0, color='brown', linestyle='--', linewidth=1, alpha=0.5)
    for pos in motor_positions:
        ax_side.plot(pos[0], -pos[2], 'ko', markersize=10)
    
    # Plot CG, Wing AC, and Neutral Point
    ax_side.plot(props.cg[0], -props.cg[2], 'bo', markersize=12, label="CG")
    ax_side.plot(props.ac_wing_x, -props.ac_z, 'go', markersize=10, label="Wing AC")
    ax_side.plot(props.neutral_point_x, -props.ac_z, 'ro', marker='X', markersize=12, label="Neutral Point")
    
    # Plot Wing and Tail
    ax_side.plot([props.le_root_x, props.le_root_x + props.root_chord], [0, 0], 'k-', linewidth=4, label="Wing (Root)")
    ax_side.plot([-props.cg[0] - props.vtail_arm, -props.cg[0] - props.vtail_arm], [0, -0.2], 'purple', linewidth=4, label="V-Tail")
    
    ax_side.set_xlim(-1.0, 1.0); ax_side.set_ylim(-0.5, 0.5)
    ax_side.invert_yaxis(); ax_side.set_title("Side Profile: Neutral Point & AC"); ax_side.grid(True, alpha=0.3); ax_side.legend()
    st.pyplot(fig_side)

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
    st.pyplot(fig_top)

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
st.markdown("This matrix describes the aircraft's resistance to rotation. Cross-coupling terms (\(I_{xz}\)) appear when the mass is asymmetric.")

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
st.pyplot(fig_I)

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

st.plotly_chart(fig_3d, use_container_width=True)
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

st.plotly_chart(fig_cad, use_container_width=True)