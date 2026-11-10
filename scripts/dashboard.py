
# Cache Buster: 2026-11-11-00-37-15
"""KADA UAM Aircraft Transition FlightConfigurator."""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import plotly.graph_objects as go


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path = os.path.join(base_dir, "configs", "blue_uam.yaml")
sys.path.append(base_dir)

from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.trajectory import ReferenceTrajectory
from src.control.pid import PIDController
from src.dynamics.mass_properties import MassProperties
from src.dynamics.frames import rotation_body_to_earth 

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

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
    mass = st.number_input("Mass (kg)", value=float(cfg.get('mass', 5.0)), step=0.1)
    i_xx = st.number_input("Ixx", value=float(cfg.get('inertia_xx', 0.15)), step=0.01)
    i_yy = st.number_input("Iyy", value=float(cfg.get('inertia_yy', 0.25)), step=0.01)
    i_zz = st.number_input("Izz", value=float(cfg.get('inertia_zz', 0.40)), step=0.01)

with side_tabs[1]:
    battery_cells = st.number_input("Battery (S)", value=6, step=1)
    motor_kv = st.number_input("Motor KV", value=920, step=50)
    prop_dia = st.number_input("Prop Dia (in)", value=12.0, step=0.5)
    battery_v = battery_cells * 3.7
    max_rpm = motor_kv * battery_v * 0.85
    max_omega = max_rpm * 2 * np.pi / 60
    k_thrust = 1e-5 * (prop_dia / 12.0) ** 4

with side_tabs[2]:
    def_motors = cfg.get('motor_positions', [[0.3, 0.4, 0.0, True], [0.3, -0.4, 0.0, True], [-0.3, 0.4, 0.0, False], [-0.3, -0.4, 0.0, False]])
    motor_positions = []
    for i in range(4):
        st.markdown(f"M{i+1}")
        c1, c2, c3, c4 = st.columns(4)
        mx = c1.number_input(f"X", value=float(def_motors[i][0]), step=0.05, key=f"mx{i}")
        my = c2.number_input(f"Y", value=float(def_motors[i][1]), step=0.05, key=f"my{i}")
        mz = c3.number_input(f"Z", value=float(def_motors[i][2]), step=0.01, key=f"mz{i}")
        tilts = c4.checkbox(f"Tilt", value=bool(def_motors[i][3]), key=f"tilt{i}")
        motor_positions.append([mx, my, mz, tilts])

with side_tabs[3]:
    wingspan = st.number_input("Wingspan (m)", value=2.0, step=0.1)
    chord = st.number_input("Chord (m)", value=0.4, step=0.05)
    wing_area = wingspan * chord * 1.5 
    vtail_area = st.number_input("V-Tail Area", value=0.1, step=0.01)
    vtail_arm = st.number_input("V-Tail Arm (m)", value=0.5, step=0.05)

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
def run_simulation(mass, i_xx, i_yy, i_zz, k_thrust, max_omega, wing_area, vtail_area, vtail_arm,
                    alt_kp, alt_ki, alt_kd, spd_kp, spd_ki, spd_kd, att_kp, att_ki, att_kd, rate_kp, rate_ki, rate_kd,
                    gust_enabled, random_gust, g_start, g_dur, g_mag, g_dir):
    dt, duration = 0.01, 15.0
    rotor = RotorModel(k_thrust=k_thrust, max_omega=max_omega)
    vtail = VtailModel(area=vtail_area, arm_length=vtail_arm)
    aero = AeroModel(wing_area=wing_area, vtail=vtail)
    dyn = TiltrotorDynamics(mass=mass, rotor=rotor, inertia=np.array([i_xx, i_yy, i_zz]), aero=aero)
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
    log = {"t": [], "alt": [], "ref_alt": [], "pitch": [], "desired_pitch": [], "wind": [], "tilt": [], "roll": [], "yaw": [], "ld_ratio": []}
    
    for i in range(int(duration/dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(state, r["altitude"], r["speed"], r["tilt"], dt)
        
        wind_vec = gust.get_wind(t)
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd["omega_fl"], cmd["omega_fr"], cmd["omega_rl"], cmd["omega_rr"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
        state.vel = state.vel - wind_vec
        
        if abs(np.rad2deg(state.att[1])) > 60.0:
            break
            
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
        
    info = {"start": g_start, "dur": g_dur, "mag": g_mag, "dir": g_dir}
    return {k: np.array(v) for k, v in log.items()}, info

with st.spinner("Running Ultimate simulation..."):
    data, gust_info = run_simulation(mass, i_xx, i_yy, i_zz, k_thrust, max_omega, wing_area, vtail_area, vtail_arm,
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
axes[3, 1].axis('off')

plt.tight_layout()
st.pyplot(fig)

# GEOMETRY
st.markdown("---")
st.subheader(" Aircraft Geometry & Mass Properties")
col_geo1, col_geo2 = st.columns(2)
props = MassProperties(cfg_path)
props.total_mass = mass + (4 * 0.15)
props.ac_x = 0.1 + (0.25 * chord)
props.cg = np.array([0.0, 0.0, 0.0]) 

with col_geo1:
    fig_side, ax_side = plt.subplots(figsize=(8, 4))
    ax_side.axhline(0, color='brown', linestyle='--', linewidth=1, alpha=0.5)
    for pos in motor_positions:
        ax_side.plot(pos[0], -pos[2], 'ko', markersize=10)
    ax_side.plot(props.cg[0], -props.cg[2], 'bo', markersize=12, label="CG")
    ax_side.plot(props.ac_x, -props.ac_z, 'go', markersize=12, label="AC")
    ax_side.plot([props.ac_x - (chord/2), props.ac_x + (chord/2)], [0, 0], 'k-', linewidth=4, label="Wing")
    ax_side.plot([-0.5, -0.5], [0, -0.2], 'purple', linewidth=4, label="V-Tail")
    ax_side.set_xlim(-1.0, 1.0); ax_side.set_ylim(-0.5, 0.5)
    ax_side.invert_yaxis(); ax_side.set_title("Side Profile (X-Z)"); ax_side.grid(True, alpha=0.3); ax_side.legend()
    st.pyplot(fig_side)

with col_geo2:
    fig_top, ax_top = plt.subplots(figsize=(8, 4))
    for pos in motor_positions:
        color = 'tab:blue' if pos[3] else 'gray'
        ax_top.plot(pos[0], pos[1], 'o', color=color, markersize=12)
    ax_top.plot([props.ac_x, props.ac_x], [-wingspan/2, wingspan/2], 'k-', linewidth=4, label="Wing")
    ax_top.plot(props.cg[0], props.cg[1], 'bo', markersize=12, label="CG")
    ax_top.set_xlim(-1.0, 1.0); ax_top.set_ylim(-wingspan, wingspan)
    ax_top.set_title("Top Profile (X-Y)"); ax_top.grid(True, alpha=0.3); ax_top.legend()
    st.pyplot(fig_top)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Mass", f"{props.total_mass:.2f} kg")
c2.metric("Static Margin", f"{props.get_static_margin():.3f} m")
c3.metric("Stability", "STABLE" if props.get_static_margin() > 0 else "UNSTABLE")
c4.metric("Wing Loading", f"{props.total_mass / wing_area:.1f} kg/m")

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