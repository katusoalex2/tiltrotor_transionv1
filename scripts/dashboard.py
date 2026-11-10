"""Ultimate Aircraft Configurator - Full Parameters + Intermittent Gusts."""
###### click here to open the dashboard: http://103.218.163.203:8501 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yaml
import sys
import os

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

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# --- Advanced Gust Generator ---
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

    def compute(self, state, ref_alt, ref_speed, ref_tilt, dt):
        current_alt = -state.pos[2]
        current_speed = state.vel[0]
        current_pitch = state.att[1]
        current_pitch_rate = state.omega[1]
        
        alt_error = ref_alt - current_alt
        thrust_correction = self.alt_pid.compute(0, -alt_error, dt)
        base_thrust = (self.mass * 9.81) + thrust_correction
        omega_cmd = np.sqrt(max(base_thrust, 0.0) / (4 * self.k_thrust))
        omega_cmd = np.clip(omega_cmd, 0.0, 3000.0)
        
        speed_error = ref_speed - current_speed
        pitch_from_speed = -self.speed_pid.compute(0, -speed_error, dt)
        pitch_from_alt = alt_error * 0.05 
        desired_pitch = np.clip(pitch_from_alt + pitch_from_speed, -0.26, 0.26)
        
        pitch_error = desired_pitch - current_pitch
        desired_pitch_rate = self.att_pid.compute(0, -pitch_error, dt)
        
        rate_error = desired_pitch_rate - current_pitch_rate
        vtail_cmd = -self.rate_pid.compute(0, -rate_error, dt)
        
        return {"omega": omega_cmd, "tilt": ref_tilt, "delta_left": vtail_cmd, "delta_right": vtail_cmd, 
                "desired_pitch": desired_pitch}

st.set_page_config(page_title="Ultimate Configurator", layout="wide")
st.title(" KADA Ultimate UAM Aircraft Configurator")

# ==========================================================
# LEFT SIDEBAR - 7 TABS
# ==========================================================
st.sidebar.title("??? Parameters")
side_tabs = st.sidebar.tabs(["Body", "Prop", "Geom", "Aero", "PID", "Gust"])

with side_tabs[0]:
    st.markdown("**Body & Inertia**")
    mass = st.number_input("Mass (kg)", value=float(cfg.get('mass', 5.0)), step=0.1)
    i_xx = st.number_input("Ixx (Roll)", value=float(cfg.get('inertia_xx', 0.15)), step=0.01)
    i_yy = st.number_input("Iyy (Pitch)", value=float(cfg.get('inertia_yy', 0.25)), step=0.01)
    i_zz = st.number_input("Izz (Yaw)", value=float(cfg.get('inertia_zz', 0.40)), step=0.01)

with side_tabs[1]:
    st.markdown("**Propulsion**")
    battery_cells = st.number_input("Battery (S)", value=6, step=1)
    motor_kv = st.number_input("Motor KV", value=920, step=50)
    prop_dia = st.number_input("Prop Dia (in)", value=12.0, step=0.5)
    battery_v = battery_cells * 3.7
    max_rpm = motor_kv * battery_v * 0.85
    max_omega = max_rpm * 2 * np.pi / 60
    k_thrust = 1e-5 * (prop_dia / 12.0) ** 4

with side_tabs[2]:
    st.markdown("**Motors (Geom)**")
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
    st.markdown("**Wing & V-Tail**")
    wingspan = st.number_input("Wingspan (m)", value=2.0, step=0.1)
    chord = st.number_input("Chord (m)", value=0.4, step=0.05)
    wing_area = wingspan * chord * 1.5 
    vtail_area = st.number_input("V-Tail Area", value=0.1, step=0.01)
    vtail_arm = st.number_input("V-Tail Arm (m)", value=0.5, step=0.05)

with side_tabs[4]:
    st.markdown("**Cascade PID**")
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
    st.markdown("**Gust Event**")
    gust_enabled = st.checkbox("Enable Wind Gust", value=True)
    random_gust = st.checkbox("Randomize (Monte Carlo)", value=False)
    if not random_gust:
        gust_start = st.slider("Start Time (s)", 0.0, 15.0, 5.0, 0.5)
        gust_duration = st.slider("Duration (s)", 0.5, 5.0, 2.0, 0.5)
        gust_mag = st.slider("Magnitude (m/s)", 0.0, 15.0, 6.0, 0.5)
        gust_dir = st.selectbox("Direction", ["Front", "Back", "Left", "Right", "Up", "Down", "Diagonal"])
    else:
        st.info("Gust randomized per run.")
        gust_start, gust_duration, gust_mag, gust_dir = 0, 0, 0, "Front"

# ==========================================================
# SIMULATION
# ==========================================================
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
    
    log = {"t": [], "alt": [], "ref_alt": [], "pitch": [], "desired_pitch": [], "wind": [], "tilt": []}
    
    for i in range(int(duration/dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(state, r["altitude"], r["speed"], r["tilt"], dt)
        
        wind_vec = gust.get_wind(t)
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
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
col2.metric("Max Pitch", f"{np.max(np.abs(data['pitch'])):.2f}?")
col3.metric("Peak Gust", f"{np.max(data['wind']):.2f} m/s")
col4.metric("Final Alt", f"{data['alt'][-1]:.1f} m")

st.markdown("---")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes[0, 0].plot(data["t"], data["alt"], 'b-', linewidth=2, label="Actual")
axes[0, 0].plot(data["t"], data["ref_alt"], 'k--', label="Reference")
axes[0, 0].set_title("Altitude Tracking"); axes[0, 0].grid(True); axes[0, 0].legend()

axes[0, 1].plot(data["t"], data["pitch"], 'r-', linewidth=2, label="Actual")
axes[0, 1].plot(data["t"], data["desired_pitch"], 'k--', label="Desired")
axes[0, 1].set_title("Pitch Tracking"); axes[0, 1].grid(True); axes[0, 1].legend()

axes[1, 0].plot(data["t"], data["wind"], 'g-', linewidth=2)
axes[1, 0].set_title("Wind Gust Magnitude"); axes[1, 0].grid(True)

axes[1, 1].plot(data["t"], data["tilt"], 'm-', linewidth=2)
axes[1, 1].set_title("Rotor Tilt Angle"); axes[1, 1].grid(True)

plt.tight_layout()
st.pyplot(fig)
