#!/usr/bin/env python3
"""Standalone transition analysis - generates graphs as PNG files, no dashboard."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import datetime

from src.dynamics.state import AircraftState
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.dynamics.mass_properties import MassProperties
from src.dynamics.rotor import RotorModel
from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.control.architecture import ControlLawArchitecture
from src.control.mission import MissionManager, MissionProfile, Waypoint
from src.control.transition_observer import TransitionStateObserver
from src.control.envelope_protection import EnvelopeProtection
from src.dynamics.frames import rotation_earth_to_body
from src.wind.gust import DiscreteGust

# ============================================================================
# CONFIGURATION
# ============================================================================

MASS = 5.0
WING_AREA = 0.6
WINGSPAN = 2.0
ROOT_CHORD = 0.4
TIP_CHORD = 0.25
SWEEP = 0.15
VTAIL_AREA = 0.15
VTAIL_ARM = 0.3
NUM_ROTORS = 4
PROP_DIAMETER = 0.30
PROP_PITCH = 0.10
ROTOR_KV = 950
BATTERY_CELLS = 4
BATTERY_MAH = 5000
BATTERY_C = 50

# Motor positions: (x, y, z, is_tiltable)
MOTOR_POSITIONS = [
    (0.15, 0.15, 0.05, True),   # Front-left (tiltable)
    (-0.15, 0.15, 0.05, True),  # Front-right (tiltable)
    (0.15, -0.15, 0.05, False), # Rear-left (fixed)
    (-0.15, -0.15, 0.05, False),# Rear-right (fixed)
]

# Simulation parameters
DURATION = 15.0
DT = 0.01
INITIAL_ALTITUDE = 100.0
INITIAL_SPEED = 10.0
MAX_OMEGA = 3200.0

# ============================================================================
# SETUP
# ============================================================================

output_dir = Path("simulation_results")
output_dir.mkdir(exist_ok=True)

# V-tail (must come first)
vtail = VtailModel(
    area=VTAIL_AREA,
    arm_length=VTAIL_ARM,
)

# Aircraft properties
aero = AeroModel(
    wing_area=WING_AREA,
    vtail=vtail,
)

# Rotor model
rotor = RotorModel(k_thrust=1e-5, max_omega=MAX_OMEGA)

# Dynamics - simplified initialization
dyn = TiltrotorDynamics(
    mass=MASS,
    rotor=rotor,
    inertia=np.array([0.08, 0.12, 0.14]),
    aero=aero,
)

# Control observer
observer = TransitionStateObserver(time_constant_s=0.25, cruise_speed_mps=18.0)

# Control law - simplified for now, just use cascade approach
from src.control.cascade import CascadeController
controller = CascadeController(mass=MASS, k_thrust=1e-5, hover_omega=np.sqrt(MASS * 9.81 / 1e-5))

# Mission profile: simple climb and transition
ref = MissionProfile()

# ============================================================================
# SIMULATION
# ============================================================================

print("=" * 70)
print("TILTROTOR TRANSITION ANALYSIS")
print("=" * 70)
print(f"Starting simulation ({DURATION}s duration)...")

state = AircraftState()
state.pos = np.array([0.0, 0.0, -INITIAL_ALTITUDE])
state.vel = np.array([INITIAL_SPEED, 0.0, 0.0])

log = {
    "t": [], "alt": [], "pitch": [], "tilt": [], "roll": [],
    "u_body": [], "v_body": [], "w_body": [], "V_T": [],
    "speed": [], "climb_rate": [],
    "omega_fl": [], "omega_fr": [], "omega_rl": [], "omega_rr": [],
    "delta_left": [], "delta_right": [],
}

t = 0.0
steps = 0
max_alpha = 0.0
min_alpha = 0.0

while t < DURATION:
    # Reference trajectory: climb, then transition
    if t < 5.0:
        # Climb phase
        ref_dict = {
            "altitude": INITIAL_ALTITUDE,
            "speed": INITIAL_SPEED,
            "tilt": np.deg2rad(0.0),
            "heading": 0.0,
        }
    else:
        # Transition phase: climb and tilt to cruise
        transition_progress = (t - 5.0) / (DURATION - 5.0)
        ref_dict = {
            "altitude": INITIAL_ALTITUDE + 10.0 * transition_progress,
            "speed": INITIAL_SPEED + 10.0 * transition_progress,
            "tilt": np.deg2rad(90.0 * transition_progress),
            "heading": 0.0,
        }
    
    # Compute control command
    cmd = controller.compute(t, state, ref_altitude=ref_dict["altitude"], 
                            ref_speed=ref_dict["speed"], ref_tilt=ref_dict["tilt"])
    
    # Apply tilt rate limiting
    tilt_cmd = cmd["tilt"]
    max_tilt_rate = np.deg2rad(10.0)  # 10 deg/s
    state.tilt = state.tilt + np.clip(tilt_cmd - state.tilt, -max_tilt_rate * DT, max_tilt_rate * DT)
    state.tilt = np.clip(state.tilt, 0.0, np.pi / 2.0)
    
    # Dynamics step (single omega for all rotors in this simplified version)
    state = dyn.step(
        state, 
        cmd["omega"],  # omega command
        ref_dict["tilt"],  # tilt command
        cmd.get("delta_left", 0.0),  # left control
        cmd.get("delta_right", 0.0),  # right control
        DT
    )
    
    # Calculate angle of attack
    R_e2b = rotation_earth_to_body(state.att[0], state.att[1], state.att[2])
    vel_body = R_e2b @ state.vel
    V_T = np.linalg.norm(vel_body)
    alpha_rad = np.arctan2(vel_body[2], vel_body[0])
    alpha_deg = np.rad2deg(alpha_rad)
    
    max_alpha = max(max_alpha, alpha_deg)
    min_alpha = min(min_alpha, alpha_deg)
    
    # Logging
    log["t"].append(t)
    log["alt"].append(-state.pos[2])
    log["pitch"].append(np.rad2deg(state.att[1]))
    log["tilt"].append(np.rad2deg(state.tilt))
    log["roll"].append(np.rad2deg(state.att[0]))
    log["u_body"].append(vel_body[0])
    log["v_body"].append(vel_body[1])
    log["w_body"].append(vel_body[2])
    log["V_T"].append(V_T)
    log["speed"].append(np.linalg.norm(state.vel[:2]))
    log["climb_rate"].append(-state.vel[2])
    log["omega_fl"].append(cmd["omega"])
    log["omega_fr"].append(cmd["omega"])
    log["omega_rl"].append(cmd["omega"])
    log["omega_rr"].append(cmd["omega"])
    log["delta_left"].append(cmd.get("delta_left", 0.0))
    log["delta_right"].append(cmd.get("delta_right", 0.0))
    
    t += DT
    steps += 1

print(f"Simulation complete: {steps} steps")
print(f"Max angle of attack: {max_alpha:.2f}°")
print(f"Min angle of attack: {min_alpha:.2f}°")
print(f"AoA range: [{min_alpha:.2f}°, {max_alpha:.2f}°]")

# ============================================================================
# PLOTTING
# ============================================================================

print("\nGenerating plots...")

# Convert to numpy arrays
for key in log:
    log[key] = np.array(log[key])

alpha_history = np.rad2deg(np.arctan2(log["w_body"], log["u_body"]))

# Figure 1: Angle of Attack (PRIMARY)
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(log["t"], alpha_history, 'b-', linewidth=2.5, label="Angle of Attack")
ax.axhline(15, color='r', linestyle='--', alpha=0.7, linewidth=2, label="Stall angle (15°)")
ax.axhline(-15, color='r', linestyle='--', alpha=0.7, linewidth=2)
ax.axhspan(5.0, 12.0, color='green', alpha=0.2, label="Requested target (5-12°)")
ax.axhspan(-12.0, -5.0, color='green', alpha=0.2)
ax.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Angle of Attack α (deg)", fontsize=12, fontweight='bold')
ax.set_title("Angle of Attack During Transition Flight", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')
ax.set_ylim(-25, 25)
plt.tight_layout()
plt.savefig(output_dir / "01_angle_of_attack.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 01_angle_of_attack.png")
plt.close()

# Figure 2: Pitch and Tilt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(log["t"], log["pitch"], 'b-', linewidth=2, label="Pitch")
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.set_ylabel("Pitch (deg)", fontsize=11, fontweight='bold')
ax1.set_title("Pitch Attitude During Transition", fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

ax2.plot(log["t"], log["tilt"], 'g-', linewidth=2, label="Tilt Angle")
ax2.set_xlabel("Mission time (s)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Tilt (deg)", fontsize=11, fontweight='bold')
ax2.set_title("Nacelle Tilt Angle During Transition", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(output_dir / "02_pitch_and_tilt.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 02_pitch_and_tilt.png")
plt.close()

# Figure 3: Altitude and Climb Rate
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(log["t"], log["alt"], 'purple', linewidth=2)
ax1.set_ylabel("Altitude (m)", fontsize=11, fontweight='bold')
ax1.set_title("Altitude During Transition", fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(log["t"], log["climb_rate"], 'darkblue', linewidth=2)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel("Mission time (s)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Climb Rate (m/s)", fontsize=11, fontweight='bold')
ax2.set_title("Vertical Velocity During Transition", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "03_altitude_and_climb.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 03_altitude_and_climb.png")
plt.close()

# Figure 4: Airspeed and Body Velocities
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(log["t"], log["speed"], 'b-', linewidth=2, label="Horizontal speed")
ax.plot(log["t"], log["V_T"], 'r-', linewidth=2, label="Total airspeed")
ax.plot(log["t"], np.abs(log["climb_rate"]), 'g--', linewidth=1.5, label="Vertical speed (abs)")
ax.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Speed (m/s)", fontsize=12, fontweight='bold')
ax.set_title("Airspeed During Transition", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "04_airspeed.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 04_airspeed.png")
plt.close()

# Figure 5: Body-frame Velocity Components
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(log["t"], log["u_body"], 'r-', linewidth=2, label="u (forward)")
ax.plot(log["t"], log["v_body"], 'g-', linewidth=2, label="v (lateral)")
ax.plot(log["t"], log["w_body"], 'b-', linewidth=2, label="w (vertical)")
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Velocity (m/s)", fontsize=12, fontweight='bold')
ax.set_title("Body-Frame Velocity Components", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "05_body_velocities.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 05_body_velocities.png")
plt.close()

# Figure 6: Rotor Commands
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(log["t"], log["omega_fl"], 'r-', linewidth=2, label="Front-Left (tiltable)")
ax.plot(log["t"], log["omega_fr"], 'g-', linewidth=2, label="Front-Right (tiltable)")
ax.plot(log["t"], log["omega_rl"], 'b-', linewidth=2, label="Rear-Left (fixed)")
ax.plot(log["t"], log["omega_rr"], 'm-', linewidth=2, label="Rear-Right (fixed)")
ax.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Rotor Speed (RPM)", fontsize=12, fontweight='bold')
ax.set_title("Rotor Speed Commands During Transition", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, ncol=2)
plt.tight_layout()
plt.savefig(output_dir / "06_rotor_commands.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 06_rotor_commands.png")
plt.close()

# Figure 7: Control Surface Deflections
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(log["t"], np.rad2deg(log["delta_left"]), 'b-', linewidth=2, label="Left V-tail")
ax.plot(log["t"], np.rad2deg(log["delta_right"]), 'r--', linewidth=2, label="Right V-tail")
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Deflection (deg)", fontsize=12, fontweight='bold')
ax.set_title("V-Tail Control Surface Deflections", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "07_control_surfaces.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 07_control_surfaces.png")
plt.close()

# Figure 8: 3D Flight Path (Top and Side views)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Top view
ax1.plot(log["t"], alpha_history, 'b-', linewidth=2.5)
ax1.axhline(15, color='r', linestyle='--', alpha=0.5, label="Stall limit")
ax1.axhline(-15, color='r', linestyle='--', alpha=0.5)
ax1.axhspan(5.0, 12.0, color='green', alpha=0.15, label="Target zone")
ax1.axhspan(-12.0, -5.0, color='green', alpha=0.15)
ax1.set_title("Angle of Attack Constraint Analysis", fontsize=12, fontweight='bold')
ax1.set_xlabel("Mission time (s)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Angle of Attack (deg)", fontsize=10, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_ylim(-20, 20)

# Side view
ax2_twin = ax2.twinx()
line1 = ax2.plot(log["t"], log["alt"], 'purple', linewidth=2.5, label="Altitude")
line2 = ax2_twin.plot(log["t"], log["tilt"], 'orange', linewidth=2.5, label="Tilt")
ax2.set_xlabel("Mission time (s)", fontsize=10, fontweight='bold')
ax2.set_ylabel("Altitude (m)", fontsize=10, fontweight='bold', color='purple')
ax2_twin.set_ylabel("Tilt Angle (deg)", fontsize=10, fontweight='bold', color='orange')
ax2.set_title("Altitude and Tilt Progress", fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='purple')
ax2_twin.tick_params(axis='y', labelcolor='orange')
ax2.grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='center left', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "08_summary_analysis.png", dpi=150, bbox_inches='tight')
print("  [SAVED] Saved: 08_summary_analysis.png")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 70)
print("SIMULATION SUMMARY")
print("=" * 70)

final_altitude = log["alt"][-1]
final_speed = log["speed"][-1]
final_tilt = log["tilt"][-1]

print(f"\nSimulation Results:")
print(f"  Duration: {DURATION:.1f} seconds")
print(f"  Final altitude: {final_altitude:.1f} m")
print(f"  Final speed: {final_speed:.2f} m/s")
print(f"  Final tilt: {final_tilt:.1f}°")
print(f"\nAngle of Attack Statistics:")
print(f"  Max AoA: {max_alpha:.2f}° {'[OK]' if max_alpha <= 12.0 else '[EXCEED]'}")
print(f"  Min AoA: {min_alpha:.2f}° {'[OK]' if min_alpha >= 5.0 else '[EXCEED]'}")
print(f"  Range: {max_alpha - min_alpha:.2f}°")

target_met = (max_alpha <= 12.0 and min_alpha >= 5.0)
print(f"\nTarget Envelope (5-12°): {'[SAVED] MET' if target_met else '[FAILED] NOT MET'}")

print(f"\nOutput directory: {output_dir.absolute()}")
print("=" * 70)
