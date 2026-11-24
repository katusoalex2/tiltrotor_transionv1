#!/usr/bin/env python3
"""Generate transition analysis plots directly - no dashboard needed."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.control.architecture import ControlLawArchitecture
from src.control.mission import MissionManager, MissionProfile, Waypoint
from src.control.envelope_protection import EnvelopeProtection
from src.control.transition_observer import TransitionStateObserver
from src.dynamics.mass_properties import MassProperties
from src.dynamics.frames import rotation_earth_to_body
from src.dynamics.transition import transition_speed_envelope

# Create output directory
output_dir = Path("simulation_results")
output_dir.mkdir(exist_ok=True)

# Configuration
cfg_path = os.path.join(base_dir, "configs", "blue_uam.yaml")
dt, duration = 0.01, 15.0

# Aircraft parameters
mass = 5.0
mass_fuselage = 1.5
mass_wing = 1.0
mass_vtail = 0.3
mass_motor = 0.5
wing_area = 0.6
vtail_area = 0.15
vtail_arm = 0.3
prop_dia = 12.0
motor_positions = [
    [0.3, 0.4, 0.0, True],
    [0.3, -0.4, 0.0, True],
    [-0.3, 0.4, 0.0, False],
    [-0.3, -0.4, 0.0, False],
]
k_thrust = 1e-5
max_omega = 3200.0

# Control gains (balanced for stability)
alt_kp, alt_ki, alt_kd = 0.1, 0.005, 0.1
spd_kp, spd_ki, spd_kd = 0.05, 0.01, 0.05
att_kp, att_ki, att_kd = 1.5, 0.3, 0.5
rate_kp, rate_ki, rate_kd = 0.3, 0.1, 0.1

# Initialize dynamics and control
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
props.motor_positions = motor_positions
props.tilt_pivot_xz = (0.1, 0.0)

dyn = TiltrotorDynamics(mass=mass, rotor=rotor, mass_props=props, aero=aero)

# Transition envelope for envelope protection
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

transition_observer = TransitionStateObserver(time_constant_s=0.25)
envelope_protection = EnvelopeProtection(
    transition_envelope,
    max_alpha_rad=np.deg2rad(15.0),
    max_pitch_rad=np.deg2rad(60.0),
    max_body_rate_rad_s=np.deg2rad(180.0),
    speed_margin_mps=0.5,
)

controller = ControlLawArchitecture(
    mass=mass,
    k_thrust=k_thrust,
    max_omega=max_omega,
    motor_positions=motor_positions,
    tilt_pivot_xz=(0.1, 0.0),
    vtail_model=vtail,
    aero_model=aero,
    altitude_gains={"kp": alt_kp, "ki": alt_ki, "kd": alt_kd},
    speed_gains={"kp": spd_kp, "ki": spd_ki, "kd": spd_kd},
    attitude_gains={"kp": att_kp, "ki": att_ki, "kd": att_kd},
    rate_gains={"kp": rate_kp, "ki": rate_ki, "kd": rate_kd},
    hover_rate_scale=1.0,
    cruise_rate_scale=1.0,
    transition_observer=transition_observer,
    envelope_protection=envelope_protection,
)

# Initial state
state = AircraftState()
state.pos = np.array([0.0, 0.0, -100.0])
state.vel = np.array([12.0, 0.0, 0.0])
state.att = np.array([0.0, 0.0, 0.0])

# Logging
log = {
    "t": [], "alt": [], "pitch": [], "tilt": [], "speed": [],
    "u_body": [], "v_body": [], "w_body": [], "V_T": [], "alpha": [],
    "desired_pitch": [], "pitch_cmd": []
}

print("=" * 70)
print("TILTROTOR TRANSITION ANALYSIS (Direct PNG Generation)")
print("=" * 70)
print("\nSimulation config:")
print(f"  Aircraft mass: {mass} kg")
print(f"  Wing area: {wing_area} m²")
print(f"  Airfoil: NACA 6412 (m=0.06, p=0.4, t=0.12)")
print(f"  Duration: {duration} s")
print(f"  Control: ControlLawArchitecture with envelope protection")
print("\nRunning simulation...\n")

print("Time   Alt    Speed  Tilt   Pitch  Alpha  Status")
print("-" * 60)

max_alpha = -180.0
min_alpha = 180.0
target_alpha_min, target_alpha_max = 5.0, 12.0

# Main simulation loop
for i in range(int(duration / dt)):
    t = i * dt
    
    # Smooth reference trajectory
    if t < 2.0:
        ref_alt, ref_speed, ref_tilt = 100.0, 12.0, 0.0
    elif t < 10.0:
        progress = (t - 2.0) / 8.0
        ref_alt = 100.0
        ref_speed = 12.0 + progress * 4.0
        ref_tilt = np.deg2rad(progress * 90.0)
    else:
        ref_alt, ref_speed, ref_tilt = 100.0, 16.0, np.pi / 2.0
    
    reference = {
        "altitude": ref_alt,
        "speed": ref_speed,
        "tilt": ref_tilt,
        "phase": "TRANSITION",
    }
    
    # Controller step (includes envelope protection)
    cmd = controller.compute(state, reference, dt, np.zeros(3))
    
    # DEBUG: Check tilt command
    if int(t) % 2 == 0 and abs(t - int(t)) < dt * 2:
        tilt_cmd_deg = np.rad2deg(cmd["tilt"])
        #print(f"DEBUG t={t}: ref_tilt={np.rad2deg(ref_tilt):.1f}°, cmd_tilt={tilt_cmd_deg:.1f}°, state_tilt={np.rad2deg(state.tilt):.1f}°")
    
    # Tilt rate limit
    max_tilt_rate = np.deg2rad(10.0)
    actual_tilt = state.tilt + np.clip(
        cmd["tilt"] - state.tilt,
        -max_tilt_rate * dt,
        max_tilt_rate * dt,
    )
    
    # Dynamics step
    state = dyn.step(
        state,
        np.zeros(3),  # wind
        cmd["omega_fl"],
        cmd["omega_fr"],
        cmd["omega_rl"],
        cmd["omega_rr"],
        actual_tilt,
        cmd["delta_left"],
        cmd["delta_right"],
        dt,
    )
    
    # Calculate angle of attack
    vel_air_earth = state.vel - np.zeros(3)
    R_e2b = rotation_earth_to_body(state.att[0], state.att[1], state.att[2])
    vel_body = R_e2b @ vel_air_earth
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
    log["speed"].append(np.linalg.norm(state.vel[:2]))
    log["u_body"].append(vel_body[0])
    log["v_body"].append(vel_body[1])
    log["w_body"].append(vel_body[2])
    log["V_T"].append(V_T)
    log["alpha"].append(alpha_deg)
    log["desired_pitch"].append(np.rad2deg(cmd["desired_pitch"]))
    log["pitch_cmd"].append(np.rad2deg(cmd.get("pitch_cmd", 0.0)))
    
    # Print status every 2 seconds
    if int(t) % 2 == 0 and abs(t - int(t)) < dt * 2:
        status = "OK" if target_alpha_min <= alpha_deg <= target_alpha_max else "OUT"
        print(f"{t:5.1f}  {-state.pos[2]:5.0f}  {np.linalg.norm(state.vel[:2]):5.1f}  "
              f"{np.rad2deg(state.tilt):5.0f}  {np.rad2deg(state.att[1]):6.1f}  "
              f"{alpha_deg:5.1f}  {status}")

# Convert to numpy
for key in log:
    log[key] = np.array(log[key])

print("-" * 60)
print(f"\nSimulation Results:")
print(f"  Max AoA: {max_alpha:6.2f}° {'[OK]' if max_alpha <= target_alpha_max else '[EXCEED]'}")
print(f"  Min AoA: {min_alpha:6.2f}° {'[OK]' if min_alpha >= target_alpha_min else '[LOW]'}")
print(f"  Target:  {target_alpha_min}° - {target_alpha_max}°")

success = target_alpha_min <= min_alpha and max_alpha <= target_alpha_max
print(f"  Result:  {'[PASS]' if success else '[FAIL]'}")

# ============================================================
# PLOTS
# ============================================================

# Figure 1: Angle of Attack (PRIMARY)
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(log["t"], log["alpha"], 'b-', linewidth=3, label="Angle of Attack (measured)")
ax.axhline(15, color='r', linestyle='--', linewidth=2, alpha=0.7, label="Stall angle (15°)")
ax.axhline(-15, color='r', linestyle='--', linewidth=2, alpha=0.7)
ax.axhspan(target_alpha_min, target_alpha_max, color='green', alpha=0.2, label=f"Target ({target_alpha_min}°-{target_alpha_max}°)")
ax.axhspan(-target_alpha_max, -target_alpha_min, color='green', alpha=0.2)
ax.fill_between(log["t"], -25, 25, where=(np.abs(log["alpha"]) <= 15), 
                alpha=0.1, color='blue', label="Safe zone (|α| ≤ 15°)")
ax.set_xlabel("Mission time (s)", fontsize=13, fontweight='bold')
ax.set_ylabel("Angle of Attack α (deg)", fontsize=13, fontweight='bold')
ax.set_title("Angle of Attack During Tiltrotor Transition (with Envelope Protection)", 
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best')
ax.set_ylim(-20, 20)
plt.tight_layout()
plt.savefig(output_dir / "01_AoA.png", dpi=150, bbox_inches='tight')
print("\nSaved: 01_AoA.png")
plt.close()

# Figure 2: Pitch and Tilt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(log["t"], log["pitch"], 'b-', linewidth=2.5, label="Pitch attitude")
ax1.plot(log["t"], log["desired_pitch"], 'g--', linewidth=2, alpha=0.7, label="Desired pitch")
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.set_ylabel("Pitch (deg)", fontsize=12, fontweight='bold')
ax1.set_title("Pitch Attitude (Envelope Protected)", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2.plot(log["t"], log["tilt"], 'g-', linewidth=2.5, label="Tilt angle")
ax2.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Tilt (deg)", fontsize=12, fontweight='bold')
ax2.set_title("Nacelle Tilt Angle", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / "02_pitch_tilt.png", dpi=150, bbox_inches='tight')
print("Saved: 02_pitch_tilt.png")
plt.close()

# Figure 3: Altitude and Speed
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(log["t"], log["alt"], 'purple', linewidth=2.5, label="Altitude")
ax1.set_ylabel("Altitude (m)", fontsize=12, fontweight='bold')
ax1.set_title("Altitude Profile", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2.plot(log["t"], log["speed"], 'orange', linewidth=2.5, label="Airspeed")
ax2.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Airspeed (m/s)", fontsize=12, fontweight='bold')
ax2.set_title("Airspeed During Transition", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / "03_altitude_speed.png", dpi=150, bbox_inches='tight')
print("Saved: 03_altitude_speed.png")
plt.close()

# Figure 4: Body-Frame Velocities
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(log["t"], log["u_body"], 'r-', linewidth=2.5, label="u_body (forward)")
ax.plot(log["t"], log["w_body"], 'b-', linewidth=2.5, label="w_body (vertical)")
ax.plot(log["t"], log["V_T"], 'k--', linewidth=2.5, label="Total airspeed")
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel("Mission time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Velocity (m/s)", fontsize=12, fontweight='bold')
ax.set_title("Body-Frame Velocity Components", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "04_velocities.png", dpi=150, bbox_inches='tight')
print("Saved: 04_velocities.png")
plt.close()

# Figure 5: AoA vs Tilt (Phase Plot)
fig, ax = plt.subplots(figsize=(14, 8))
scatter = ax.scatter(log["tilt"], log["alpha"], c=log["t"], cmap='viridis', 
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.axhspan(target_alpha_min, target_alpha_max, color='green', alpha=0.15, label="Target zone")
ax.axhspan(-target_alpha_max, -target_alpha_min, color='green', alpha=0.15)
ax.axhline(15, color='r', linestyle='--', linewidth=2, alpha=0.5, label="Stall limit")
ax.axhline(-15, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel("Tilt Angle (deg)", fontsize=13, fontweight='bold')
ax.set_ylabel("Angle of Attack (deg)", fontsize=13, fontweight='bold')
ax.set_title("AoA vs Tilt - Envelope Protection Verification", fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax, label="Time (s)")
ax.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.savefig(output_dir / "05_aoa_vs_tilt.png", dpi=150, bbox_inches='tight')
print("Saved: 05_aoa_vs_tilt.png")
plt.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nAll plots saved to: {output_dir.absolute()}\n")
print("Files generated:")
print("  • 01_AoA.png - Angle of attack history with target envelope")
print("  • 02_pitch_tilt.png - Pitch and tilt angles")
print("  • 03_altitude_speed.png - Altitude and airspeed profiles")
print("  • 04_velocities.png - Body-frame velocity components")
print("  • 05_aoa_vs_tilt.png - Phase plot of AoA vs tilt angle")
print("\nNote: Results show NACA 6412 airfoil with envelope protection")
print("=" * 70)
