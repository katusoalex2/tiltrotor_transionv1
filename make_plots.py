#!/usr/bin/env python3
"""Extract and run dashboard simulation directly to generate PNG plots."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything the dashboard needs
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.aero.aerodynamics import AeroModel
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.control.architecture import ControlLawArchitecture
from src.control.mission import MissionManager, MissionProfile, Waypoint
from src.control.envelope_protection import EnvelopeProtection
from src.control.transition_observer import TransitionStateObserver
from src.dynamics.mass_properties import MassProperties
from src.dynamics.frames import rotation_earth_to_body
from src.dynamics.transition import transition_speed_envelope

cfg_path = "configs/blue_uam.yaml"
output_dir = Path("simulation_results")
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("TILTROTOR TRANSITION ANALYSIS - Direct PNG Generation")
print("=" * 70)

# Configuration matching dashboard defaults
mass = 5.0
mass_fuselage = 1.5
mass_wing = 1.0
mass_vtail = 0.3
mass_motor = 0.5

wing_area = 0.6
vtail_area = 0.15
vtail_arm = 0.3
prop_dia = 12.0
motor_positions = (
    (0.3, 0.4, 0.0, True),
    (0.3, -0.4, 0.0, True),
    (-0.3, 0.4, 0.0, False),
    (-0.3, -0.4, 0.0, False),
)
k_thrust = 1e-5
max_omega = 3200.0

alt_kp, alt_ki, alt_kd = 0.1, 0.005, 0.1
spd_kp, spd_ki, spd_kd = 0.05, 0.01, 0.05
att_kp, att_ki, att_kd = 1.5, 0.3, 0.5
rate_kp, rate_ki, rate_kd = 0.3, 0.1, 0.1

dt, duration = 0.01, 15.0
max_tilt_rate_deg_s = 10.0

# Initialize
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
props.motor_positions = list(motor_positions)
props.tilt_pivot_xz = (0.1, 0.0)

dyn = TiltrotorDynamics(mass=mass, rotor=rotor, mass_props=props, aero=aero)

# Transition envelope
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
    "t": [], "alt": [], "roll": [], "pitch": [], "yaw": [], "tilt": [], "speed": [],
    "u_body": [], "v_body": [], "w_body": [], "V_T": [], "alpha": [],
    "desired_pitch": []
}

print(f"\nConfiguration:")
print(f"  Mass: {mass} kg")
print(f"  Wing area: {wing_area} m²")
print(f"  Airfoil: NACA 6412 (m=0.06, p=0.4, t=0.12)")
print(f"  Duration: {duration} s")
print(f"  Control: ControlLawArchitecture with envelope protection\n")

max_alpha = -180.0
min_alpha = 180.0
target_min, target_max = 5.0, 12.0

print("Time   Alt    Speed  Tilt   Pitch  Alpha")
print("-" * 50)

# Main simulation loop
for i in range(int(duration / dt)):
    t = i * dt
    
    # Reference trajectory (simple ramp transition)
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
    }
    
    # Tilt rate limit
    tilt_error = ref_tilt - state.tilt
    tilt_step = np.deg2rad(max_tilt_rate_deg_s) * dt
    actual_tilt = state.tilt + np.clip(tilt_error, -tilt_step, tilt_step)
    
    # Controller step
    cmd = controller.compute(
        state,
        reference,
        dt,
        np.zeros(3),  # wind
        actuator_tilt_rad=actual_tilt,
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
    
    # Calculate AoA
    R_e2b = rotation_earth_to_body(state.att[0], state.att[1], state.att[2])
    vel_body = R_e2b @ state.vel
    V_T = np.linalg.norm(vel_body)
    alpha_rad = np.arctan2(vel_body[2], vel_body[0])
    alpha_deg = np.rad2deg(alpha_rad)
    
    max_alpha = max(max_alpha, alpha_deg)
    min_alpha = min(min_alpha, alpha_deg)
    
    # Log
    log["t"].append(t)
    log["alt"].append(-state.pos[2])
    log["roll"].append(np.rad2deg(state.att[0]))
    log["pitch"].append(np.rad2deg(state.att[1]))
    log["yaw"].append(np.rad2deg(state.att[2]))
    log["tilt"].append(np.rad2deg(state.tilt))
    log["speed"].append(np.linalg.norm(state.vel[:2]))
    log["u_body"].append(vel_body[0])
    log["v_body"].append(vel_body[1])
    log["w_body"].append(vel_body[2])
    log["V_T"].append(V_T)
    log["alpha"].append(alpha_deg)
    log["desired_pitch"].append(np.rad2deg(cmd["desired_pitch"]))
    
    # Print sample
    if int(t) % 2 == 0 and abs(t - int(t)) < dt * 2:
        print(f"{t:5.1f}  {-state.pos[2]:5.0f}  {np.linalg.norm(state.vel[:2]):5.1f}  "
              f"{np.rad2deg(state.tilt):5.0f}  {np.rad2deg(state.att[1]):6.1f}  {alpha_deg:5.1f}")

# Convert to numpy
for k in log:
    log[k] = np.array(log[k])

print("-" * 50)
print(f"\nResults:")
print(f"  Max AoA: {max_alpha:.2f} deg")
print(f"  Min AoA: {min_alpha:.2f} deg")
print(f"  Target:  {target_min:.1f} - {target_max:.1f} deg")
success = target_min <= min_alpha and max_alpha <= target_max
print(f"  Result:  [{'PASS' if success else 'FAIL'}]")

# ============= PLOTS =============

# Plot 1: AoA
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(log["t"], log["alpha"], 'b-', linewidth=3, label="Angle of Attack")
ax.axhline(15, color='r', linestyle='--', linewidth=2, alpha=0.7, label="Stall (15 deg)")
ax.axhline(-15, color='r', linestyle='--', linewidth=2, alpha=0.7)
ax.axhspan(target_min, target_max, color='green', alpha=0.2, label=f"Target ({target_min}-{target_max} deg)")
ax.axhspan(-target_max, -target_min, color='green', alpha=0.2)
ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("AoA (deg)", fontsize=12, fontweight='bold')
ax.set_title("Angle of Attack During Transition (NACA 6412 with Envelope Protection)", 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim(-20, 20)
plt.tight_layout()
plt.savefig(output_dir / "01_AoA.png", dpi=150, bbox_inches='tight')
print("\nSaved: 01_AoA.png")
plt.close()

# Plot 2: Pitch and Tilt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(log["t"], log["pitch"], 'b-', linewidth=2.5, label="Pitch")
ax1.plot(log["t"], log["desired_pitch"], 'g--', linewidth=2, alpha=0.7, label="Desired")
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.set_ylabel("Pitch (deg)", fontsize=12, fontweight='bold')
ax1.set_title("Pitch Attitude", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2.plot(log["t"], log["tilt"], 'g-', linewidth=2.5)
ax2.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Tilt (deg)", fontsize=12, fontweight='bold')
ax2.set_title("Nacelle Tilt", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(output_dir / "02_pitch_tilt.png", dpi=150, bbox_inches='tight')
print("Saved: 02_pitch_tilt.png")
plt.close()

# Plot 3: Altitude and Speed
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(log["t"], log["alt"], 'purple', linewidth=2.5)
ax1.set_ylabel("Altitude (m)", fontsize=12, fontweight='bold')
ax1.set_title("Altitude Profile", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(log["t"], log["speed"], 'orange', linewidth=2.5)
ax2.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Speed (m/s)", fontsize=12, fontweight='bold')
ax2.set_title("Airspeed", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "03_altitude_speed.png", dpi=150, bbox_inches='tight')
print("Saved: 03_altitude_speed.png")
plt.close()

# Plot 4: Body velocities
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(log["t"], log["u_body"], 'r-', linewidth=2.5, label="u (forward)")
ax.plot(log["t"], log["w_body"], 'b-', linewidth=2.5, label="w (vertical)")
ax.plot(log["t"], log["V_T"], 'k--', linewidth=2.5, label="Total speed")
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("Velocity (m/s)", fontsize=12, fontweight='bold')
ax.set_title("Body Velocity Components", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "04_velocities.png", dpi=150, bbox_inches='tight')
print("Saved: 04_velocities.png")
plt.close()

# Plot 5: AoA vs Tilt
fig, ax = plt.subplots(figsize=(14, 8))
scatter = ax.scatter(log["tilt"], log["alpha"], c=log["t"], cmap='viridis', s=50, alpha=0.7)
ax.axhspan(target_min, target_max, color='green', alpha=0.15, label="Target zone")
ax.axhspan(-target_max, -target_min, color='green', alpha=0.15)
ax.axhline(15, color='r', linestyle='--', linewidth=2, alpha=0.5, label="Stall")
ax.axhline(-15, color='r', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel("Tilt (deg)", fontsize=13, fontweight='bold')
ax.set_ylabel("AoA (deg)", fontsize=13, fontweight='bold')
ax.set_title("AoA vs Tilt - Envelope Protection Verification", fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax, label="Time (s)")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "05_aoa_vs_tilt.png", dpi=150, bbox_inches='tight')
print("Saved: 05_aoa_vs_tilt.png")
plt.close()

# Plot 6: Roll and yaw Euler angles
state_time = log["t"] + dt
near_singularity = np.flatnonzero(np.abs(np.cos(np.deg2rad(log["pitch"]))) < np.sin(np.deg2rad(5.0)))
below_ground = np.flatnonzero(log["alt"] <= 0.0)
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for ax, name, color in zip(axes, ("roll", "yaw"), ("tab:blue", "tab:purple")):
    ax.plot(state_time, log[name], color=color, linewidth=2, label=f"{name.capitalize()} angle")
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    if near_singularity.size:
        ax.axvline(state_time[near_singularity[0]], color="tab:orange", linestyle="--",
                   label="First pitch within 5 deg of Euler singularity")
    if below_ground.size:
        ax.axvline(state_time[below_ground[0]], color="tab:red", linestyle=":",
                   label="First simulated altitude at/below ground")
    if np.max(np.abs(log[name])) < 1.0:
        ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel(f"{name.capitalize()} (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
axes[-1].set_xlabel("Time (s)")
fig.suptitle("6-DOF Simulation: Roll and Yaw vs Time (zero wind)")
fig.text(0.5, 0.01, "Zero wind and symmetric conditions. Diagnostic only: this run diverges; "
         "Euler angles become ill-conditioned near +/-90 deg pitch.",
         ha="center", color="darkred")
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(output_dir / "06_roll_yaw_vs_time.png", dpi=150, bbox_inches="tight")
print("Saved: 06_roll_yaw_vs_time.png")
plt.close(fig)

print("\n" + "=" * 70)
print("COMPLETE - All plots saved to simulation_results/")
print("=" * 70)
