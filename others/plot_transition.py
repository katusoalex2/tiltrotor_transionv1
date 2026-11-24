#!/usr/bin/env python3
"""Generate transition plots directly - simple and reliable."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.cascade import CascadeController
from src.dynamics.frames import rotation_earth_to_body

# Output
output_dir = Path("simulation_results")
output_dir.mkdir(exist_ok=True)

# Configuration
mass = 5.0
wing_area = 0.6
vtail_area = 0.15
vtail_arm = 0.3
k_thrust = 1e-5
max_omega = 3200.0
dt, duration = 0.01, 15.0

# Initialize
rotor = RotorModel(k_thrust=k_thrust, max_omega=max_omega)
vtail = VtailModel(area=vtail_area, arm_length=vtail_arm)
aero = AeroModel(wing_area=wing_area, vtail=vtail, prop_diameter_m=12.0 * 0.0254)
dyn = TiltrotorDynamics(mass=mass, rotor=rotor, inertia=np.array([0.08, 0.12, 0.14]), aero=aero)

omega_hover = np.sqrt(mass * 9.81 / k_thrust)
controller = CascadeController(mass=mass, k_thrust=k_thrust, hover_omega=omega_hover)

# Initial state
state = AircraftState()
state.pos = np.array([0.0, 0.0, -100.0])
state.vel = np.array([12.0, 0.0, 0.0])
state.att = np.array([0.0, 0.0, 0.0])

# Logs
log = {
    "t": [], "alt": [], "pitch": [], "tilt": [], "speed": [],
    "u_body": [], "v_body": [], "w_body": [], "V_T": [], "alpha": []
}

print("=" * 70)
print("TILTROTOR TRANSITION ANALYSIS")
print("=" * 70)
print(f"Configuration:")
print(f"  Mass: {mass} kg")
print(f"  Wing area: {wing_area} m²")
print(f"  Airfoil: NACA 6412 (m=0.06, p=0.4, t=0.12)")
print(f"  Duration: {duration} s")
print(f"  Control: CascadeController with AoA limiter")
print()

max_alpha = -180.0
min_alpha = 180.0
target_min, target_max = 5.0, 12.0

# Simulation
for i in range(int(duration / dt)):
    t = i * dt
    
    # Reference trajectory
    if t < 2.0:
        ref_alt, ref_speed, ref_tilt = 100.0, 12.0, 0.0
    elif t < 10.0:
        progress = (t - 2.0) / 8.0
        ref_alt = 100.0
        ref_speed = 12.0 + progress * 4.0
        ref_tilt = np.deg2rad(progress * 90.0)
    else:
        ref_alt, ref_speed, ref_tilt = 100.0, 16.0, np.pi / 2.0
    
    # Control (CascadeController already has AoA limiter)
    cmd = controller.compute(t, state, ref_alt, ref_speed, ref_tilt)
    
    # Tilt rate limit
    max_tilt_rate = np.deg2rad(10.0)
    actual_tilt = state.tilt + np.clip(
        cmd["tilt"] - state.tilt,
        -max_tilt_rate * dt,
        max_tilt_rate * dt
    )
    state.tilt = actual_tilt
    
    # Dynamics
    state = dyn.step(state, cmd["omega"], cmd["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
    
    # AoA calculation
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
    log["pitch"].append(np.rad2deg(state.att[1]))
    log["tilt"].append(np.rad2deg(state.tilt))
    log["speed"].append(np.linalg.norm(state.vel[:2]))
    log["u_body"].append(vel_body[0])
    log["v_body"].append(vel_body[1])
    log["w_body"].append(vel_body[2])
    log["V_T"].append(V_T)
    log["alpha"].append(alpha_deg)

# Convert
for k in log:
    log[k] = np.array(log[k])

print("Time   Alt    Speed  Tilt   Pitch  Alpha")
print("-" * 50)
for i in [0, 100, 200, 400, 600, 800, 1000, 1200, 1400]:
    if i < len(log["t"]):
        t = log["t"][i]
        print(f"{t:5.1f}  {log['alt'][i]:5.0f}  {log['speed'][i]:5.1f}  "
              f"{log['tilt'][i]:5.0f}  {log['pitch'][i]:6.1f}  {log['alpha'][i]:5.1f}")

print("-" * 50)
print(f"\nResults:")
print(f"  Max AoA: {max_alpha:.2f} deg")
print(f"  Min AoA: {min_alpha:.2f} deg")
print(f"  Target:  {target_min:.1f} - {target_max:.1f} deg")
success = target_min <= min_alpha and max_alpha <= target_max
print(f"  Result:  [{'PASS' if success else 'FAIL'}]")

# ============== PLOTS ==============

# Plot 1: AoA
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(log["t"], log["alpha"], 'b-', linewidth=3, label="Angle of Attack")
ax.axhline(15, color='r', linestyle='--', linewidth=2, alpha=0.7, label="Stall (15 deg)")
ax.axhline(-15, color='r', linestyle='--', linewidth=2, alpha=0.7)
ax.axhspan(target_min, target_max, color='green', alpha=0.2, label=f"Target ({target_min}-{target_max} deg)")
ax.axhspan(-target_max, -target_min, color='green', alpha=0.2)
ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
ax.set_ylabel("AoA (deg)", fontsize=12, fontweight='bold')
ax.set_title("Angle of Attack During Transition (NACA 6412 with AoA Limiter)", fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim(-20, 20)
plt.tight_layout()
plt.savefig(output_dir / "01_AoA.png", dpi=150, bbox_inches='tight')
print("\nSaved: 01_AoA.png")
plt.close()

# Plot 2: Pitch and Tilt
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
ax1.plot(log["t"], log["pitch"], 'b-', linewidth=2.5)
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.set_ylabel("Pitch (deg)", fontsize=12, fontweight='bold')
ax1.set_title("Pitch Attitude", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(log["t"], log["tilt"], 'g-', linewidth=2.5)
ax2.set_xlabel("Time (s)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Tilt (deg)", fontsize=12, fontweight='bold')
ax2.set_title("Nacelle Tilt Angle", fontsize=13, fontweight='bold')
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
ax.set_title("AoA vs Tilt - Limiter Verification", fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax, label="Time (s)")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(output_dir / "05_aoa_vs_tilt.png", dpi=150, bbox_inches='tight')
print("Saved: 05_aoa_vs_tilt.png")
plt.close()

print("\n" + "=" * 70)
print("COMPLETE - All plots saved to simulation_results/")
print("=" * 70)
