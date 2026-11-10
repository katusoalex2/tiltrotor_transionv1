"""Verify that corridor-reference tracking is better than constant-rate tilt."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.cascade import CascadeController
from src.control.trajectory import ReferenceTrajectory, LinearReference


def run_sim(reference, duration, dt=0.01):
    """Run a transition and return the full log."""
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)

    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)
    controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    log = {
        "t": [], "alt": [], "ref_alt": [], "speed": [], "ref_speed": [],
        "tilt": [], "ref_tilt": [], "pitch": [], "desired_pitch": [],
        "alt_err": [], "speed_err": [], "tilt_err": [],
    }

    for i in range(int(duration / dt)):
        t = i * dt
        r = reference.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)

        actual_alt = -state.pos[2]
        actual_speed = state.vel[0]

        log["t"].append(t)
        log["alt"].append(actual_alt)
        log["ref_alt"].append(r["altitude"])
        log["speed"].append(actual_speed)
        log["ref_speed"].append(r["speed"])
        log["tilt"].append(np.rad2deg(r["tilt"]))
        log["ref_tilt"].append(np.rad2deg(r["tilt"]))
        log["pitch"].append(np.rad2deg(state.att[1]))
        log["desired_pitch"].append(np.rad2deg(cmd["desired_pitch"]))
        log["alt_err"].append(actual_alt - r["altitude"])
        log["speed_err"].append(actual_speed - r["speed"])
        log["tilt_err"].append(0.0)  # tilt is exactly tracked

    return {k: np.array(v) for k, v in log.items()}


def compute_metrics(log):
    """Compute tracking performance metrics."""
    return {
        "max_alt_err": float(np.max(np.abs(log["alt_err"]))),
        "rms_alt_err": float(np.sqrt(np.mean(log["alt_err"] ** 2))),
        "max_speed_err": float(np.max(np.abs(log["speed_err"]))),
        "rms_speed_err": float(np.sqrt(np.mean(log["speed_err"] ** 2))),
        "max_alt_loss": float(100 - log["alt"].min()),
        "max_pitch": float(np.max(np.abs(log["pitch"]))),
    }


def main():
    # Load both references
    corridor_path = "configs/robust_corridor.csv"
    if not os.path.exists(corridor_path):
        print(f"ERROR: {corridor_path} not found. Run scripts/robust_corridor.py first.")
        return

    ref_opt = ReferenceTrajectory(corridor_path, speed_max=12.0)
    duration = ref_opt.t_end + 5.0

    # Constant-rate baseline (same total transition time, linear tilt)
    ref_lin = LinearReference(start_altitude=100.0, target_speed=12.0, transition_time=ref_opt.t_end)

    print("Running OPTIMIZED corridor tracking...")
    log_opt = run_sim(ref_opt, duration)
    metrics_opt = compute_metrics(log_opt)

    print("Running CONSTANT-RATE tilt tracking...")
    log_lin = run_sim(ref_lin, duration)
    metrics_lin = compute_metrics(log_lin)

    # --- Print comparison ---
    print("\n=== Tracking Performance Comparison ===")
    print(f"{'Metric':<28} {'Constant-Rate':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-" * 76)

    for key in ["max_alt_err", "rms_alt_err", "max_speed_err", "rms_speed_err", "max_alt_loss", "max_pitch"]:
        lin = metrics_lin[key]
        opt = metrics_opt[key]
        improvement = ((lin - opt) / lin * 100) if lin > 1e-6 else 0.0
        print(f"{key:<28} {lin:>15.3f} {opt:>15.3f} {improvement:>14.1f}%")

    # Verdict
    print("\n=== Verdict ===")
    win_metrics = sum([
        metrics_opt["max_alt_err"] < metrics_lin["max_alt_err"],
        metrics_opt["rms_alt_err"] < metrics_lin["rms_alt_err"],
        metrics_opt["max_alt_loss"] < metrics_lin["max_alt_loss"],
        metrics_opt["max_pitch"] < metrics_lin["max_pitch"],
    ])
    if win_metrics >= 3:
        print(f"✓ OPTIMIZED corridor wins on {win_metrics}/4 metrics. Corridor reference is verified.")
    elif win_metrics >= 2:
        print(f"~ MIXED: Optimized wins {win_metrics}/4. Partial improvement.")
    else:
        print(f"✗ Constant-rate wins on most metrics. Need to investigate.")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Altitude tracking
    axes[0, 0].plot(log_opt["t"], log_opt["alt"], 'b-', label="Optimized", linewidth=2)
    axes[0, 0].plot(log_opt["t"], log_opt["ref_alt"], 'b:', label="Optimized ref", linewidth=1)
    axes[0, 0].plot(log_lin["t"], log_lin["alt"], 'r--', label="Constant-rate", linewidth=2)
    axes[0, 0].set_xlabel("Time (s)"); axes[0, 0].set_ylabel("Altitude (m)")
    axes[0, 0].set_title("Altitude Tracking")
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    # Altitude error
    axes[0, 1].plot(log_opt["t"], log_opt["alt_err"], 'b-', linewidth=2, label="Optimized")
    axes[0, 1].plot(log_lin["t"], log_lin["alt_err"], 'r--', linewidth=2, label="Constant-rate")
    axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel("Time (s)"); axes[0, 1].set_ylabel("Altitude error (m)")
    axes[0, 1].set_title("Altitude Tracking Error")
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    # Speed tracking
    axes[1, 0].plot(log_opt["t"], log_opt["speed"], 'b-', label="Optimized", linewidth=2)
    axes[1, 0].plot(log_opt["t"], log_opt["ref_speed"], 'b:', label="Optimized ref", linewidth=1)
    axes[1, 0].plot(log_lin["t"], log_lin["speed"], 'r--', label="Constant-rate", linewidth=2)
    axes[1, 0].set_xlabel("Time (s)"); axes[1, 0].set_ylabel("Speed (m/s)")
    axes[1, 0].set_title("Forward Speed Tracking")
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    # Pitch excursion
    axes[1, 1].plot(log_opt["t"], log_opt["pitch"], 'b-', linewidth=2, label="Optimized")
    axes[1, 1].plot(log_lin["t"], log_lin["pitch"], 'r--', linewidth=2, label="Constant-rate")
    axes[1, 1].axhline(0, color='k', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel("Time (s)"); axes[1, 1].set_ylabel("Pitch (deg)")
    axes[1, 1].set_title("Pitch Excursion")
    axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("tracking_verification.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()