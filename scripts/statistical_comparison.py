"""Deep statistical comparison of PID vs INDI under wind gusts.

Reruns the Monte Carlo (or loads existing data if cached) and produces:
    1. Per-gust-intensity binned comparison
    2. Tail analysis (worst 10% of runs)
    3. Pareto frontier (altitude loss vs saturation)
    4. Best/worst case rankings
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.cascade import CascadeController
from src.control.trajectory import ReferenceTrajectory
from src.control.pid import PIDController
from src.control.indi import INDIController
from src.wind.gust import DrydenGust


class PIDCascade:
    """Cascade controller using PID for inner pitch loop."""
    def __init__(self, mass, k_thrust):
        self.mass = mass
        self.k_thrust = k_thrust
        self.alt_pid = PIDController(kp=0.1, ki=0.005, kd=0.1, limit=0.25)
        self.speed_pid = PIDController(kp=0.03, ki=0.002, kd=0.01, limit=0.20)
        self.pitch_pid = PIDController(kp=0.6, ki=0.05, kd=0.4, limit=0.2)

    def compute(self, t, state, ref_alt, ref_speed, ref_tilt):
        current_alt = -state.pos[2]
        current_speed = state.vel[0]
        current_pitch = state.att[1]

        pitch_from_alt = self.alt_pid.compute(ref_alt, current_alt, 0.02)
        pitch_from_speed = -self.speed_pid.compute(ref_speed, current_speed, 0.02)
        desired_pitch = np.clip(pitch_from_alt + pitch_from_speed, -0.35, 0.35)

        vtail_cmd = -self.pitch_pid.compute(desired_pitch, current_pitch, 0.01)

        altitude_error = ref_alt - current_alt
        total_thrust = (self.mass * 9.81) + (altitude_error * 0.5)
        omega_cmd = np.sqrt(max(total_thrust, 0.0) / self.k_thrust)
        omega_cmd = np.clip(omega_cmd, 0.0, 3000.0)

        return {"omega": omega_cmd, "tilt": ref_tilt, "delta_left": vtail_cmd,
                "delta_right": vtail_cmd, "desired_pitch": desired_pitch}


def run_one(gust_sigma, seed, controller_type, dt=0.01):
    """Single Monte Carlo run."""
    duration = 20.0

    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)
    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)

    if controller_type == "pid":
        controller = PIDCascade(mass=5.0, k_thrust=1e-5)
    else:
        controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)

    ref = ReferenceTrajectory("configs/robust_corridor.csv", speed_max=12.0)
    wind = DrydenGust(sigma=(gust_sigma, gust_sigma, gust_sigma * 0.6),
                       airspeed=12.0, seed=seed)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    alt_history, pitch_history, vtail_history = [], [], []
    saturated_count = 0
    crashed = False

    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])

        wind_vec = wind.step(dt)
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd["omega"], r["tilt"],
                         cmd["delta_left"], cmd["delta_right"], dt)
        state.vel = state.vel - wind_vec

        alt_history.append(-state.pos[2])
        pitch_history.append(np.rad2deg(state.att[1]))
        vtail_history.append(cmd["delta_left"])

        if abs(cmd["delta_left"]) >= 0.199:
            saturated_count += 1
        if -state.pos[2] < 0:
            crashed = True
            break

    alt_history = np.array(alt_history) if alt_history else np.array([100.0])
    pitch_history = np.array(pitch_history) if pitch_history else np.array([0.0])

    return {
        "sigma": gust_sigma,
        "max_alt_loss": float(100.0 - alt_history.min()) if not crashed else 999.0,
        "pitch_rms": float(np.sqrt(np.mean(pitch_history**2))) if not crashed else 999.0,
        "saturation_count": int(saturated_count),
        "crashed": crashed,
        "alt_history": alt_history,
    }


def main():
    n_runs = 200
    print(f"Running statistical comparison ({n_runs} runs)...")
    print("This will take 2-4 minutes. Please wait...\n")

    rng = np.random.default_rng(2024)
    pid_results, indi_results = [], []

    for i in range(n_runs):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{n_runs}")

        seed = int(rng.integers(0, 100000))
        sigma = float(rng.uniform(0.5, 5.0))

        pid_results.append(run_one(sigma, seed, "pid"))
        indi_results.append(run_one(sigma, seed, "indi"))

    # ============================================================
    # ANALYSIS 1: Per-gust-intensity comparison
    # ============================================================
    print("\n" + "=" * 75)
    print("ANALYSIS 1: PER-GUST-INTENSITY COMPARISON")
    print("=" * 75)
    print(f"\n{'Sigma range':>15} {'Count':>8} {'PID mean':>12} {'INDI mean':>12} {'Winner':>10}")
    print("-" * 60)

    sigma_bins = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5)]
    for lo, hi in sigma_bins:
        pid_vals = [r["max_alt_loss"] for r in pid_results
                    if lo <= r["sigma"] < hi and not r["crashed"]]
        indi_vals = [r["max_alt_loss"] for r in indi_results
                     if lo <= r["sigma"] < hi and not r["crashed"]]
        if not pid_vals or not indi_vals:
            continue
        pid_mean = np.mean(pid_vals)
        indi_mean = np.mean(indi_vals)
        winner = "INDI" if indi_mean < pid_mean else "PID"
        print(f"  {lo:.1f}–{hi:.1f} m/s  {len(pid_vals):>8}  {pid_mean:>11.2f}m  {indi_mean:>11.2f}m  {winner:>10}")

    # ============================================================
    # ANALYSIS 2: Tail analysis (worst 10% of runs)
    # ============================================================
    print("\n" + "=" * 75)
    print("ANALYSIS 2: TAIL ANALYSIS (worst 10% of runs)")
    print("=" * 75)

    pid_losses = np.array([r["max_alt_loss"] for r in pid_results if not r["crashed"]])
    indi_losses = np.array([r["max_alt_loss"] for r in indi_results if not r["crashed"]])

    pid_95 = np.percentile(pid_losses, 95)
    indi_95 = np.percentile(indi_losses, 95)
    pid_99 = np.percentile(pid_losses, 99) if len(pid_losses) > 100 else pid_losses.max()
    indi_99 = np.percentile(indi_losses, 99) if len(indi_losses) > 100 else indi_losses.max()

    print(f"\n{'Metric':<30} {'PID':>12} {'INDI':>12}")
    print("-" * 55)
    print(f"{'Median altitude loss':<30} {np.median(pid_losses):>11.2f}m {np.median(indi_losses):>11.2f}m")
    print(f"{'95th percentile':<30} {pid_95:>11.2f}m {indi_95:>11.2f}m")
    print(f"{'99th percentile':<30} {pid_99:>11.2f}m {indi_99:>11.2f}m")
    print(f"{'Max (worst case)':<30} {pid_losses.max():>11.2f}m {indi_losses.max():>11.2f}m")

    ratio_95 = pid_95 / indi_95 if indi_95 > 0 else float('inf')
    print(f"\n95th percentile ratio (PID / INDI): {ratio_95:.2f}×")
    if ratio_95 > 1.1:
        print(f"  → INDI tail is {ratio_95:.2f}× tighter than PID's worst 5%")
    else:
        print(f"  → Comparable tail behavior")

    # ============================================================
    # ANALYSIS 3: Crash rate by gust intensity
    # ============================================================
    print("\n" + "=" * 75)
    print("ANALYSIS 3: CRASH RATE BY GUST INTENSITY")
    print("=" * 75)
    print(f"\n{'Sigma range':>15} {'PID crash%':>12} {'INDI crash%':>12} {'Winner':>10}")
    print("-" * 55)

    for lo, hi in sigma_bins:
        pid_bin = [r for r in pid_results if lo <= r["sigma"] < hi]
        indi_bin = [r for r in indi_results if lo <= r["sigma"] < hi]
        if not pid_bin:
            continue
        pid_crash = sum(r["crashed"] for r in pid_bin) / len(pid_bin) * 100
        indi_crash = sum(r["crashed"] for r in indi_bin) / len(indi_bin) * 100
        if abs(pid_crash - indi_crash) < 1.0:
            winner = "TIE"
        else:
            winner = "INDI" if indi_crash < pid_crash else "PID"
        print(f"  {lo:.1f}–{hi:.1f} m/s {pid_crash:>11.1f}% {indi_crash:>11.1f}% {winner:>10}")

    # ============================================================
    # PLOTS
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Binned altitude loss vs gust intensity
    bin_centers = [(lo + hi) / 2 for lo, hi in sigma_bins]
    pid_bin_means, indi_bin_means, pid_bin_std, indi_bin_std = [], [], [], []
    for lo, hi in sigma_bins:
        p = [r["max_alt_loss"] for r in pid_results if lo <= r["sigma"] < hi and not r["crashed"]]
        i = [r["max_alt_loss"] for r in indi_results if lo <= r["sigma"] < hi and not r["crashed"]]
        pid_bin_means.append(np.mean(p) if p else 0)
        indi_bin_means.append(np.mean(i) if i else 0)
        pid_bin_std.append(np.std(p) if p else 0)
        indi_bin_std.append(np.std(i) if i else 0)

    axes[0, 0].errorbar(bin_centers, pid_bin_means, yerr=pid_bin_std,
                         fmt='o-', color='tab:red', label="PID", capsize=4, linewidth=2)
    axes[0, 0].errorbar(bin_centers, indi_bin_means, yerr=indi_bin_std,
                         fmt='s-', color='tab:blue', label="INDI", capsize=4, linewidth=2)
    axes[0, 0].set_xlabel("Gust sigma (m/s)")
    axes[0, 0].set_ylabel("Mean altitude loss (m)")
    axes[0, 0].set_title("Altitude Loss by Gust Intensity")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Tail comparison (CDF)
    pid_sorted = np.sort(pid_losses)
    indi_sorted = np.sort(indi_losses)
    pid_cdf = np.arange(1, len(pid_sorted) + 1) / len(pid_sorted)
    indi_cdf = np.arange(1, len(indi_sorted) + 1) / len(indi_sorted)

    axes[0, 1].plot(pid_sorted, pid_cdf, 'r-', label="PID", linewidth=2)
    axes[0, 1].plot(indi_sorted, indi_cdf, 'b-', label="INDI", linewidth=2)
    axes[0, 1].axhline(0.95, color='k', linestyle=':', alpha=0.5, label="95th pct")
    axes[0, 1].set_xlabel("Max altitude loss (m)")
    axes[0, 1].set_ylabel("Cumulative probability")
    axes[0, 1].set_title("CDF: Altitude Loss (tail behavior)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Pareto (altitude loss vs saturation)
    axes[1, 0].scatter([r["saturation_count"] for r in pid_results if not r["crashed"]],
                        [r["max_alt_loss"] for r in pid_results if not r["crashed"]],
                        alpha=0.4, c='tab:red', label="PID", s=20)
    axes[1, 0].scatter([r["saturation_count"] for r in indi_results if not r["crashed"]],
                        [r["max_alt_loss"] for r in indi_results if not r["crashed"]],
                        alpha=0.4, c='tab:blue', label="INDI", s=20)
    axes[1, 0].set_xlabel("Saturation events")
    axes[1, 0].set_ylabel("Max altitude loss (m)")
    axes[1, 0].set_title("Pareto: Altitude Loss vs Actuator Effort")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Pitch RMS distribution
    pid_pitch = [r["pitch_rms"] for r in pid_results if not r["crashed"]]
    indi_pitch = [r["pitch_rms"] for r in indi_results if not r["crashed"]]
    axes[1, 1].boxplot([pid_pitch, indi_pitch], tick_labels=["PID", "INDI"], patch_artist=True,
                        boxprops=dict(facecolor='lightgray'))
    axes[1, 1].set_ylabel("Pitch RMS (deg)")
    axes[1, 1].set_title("Pitch Stability Comparison")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("statistical_comparison.png", dpi=100)
    plt.show()

    # ============================================================
    # Final verdict
    # ============================================================
    print("\n" + "=" * 75)
    print("VERDICT")
    print("=" * 75)
    
    pid_crash_total = sum(1 for r in pid_results if r["crashed"])
    indi_crash_total = sum(1 for r in indi_results if r["crashed"])
    
    print(f"\nCrashes: PID={pid_crash_total}/{n_runs}, INDI={indi_crash_total}/{n_runs}")
    if indi_crash_total < pid_crash_total:
        print(f"→ INDI is MORE ROBUST by {pid_crash_total - indi_crash_total} crashes")
    elif pid_crash_total < indi_crash_total:
        print(f"→ PID is MORE ROBUST by {indi_crash_total - pid_crash_total} crashes")
    else:
        print("→ Comparable robustness (same crash count)")


if __name__ == "__main__":
    main()