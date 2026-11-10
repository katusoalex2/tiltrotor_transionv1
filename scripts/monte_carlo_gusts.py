"""Monte Carlo simulation: 200 runs with random gusts, PID vs INDI comparison."""

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
    """Cascade controller using PID for the inner pitch loop."""
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

        return {"omega": omega_cmd, "tilt": ref_tilt, "delta_left": vtail_cmd, "delta_right": vtail_cmd,
                "desired_pitch": desired_pitch}


def run_one(gust_sigma: float, seed: int, controller_type: str, dt: float = 0.01):
    """Run a single transition with wind gusts."""
    duration = 20.0

    # --- Setup aircraft ---
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)
    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)

    # --- Setup controller ---
    if controller_type == "pid":
        controller = PIDCascade(mass=5.0, k_thrust=1e-5)
    else:
        controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)

    # --- Setup reference & wind ---
    ref = ReferenceTrajectory("configs/robust_corridor.csv", speed_max=12.0)
    wind = DrydenGust(sigma=(gust_sigma, gust_sigma, gust_sigma * 0.6), airspeed=12.0, seed=seed)

    # --- Initial state ---
    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    alt_history = []
    pitch_history = []
    vtail_history = []
    saturated_count = 0
    crashed = False

    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])

        # Inject gust into aircraft velocity for aerodynamics
        wind_vec = wind.step(dt)
        # Gust adds to airspeed (world frame). Simplification: add to state velocity.
        state.vel_effective = state.vel + wind_vec if not hasattr(state, "vel_effective") else state.vel + wind_vec

        # Step physics (note: our current dynamics doesn't accept wind directly,
        # so we inject via velocity before the step)
        # Quick hack: temporarily add wind to velocity for the aero computation
        original_vel = state.vel.copy()
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
        # Remove wind from state velocity (wind is external, doesn't affect inertial velocity)
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
        "max_alt_loss": float(100.0 - alt_history.min()) if not crashed else 999.0,
        "pitch_rms": float(np.sqrt(np.mean(pitch_history**2))) if not crashed else 999.0,
        "saturation_count": int(saturated_count),
        "crashed": crashed,
    }


def main():
    n_runs = 200
    print(f"Running Monte Carlo ({n_runs} runs, PID vs INDI)...")
    print("This will take 2-4 minutes. Please wait...\n")

    rng = np.random.default_rng(2024)

    pid_results = []
    indi_results = []

    for i in range(n_runs):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{n_runs}")

        seed = int(rng.integers(0, 100000))
        sigma = float(rng.uniform(0.5, 5.0))

        pid_res = run_one(gust_sigma=sigma, seed=seed, controller_type="pid")
        indi_res = run_one(gust_sigma=sigma, seed=seed, controller_type="indi")

        pid_results.append({"sigma": sigma, **pid_res})
        indi_results.append({"sigma": sigma, **indi_res})

    # --- Aggregate stats ---
    print("\n" + "=" * 70)
    print("MONTE CARLO RESULTS — PID vs INDI")
    print("=" * 70)

    def summarize(results, label):
        crash_rate = sum(r["crashed"] for r in results) / len(results) * 100
        valid = [r for r in results if not r["crashed"]]
        alt_losses = [r["max_alt_loss"] for r in valid] if valid else [999.0]
        pitch_rms = [r["pitch_rms"] for r in valid] if valid else [999.0]
        sat = [r["saturation_count"] for r in valid] if valid else [0]

        print(f"\n[{label}]")
        print(f"  Crashes:                {sum(r['crashed'] for r in results)} / {len(results)} ({crash_rate:.1f}%)")
        print(f"  Mean alt loss:          {np.mean(alt_losses):.2f} m")
        print(f"  Median alt loss:        {np.median(alt_losses):.2f} m")
        print(f"  95th percentile loss:   {np.percentile(alt_losses, 95):.2f} m")
        print(f"  Mean pitch RMS:         {np.mean(pitch_rms):.2f} deg")
        print(f"  95th pct pitch RMS:     {np.percentile(pitch_rms, 95):.2f} deg")
        print(f"  Mean saturation events: {np.mean(sat):.1f}")
        return {
            "crash_rate": crash_rate,
            "alt_losses": alt_losses,
            "pitch_rms": pitch_rms,
            "sat": sat,
        }

    pid_summary = summarize(pid_results, "PID")
    indi_summary = summarize(indi_results, "INDI")

    # --- Statistical tests ---
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    if len(pid_summary["alt_losses"]) > 10 and len(indi_summary["alt_losses"]) > 10:
        t_stat, p_val = stats.ttest_ind(pid_summary["alt_losses"], indi_summary["alt_losses"])
        print(f"\nAltitude loss t-test: t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            winner = "INDI" if np.mean(indi_summary["alt_losses"]) < np.mean(pid_summary["alt_losses"]) else "PID"
            print(f"  → {winner} is SIGNIFICANTLY better (p < 0.05)")
        else:
            print(f"  → No significant difference at p < 0.05")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Altitude loss distribution
    axes[0, 0].hist(pid_summary["alt_losses"], bins=30, alpha=0.5, label="PID", color="tab:red")
    axes[0, 0].hist(indi_summary["alt_losses"], bins=30, alpha=0.5, label="INDI", color="tab:blue")
    axes[0, 0].set_xlabel("Max altitude loss (m)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Altitude Loss Distribution")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Pitch RMS distribution
    axes[0, 1].hist(pid_summary["pitch_rms"], bins=30, alpha=0.5, label="PID", color="tab:red")
    axes[0, 1].hist(indi_summary["pitch_rms"], bins=30, alpha=0.5, label="INDI", color="tab:blue")
    axes[0, 1].set_xlabel("Pitch RMS (deg)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Pitch Excursion Distribution")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Altitude loss vs gust intensity (scatter)
    pid_sigmas = [r["sigma"] for r in pid_results if not r["crashed"]]
    pid_alt = [r["max_alt_loss"] for r in pid_results if not r["crashed"]]
    indi_sigmas = [r["sigma"] for r in indi_results if not r["crashed"]]
    indi_alt = [r["max_alt_loss"] for r in indi_results if not r["crashed"]]

    axes[1, 0].scatter(pid_sigmas, pid_alt, alpha=0.4, label="PID", color="tab:red", s=15)
    axes[1, 0].scatter(indi_sigmas, indi_alt, alpha=0.4, label="INDI", color="tab:blue", s=15)
    axes[1, 0].set_xlabel("Gust sigma (m/s)")
    axes[1, 0].set_ylabel("Max altitude loss (m)")
    axes[1, 0].set_title("Altitude Loss vs Gust Intensity")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Saturation count comparison
    axes[1, 1].hist(pid_summary["sat"], bins=20, alpha=0.5, label="PID", color="tab:red")
    axes[1, 1].hist(indi_summary["sat"], bins=20, alpha=0.5, label="INDI", color="tab:blue")
    axes[1, 1].set_xlabel("V-tail saturation events")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Actuator Saturation")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("monte_carlo_gusts.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()