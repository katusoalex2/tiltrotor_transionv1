"""Run a transition using the optimized robust corridor as reference."""

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


def run_with_reference(reference, duration=20.0, dt=0.01):
    """Run a transition following the given reference trajectory."""
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)

    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)
    controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    log = {"t": [], "alt": [], "speed": [], "pitch": [], "tilt": [], "ref_alt": [], "ref_tilt": []}

    for i in range(int(duration / dt)):
        t = i * dt
        r = reference.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)

        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["speed"].append(state.vel[0])
        log["pitch"].append(np.rad2deg(state.att[1]))
        log["tilt"].append(np.rad2deg(r["tilt"]))
        log["ref_alt"].append(r["altitude"])
        log["ref_tilt"].append(np.rad2deg(r["tilt"]))

    return {k: np.array(v) for k, v in log.items()}


def main():
    corridor_path = "configs/robust_corridor.csv"
    if not os.path.exists(corridor_path):
        print(f"ERROR: {corridor_path} not found.")
        print("Run scripts/robust_corridor.py first to generate it.")
        return

    print("Running transition with OPTIMIZED corridor reference...")
    ref_opt = ReferenceTrajectory(corridor_path, speed_max=12.0)
    log_opt = run_with_reference(ref_opt, duration=ref_opt.t_end + 5.0)

    print("Running transition with LINEAR reference (comparison)...")
    ref_lin = LinearReference(start_altitude=100.0, target_speed=12.0, transition_time=ref_opt.t_end)
    log_lin = run_with_reference(ref_lin, duration=ref_opt.t_end + 5.0)

    # Comparison metrics
    print("\n=== Optimized vs Linear (Closed-Loop) ===")
    print(f"{'Metric':<28} {'Optimized':>12} {'Linear':>12}")
    print("-" * 56)
    print(f"{'Max altitude loss (m)':<28} {100 - log_opt['alt'].min():>12.2f} {100 - log_lin['alt'].min():>12.2f}")
    print(f"{'Max |pitch| (deg)':<28} {np.max(np.abs(log_opt['pitch'])):>12.2f} {np.max(np.abs(log_lin['pitch'])):>12.2f}")
    print(f"{'Final altitude (m)':<28} {log_opt['alt'][-1]:>12.2f} {log_lin['alt'][-1]:>12.2f}")
    print(f"{'Final speed (m/s)':<28} {log_opt['speed'][-1]:>12.2f} {log_lin['speed'][-1]:>12.2f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(log_opt["t"], log_opt["tilt"], 'b-', linewidth=2, label="Optimized corridor")
    axes[0].plot(log_lin["t"], log_lin["tilt"], 'r--', linewidth=2, label="Linear")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Tilt (deg)")
    axes[0].set_title("Tilt Reference")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(log_opt["t"], log_opt["alt"], 'b-', linewidth=2, label="Actual altitude")
    axes[1].plot(log_opt["t"], log_opt["ref_alt"], 'k:', linewidth=1, label="Reference altitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Altitude Tracking (Optimized)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(log_opt["t"], log_opt["speed"], 'g-', linewidth=2, label="Optimized")
    axes[2].plot(log_lin["t"], log_lin["speed"], 'r--', linewidth=2, label="Linear")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Forward speed (m/s)")
    axes[2].set_title("Forward Speed")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("corridor_reference_run.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()