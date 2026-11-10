"""Sensitivity sweep: how does transition duration affect performance?"""

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
from src.control.reference import TransitionReference


def run_transition(transition_time: float, duration: float = 30.0, dt: float = 0.01):
    """Run one transition and return a results dictionary."""
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)

    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)
    controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)
    ref = TransitionReference(start_altitude=100.0, target_speed=12.0, transition_time=transition_time)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    alt_log = []
    pitch_log = []
    t_log = []

    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])
        state = dyn.step(state, cmd["omega"], cmd["tilt"], cmd["delta_left"], cmd["delta_right"], dt)

        alt_log.append(-state.pos[2])
        pitch_log.append(np.rad2deg(state.att[1]))
        t_log.append(t)

    alt_log = np.array(alt_log)
    pitch_log = np.array(pitch_log)

    return {
        "t": np.array(t_log),
        "alt": alt_log,
        "pitch": pitch_log,
        "max_alt_loss": 100.0 - alt_log.min(),   # Positive = loss
        "final_alt": alt_log[-1],
        "max_pitch": np.max(np.abs(pitch_log)),
        "pitch_rms": np.sqrt(np.mean(pitch_log**2)),
    }


def main():
    transition_times = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    results = []

    print(f"{'T (s)':>6} | {'Max Alt Loss (m)':>18} | {'Final Alt (m)':>14} | {'Max |Pitch| (deg)':>20}")
    print("-" * 70)

    for T in transition_times:
        r = run_transition(T, duration=max(T + 10.0, 20.0))
        results.append({"T": T, **r})
        print(f"{T:>6.1f} | {r['max_alt_loss']:>18.2f} | {r['final_alt']:>14.2f} | {r['max_pitch']:>20.2f}")

    # --- Plot 1: Altitude loss vs T ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Ts = [r["T"] for r in results]
    losses = [r["max_alt_loss"] for r in results]
    max_pitches = [r["max_pitch"] for r in results]

    axes[0].plot(Ts, losses, 'o-', color='tab:red', linewidth=2, markersize=8)
    axes[0].axhline(10.0, color='green', linestyle='--', label="Excellent (<10m)")
    axes[0].axhline(20.0, color='orange', linestyle='--', label="Acceptable (<20m)")
    axes[0].axhline(50.0, color='red', linestyle='--', label="Danger (>50m)")
    axes[0].set_xlabel("Transition Duration T (s)")
    axes[0].set_ylabel("Max Altitude Loss (m)")
    axes[0].set_title("Altitude Loss vs Transition Duration")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # --- Plot 2: Max pitch vs T ---
    axes[1].plot(Ts, max_pitches, 's-', color='tab:blue', linewidth=2, markersize=8)
    axes[1].axhline(15.0, color='orange', linestyle='--', label="Pitch limit (15°)")
    axes[1].axhline(30.0, color='red', linestyle='--', label="Pitch danger (30°)")
    axes[1].set_xlabel("Transition Duration T (s)")
    axes[1].set_ylabel("Max |Pitch| (deg)")
    axes[1].set_title("Max Pitch Excursion vs Transition Duration")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("sensitivity_sweep.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()