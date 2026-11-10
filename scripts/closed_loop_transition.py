"""Closed-loop transition run with cascade controller and gain scheduling."""

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


def run_transition(transition_time: float = 10.0, duration: float = 15.0, dt: float = 0.01):
    """Run one closed-loop transition and return a data log."""
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)

    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)
    controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)
    ref = TransitionReference(start_altitude=100.0, target_speed=20.0, transition_time=transition_time)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    log = {
        "t": [], "alt": [], "speed": [], "pitch": [], "desired_pitch": [],
        "tilt": [], "omega": [], "vtail": [], "lift": [], "drag": [],
    }

    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])

        state = dyn.step(state, cmd["omega"], cmd["tilt"], cmd["delta_left"], cmd["delta_right"], dt)

        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["speed"].append(state.vel[0])
        log["pitch"].append(np.rad2deg(state.att[1]))
        log["desired_pitch"].append(np.rad2deg(cmd["desired_pitch"]))
        log["tilt"].append(np.rad2deg(cmd["tilt"]))
        log["omega"].append(cmd["omega"])
        log["vtail"].append(cmd["delta_left"])
        log["lift"].append(state.lift)
        log["drag"].append(state.drag)

    return {k: np.array(v) for k, v in log.items()}


def analyze(log):
    """Compute performance metrics."""
    alt = log["alt"]
    pitch = log["pitch"]
    vtail = log["vtail"]
    omega = log["omega"]

    print("\n=== Transition Performance Metrics ===")
    print(f"Initial altitude:       {alt[0]:.1f} m")
    print(f"Final altitude:         {alt[-1]:.1f} m")
    print(f"Max altitude loss:      {alt.min() - 100.0:+.2f} m")
    print(f"Max altitude gain:      {alt.max() - 100.0:+.2f} m")
    print(f"Max |pitch|:            {np.max(np.abs(pitch)):.2f} deg")
    print(f"Pitch RMS:              {np.sqrt(np.mean(pitch**2)):.2f} deg")
    print(f"V-tail RMS deflection:  {np.rad2deg(np.sqrt(np.mean(vtail**2))):.2f} deg")
    print(f"V-tail peak deflection: {np.rad2deg(np.max(np.abs(vtail))):.2f} deg")
    print(f"Motor omega range:      [{omega.min():.0f}, {omega.max():.0f}] rad/s")


def main():
    print("Running closed-loop transition...")
    log = run_transition(transition_time=10.0, duration=15.0)
    analyze(log)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(log["t"], log["alt"], 'b-')
    axes[0, 0].axhline(100, color='r', linestyle='--', label="Target")
    axes[0, 0].set_title("Altitude")
    axes[0, 0].set_ylabel("m"); axes[0, 0].grid(True); axes[0, 0].legend()

    axes[0, 1].plot(log["t"], log["speed"], 'g-')
    axes[0, 1].axhline(20, color='r', linestyle='--', label="Target")
    axes[0, 1].set_title("Forward Speed")
    axes[0, 1].set_ylabel("m/s"); axes[0, 1].grid(True); axes[0, 1].legend()

    axes[0, 2].plot(log["t"], log["tilt"], 'm-')
    axes[0, 2].set_title("Rotor Tilt")
    axes[0, 2].set_ylabel("deg"); axes[0, 2].grid(True)

    axes[1, 0].plot(log["t"], log["pitch"], 'r-', label="Actual")
    axes[1, 0].plot(log["t"], log["desired_pitch"], 'k--', label="Desired")
    axes[1, 0].set_title("Pitch Angle")
    axes[1, 0].set_ylabel("deg"); axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].grid(True); axes[1, 0].legend()

    axes[1, 1].plot(log["t"], np.rad2deg(log["vtail"]), 'c-')
    axes[1, 1].set_title("V-Tail Deflection")
    axes[1, 1].set_ylabel("deg"); axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True)

    axes[1, 2].plot(log["t"], log["omega"], 'orange')
    axes[1, 2].set_title("Motor Command (omega)")
    axes[1, 2].set_ylabel("rad/s"); axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig("closed_loop_transition.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()