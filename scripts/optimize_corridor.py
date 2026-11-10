"""Optimize the tilt profile during transition using scipy."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.cascade import CascadeController
from src.control.reference import TransitionReference


# --- Simulation setup ---
N_PIECES = 5               # Number of tilt profile segments
DURATION = 20.0            # Fixed total time budget
DT = 0.02                  # Simulation timestep (coarser for speed)

MAX_TILT_RATE = 0.35       # rad/s = 20 deg/s
MAX_ALT_LOSS = 10.0        # meters
MAX_SPEED = 20.0           # m/s


def simulate(tilt_profile: np.ndarray, duration: float = DURATION) -> dict:
    """Run a full transition given a tilt profile sampled over N_PIECES segments.

    tilt_profile: array of N_PIECES values, each is tilt angle at the end of
                  the corresponding time segment. Should start at 0 and end at pi/2.
    """
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)

    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)
    controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)
    ref = TransitionReference(start_altitude=100.0, target_speed=12.0, transition_time=duration)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    # Build a tilt schedule from the piecewise profile
    dt_seg = duration / N_PIECES
    tilt_schedule = np.zeros(int(duration / DT) + 1)
    t = 0.0
    for i in range(int(duration / DT)):
        seg = min(int(t / dt_seg), N_PIECES - 1)
        tilt_schedule[i] = tilt_profile[seg]
        t += DT

    alt_history = []
    speed_history = []
    for i in range(int(duration / DT)):
        t = i * DT
        current_tilt = tilt_schedule[i]
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], current_tilt)
        state = dyn.step(state, cmd["omega"], current_tilt, cmd["delta_left"], cmd["delta_right"], DT)
        alt_history.append(-state.pos[2])
        speed_history.append(state.vel[0])

    alt_history = np.array(alt_history)
    speed_history = np.array(speed_history)

    return {
        "alt": alt_history,
        "speed": speed_history,
        "max_alt_loss": 100.0 - alt_history.min(),
        "max_speed": speed_history.max(),
        "final_tilt": tilt_profile[-1],
    }


def objective(tilt_profile: np.ndarray) -> float:
    """Objective: minimize max altitude loss (we want a stable transition)."""
    result = simulate(tilt_profile)
    return result["max_alt_loss"]


def constraint_tilt_rate(tilt_profile: np.ndarray) -> float:
    """Constraint: tilt rate between segments must not exceed MAX_TILT_RATE."""
    dt_seg = DURATION / N_PIECES
    max_rate = 0.0
    prev = 0.0
    for t in tilt_profile:
        rate = abs(t - prev) / dt_seg
        if rate > max_rate: max_rate = rate
        prev = t
    return MAX_TILT_RATE - max_rate  # >= 0 means satisfied


def constraint_tilt_end(tilt_profile: np.ndarray) -> float:
    """Constraint: final tilt must be 90 degrees (pi/2)."""
    return tilt_profile[-1] - (np.pi / 2)


def constraint_speed(tilt_profile: np.ndarray) -> float:
    """Constraint: max speed should not exceed MAX_SPEED."""
    result = simulate(tilt_profile)
    return MAX_SPEED - result["max_speed"]


def main():
    print("Running corridor optimization (this may take 1-2 minutes)...")

    # Initial guess: linear ramp
    x0 = np.linspace(0.0, np.pi / 2, N_PIECES)

    # Bounds: each tilt between 0 and 90 deg
    bounds = [(0.0, np.pi / 2) for _ in range(N_PIECES)]

    # Constraints
    constraints = [
        {"type": "ineq", "fun": constraint_tilt_rate},
        {"type": "eq", "fun": constraint_tilt_end},
        {"type": "ineq", "fun": constraint_speed},
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 100, "disp": True},
    )

    print("\n=== Optimization Result ===")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Optimal max altitude loss: {result.fun:.2f} m")
    print(f"Optimal tilt profile (deg): {np.rad2deg(result.x).round(1)}")

    # --- Simulate with optimal and compare against linear baseline ---
    linear_profile = np.linspace(0.0, np.pi / 2, N_PIECES)
    baseline = simulate(linear_profile)
    optimal = simulate(result.x)

    print("\n=== Comparison ===")
    print(f"{'Metric':<25} {'Linear':>12} {'Optimal':>12}")
    print(f"{'Max altitude loss (m)':<25} {baseline['max_alt_loss']:>12.2f} {optimal['max_alt_loss']:>12.2f}")
    print(f"{'Max speed (m/s)':<25} {baseline['max_speed']:>12.2f} {optimal['max_speed']:>12.2f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    times = np.linspace(0, DURATION, len(optimal["alt"]))
    t_linear = np.linspace(0, DURATION, len(baseline["alt"]))
    ax.plot(times, optimal["alt"], label="Optimal Tilt Profile", color="tab:blue")
    ax.plot(t_linear, baseline["alt"], label="Linear Tilt Profile", color="tab:red", linestyle="--")
    ax.axhline(100 - MAX_ALT_LOSS, color='orange', linestyle=':', label=f"Max allowed loss ({MAX_ALT_LOSS}m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Optimized vs Linear Transition Corridor")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("optimized_corridor.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()