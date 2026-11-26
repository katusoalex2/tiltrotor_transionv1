"""Robust corridor: nominal vs wind-margin-enhanced."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.reduced_transition import altitude_acceleration


N = 50
T_TOTAL = 15.0
DT = T_TOTAL / N
TILT_MAX = np.pi / 2
ALT_START = 100.0


def altitude_dynamics(tilt, vz, tilt_rate_extra=0.0, aircraft=None):
    """Approximate vertical acceleration given tilt angle."""
    return altitude_acceleration(tilt, aircraft if aircraft is not None else load_aircraft_parameters()) + tilt_rate_extra


def solve_corridor(robust: bool):
    """Solve corridor optimization. If robust=True, tighten constraints."""
    aircraft = load_aircraft_parameters()
    opti = ca.Opti()

    tilt = opti.variable(N + 1)
    tilt_dot = opti.variable(N)
    vz = opti.variable(N + 1)
    alt = opti.variable(N + 1)
    max_loss = opti.variable()

    # --- Base dynamics constraints ---
    for k in range(N):
        a_z = altitude_dynamics(tilt[k], vz[k], aircraft=aircraft)
        opti.subject_to(alt[k + 1] == alt[k] + vz[k] * DT)
        opti.subject_to(vz[k + 1] == vz[k] + a_z * DT)
        opti.subject_to(tilt[k + 1] == tilt[k] + tilt_dot[k] * DT)

    # --- Boundary conditions ---
    opti.subject_to(tilt[0] == 0.0)
    opti.subject_to(tilt[N] == TILT_MAX)
    opti.subject_to(alt[0] == ALT_START)
    opti.subject_to(vz[0] == 0.0)
    opti.subject_to(vz[N] == 0.0)

    # --- Constraint: TILT RATE LIMIT ---
    if robust:
        # Reduced-rate scenario: 85% of configured maximum.
        max_tilt_rate = aircraft.max_tilt_rate * 0.85
    else:
        max_tilt_rate = aircraft.max_tilt_rate

    opti.subject_to(opti.bounded(0.0, tilt, TILT_MAX))
    opti.subject_to(opti.bounded(-max_tilt_rate, tilt_dot, max_tilt_rate))

    # --- Constraint: ALTITUDE LOSS BOUND ---
    if robust:
        # ROBUST: don't allow more than 10 m altitude loss (achievable)
        alt_floor = ALT_START - 10.0
    else:
        alt_floor = ALT_START - 10.0

    opti.subject_to(opti.bounded(alt_floor, alt, ALT_START + 10.0))

    # Objective: minimize max altitude loss
    opti.subject_to(max_loss >= ALT_START - alt)
    opti.minimize(max_loss)

    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 500})

    opti.set_initial(tilt, np.linspace(0, TILT_MAX, N + 1))
    opti.set_initial(tilt_dot, np.full(N, TILT_MAX / T_TOTAL))
    opti.set_initial(vz, np.zeros(N + 1))
    opti.set_initial(alt, np.full(N + 1, ALT_START))
    opti.set_initial(max_loss, 5.0)

    sol = opti.solve()

    return {
        "tilt": np.array(sol.value(tilt)),
        "tilt_dot": np.array(sol.value(tilt_dot)),
        "alt": np.array(sol.value(alt)),
        "vz": np.array(sol.value(vz)),
        "max_loss": float(sol.value(max_loss)),
        "max_tilt_rate_used": float(np.max(np.abs(np.array(sol.value(tilt_dot))))),
    }


def main():
    print("Solving NOMINAL corridor (full tilt rate, 10m loss allowed)...")
    nominal = solve_corridor(robust=False)
    print(f"  Max altitude loss: {nominal['max_loss']:.2f} m")
    print(f"  Max tilt rate used: {np.rad2deg(nominal['max_tilt_rate_used']):.2f} deg/s")

    print("\nSolving ROBUST corridor (85% tilt rate cap, 10m loss allowed)...")
    robust = solve_corridor(robust=True)
    print(f"  Max altitude loss: {robust['max_loss']:.2f} m")
    print(f"  Max tilt rate used: {np.rad2deg(robust['max_tilt_rate_used']):.2f} deg/s")

    # --- Comparison table ---
    print("\n=== Nominal vs Robust ===")
    print(f"{'Metric':<30} {'Nominal':>12} {'Robust':>12}")
    print("-" * 58)
    print(f"{'Max altitude loss (m)':<30} {nominal['max_loss']:>12.2f} {robust['max_loss']:>12.2f}")
    print(f"{'Max tilt rate (deg/s)':<30} {np.rad2deg(nominal['max_tilt_rate_used']):>12.2f} {np.rad2deg(robust['max_tilt_rate_used']):>12.2f}")
    print(f"{'Final altitude (m)':<30} {nominal['alt'][-1]:>12.2f} {robust['alt'][-1]:>12.2f}")

    # --- Plot ---
    t_arr = np.linspace(0, T_TOTAL, N + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(t_arr, np.rad2deg(nominal["tilt"]), 'b-', linewidth=2, label="Nominal")
    axes[0].plot(t_arr, np.rad2deg(robust["tilt"]), 'r--', linewidth=2, label="Robust")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Tilt angle (deg)")
    axes[0].set_title("Tilt Profiles: Nominal vs Robust")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_arr, nominal["alt"], 'b-', linewidth=2, label="Nominal")
    axes[1].plot(t_arr, robust["alt"], 'r--', linewidth=2, label="Robust")
    axes[1].axhline(90, color='orange', linestyle=':', label="Nominal floor (90m)")
    axes[1].axhline(90, color='red', linestyle=':', label="Robust floor (90m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Altitude Profiles: Nominal vs Robust")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t_arr[:-1], np.rad2deg(nominal["tilt_dot"]), 'b-', linewidth=2, label="Nominal")
    axes[2].plot(t_arr[:-1], np.rad2deg(robust["tilt_dot"]), 'r--', linewidth=2, label="Robust")
    configured_rate = np.rad2deg(load_aircraft_parameters().max_tilt_rate)
    axes[2].axhline(configured_rate, color='k', linestyle=':', label="Configured limit")
    axes[2].axhline(0.85 * configured_rate, color='red', linestyle=':', label="85% cap")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Tilt rate (deg/s)")
    axes[2].set_title("Tilt Rate: Nominal vs Robust")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("robust_corridor.png", dpi=100)
    plt.show()

    # --- Save the robust corridor as the final specification ---
    corridor_path = os.path.join(os.path.dirname(__file__), "..", "UAM Configurations", "robust_corridor.csv")
    np.savetxt(
        corridor_path,
        np.column_stack([t_arr, np.rad2deg(robust["tilt"]), robust["alt"]]),
        delimiter=",",
        header="time_s,tilt_deg,altitude_m",
        comments="",
    )
    print(f"\nSaved robust corridor to {corridor_path}")


if __name__ == "__main__":
    main()
