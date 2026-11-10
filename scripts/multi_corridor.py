"""Generate three different optimal corridors with different objectives."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


N = 50
T_TOTAL = 15.0
DT = T_TOTAL / N
MAX_TILT_RATE = 0.35
ALT_START = 100.0
TILT_MAX = np.pi / 2


def altitude_dynamics(tilt, vz):
    """Approximate vertical acceleration given tilt angle."""
    mass = 5.0
    g = 9.81
    k_thrust = 1e-5
    omega_hover = np.sqrt(mass * g / k_thrust)
    thrust_vert = k_thrust * (omega_hover ** 2) * np.cos(tilt)
    rho = 1.225
    wing_area = 1.5
    v_forward = 12.0 * (tilt / TILT_MAX)
    CL = 0.5
    lift = 0.5 * rho * wing_area * CL * (v_forward ** 2) * np.sin(tilt)
    return (thrust_vert + lift - mass * g) / mass


def solve_corridor(objective_type: str):
    """Solve the corridor optimization with a specific objective."""
    opti = ca.Opti()

    tilt = opti.variable(N + 1)
    tilt_dot = opti.variable(N)
    vz = opti.variable(N + 1)
    alt = opti.variable(N + 1)

    # Slack variables for tracking metrics
    max_loss = opti.variable()
    total_tilt_effort = 0
    for k in range(N):
        total_tilt_effort += tilt_dot[k] ** 2

    # --- Constraints (always the same) ---
    for k in range(N):
        a_z = altitude_dynamics(tilt[k], vz[k])
        opti.subject_to(alt[k + 1] == alt[k] + vz[k] * DT)
        opti.subject_to(vz[k + 1] == vz[k] + a_z * DT)
        opti.subject_to(tilt[k + 1] == tilt[k] + tilt_dot[k] * DT)

    opti.subject_to(tilt[0] == 0.0)
    opti.subject_to(tilt[N] == TILT_MAX)
    opti.subject_to(alt[0] == ALT_START)
    opti.subject_to(vz[0] == 0.0)
    opti.subject_to(vz[N] == 0.0)
    opti.subject_to(opti.bounded(0.0, tilt, TILT_MAX))
    opti.subject_to(opti.bounded(-MAX_TILT_RATE, tilt_dot, MAX_TILT_RATE))
    opti.subject_to(max_loss >= ALT_START - alt)

    # --- Objective selection ---
    if objective_type == "fast":
        # Minimize time spent away from cruise: tilt quickly at start
        # Approximate: minimize integral of (1 - tilt/TILT_MAX) * (1 - altitude weight)
        # Simpler: minimize tilt-area (rush to cruise)
        obj = total_tilt_effort * 0 + max_loss  # baseline: min loss but with tilt-rate cap
        # Better: reward early tilt
        for k in range(N):
            obj += 0.1 * (TILT_MAX - tilt[k]) * DT
    elif objective_type == "safe":
        # Minimize max altitude loss
        obj = max_loss
    elif objective_type == "smooth":
        # Minimize actuator effort (tilt rate squared)
        obj = total_tilt_effort
    else:
        raise ValueError(f"Unknown objective: {objective_type}")

    opti.minimize(obj)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 500})

    opti.set_initial(tilt, np.linspace(0, TILT_MAX, N + 1))
    opti.set_initial(tilt_dot, np.full(N, TILT_MAX / T_TOTAL))
    opti.set_initial(vz, np.zeros(N + 1))
    opti.set_initial(alt, np.full(N + 1, ALT_START))
    opti.set_initial(max_loss, 5.0)

    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug

    return {
        "tilt": np.array(sol.value(tilt)),
        "tilt_dot": np.array(sol.value(tilt_dot)),
        "alt": np.array(sol.value(alt)),
        "vz": np.array(sol.value(vz)),
        "max_loss": float(sol.value(max_loss)),
        "total_effort": float(np.sum(np.array(sol.value(tilt_dot)) ** 2)),
    }


def main():
    print("Solving three corridors (fast / safe / smooth)...")

    results = {}
    for label in ["fast", "safe", "smooth"]:
        print(f"\n--- Solving: {label} ---")
        try:
            results[label] = solve_corridor(label)
            r = results[label]
            print(f"  Max altitude loss: {r['max_loss']:.2f} m")
            print(f"  Total tilt effort: {r['total_effort']:.4f}")
            print(f"  Max tilt rate: {np.rad2deg(np.max(np.abs(r['tilt_dot']))):.2f} deg/s")
        except Exception as e:
            print(f"  Failed: {e}")

    # --- Plot all corridors ---
    t_arr = np.linspace(0, T_TOTAL, N + 1)
    colors = {"fast": "tab:red", "safe": "tab:green", "smooth": "tab:blue"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Tilt profiles
    for label, r in results.items():
        axes[0].plot(t_arr, np.rad2deg(r["tilt"]), color=colors[label], linewidth=2, label=label)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Tilt angle (deg)")
    axes[0].set_title("Tilt Profiles")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Altitude profiles
    for label, r in results.items():
        axes[1].plot(t_arr, r["alt"], color=colors[label], linewidth=2, label=label)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Altitude Profiles")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Tilt rates
    for label, r in results.items():
        axes[2].plot(t_arr[:-1], np.rad2deg(r["tilt_dot"]), color=colors[label], linewidth=2, label=label)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Tilt rate (deg/s)")
    axes[2].set_title("Actuator Effort (Tilt Rate)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("multi_corridor.png", dpi=100)
    plt.show()

    # --- Summary table ---
    print("\n=== Summary Table ===")
    print(f"{'Objective':<10} | {'Max Alt Loss (m)':>18} | {'Total Effort':>14} | {'Max Tilt Rate (deg/s)':>22}")
    print("-" * 75)
    for label, r in results.items():
        print(f"{label:<10} | {r['max_loss']:>18.2f} | {r['total_effort']:>14.4f} | {np.rad2deg(np.max(np.abs(r['tilt_dot']))):>22.2f}")


if __name__ == "__main__":
    main()
    