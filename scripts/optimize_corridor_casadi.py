"""Optimize the transition corridor using CasADi + IPOPT."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


# --- Problem parameters ---
N = 50                      # Number of control segments (time samples)
T_TOTAL = 15.0              # Total time (s)
DT = T_TOTAL / N            # Timestep per segment
MAX_TILT_RATE = 0.35        # rad/s
ALT_MIN = 90.0              # Minimum altitude (m)
ALT_START = 100.0           # Starting altitude (m)
TILT_MAX = np.pi / 2        # 90 deg


# --- Dynamics approximation (simplified 1D altitude model) ---
# For optimization speed, we approximate the vertical dynamics using a
# simplified model (this is normal in trajectory optimization):
#   alt_dot = v_z
#   v_z_dot = (thrust_vert + lift - m*g) / m
#
# Where:
#   thrust_vert = k_thrust * omega^2 * cos(tilt)
#   lift = 0.5 * rho * wing_area * CL * V^2
#       (approximated as a function of tilt: 0 at 0 deg, max at 90 deg)
def altitude_dynamics(tilt, vz, alt):
    """Approximate vertical acceleration given state."""
    mass = 5.0
    g = 9.81
    k_thrust = 1e-5
    omega_hover = np.sqrt(mass * g / k_thrust)  # ~2215 rad/s
    thrust_vert = k_thrust * (omega_hover ** 2) * np.cos(tilt)
    
    # Lift builds up with tilt (approximation: linear in tilt)
    rho = 1.225
    wing_area = 1.5
    v_forward = 12.0 * (tilt / TILT_MAX)  # approximate speed
    CL = 0.5
    lift = 0.5 * rho * wing_area * CL * (v_forward ** 2) * np.sin(tilt)
    
    a_z = (thrust_vert + lift - mass * g) / mass
    return a_z


def main():
    print("Setting up CasADi optimization...")
    opti = ca.Opti()

    # --- Decision variables ---
    tilt = opti.variable(N + 1)      # tilt angle at each step
    tilt_dot = opti.variable(N)       # tilt rate
    vz = opti.variable(N + 1)         # vertical velocity
    alt = opti.variable(N + 1)        # altitude

    # --- Objective: minimize maximum altitude loss ---
    # Introduce a slack variable for max loss, minimize that
    max_loss = opti.variable()
    opti.subject_to(max_loss >= ALT_START - alt)
    opti.minimize(max_loss)

    # --- Constraints: dynamics ---
    for k in range(N):
        a_z = altitude_dynamics(tilt[k], vz[k], alt[k])
        opti.subject_to(alt[k + 1] == alt[k] + vz[k] * DT)
        opti.subject_to(vz[k + 1] == vz[k] + a_z * DT)
        opti.subject_to(tilt[k + 1] == tilt[k] + tilt_dot[k] * DT)

    # --- Boundary conditions ---
    opti.subject_to(tilt[0] == 0.0)
    opti.subject_to(tilt[N] == TILT_MAX)
    opti.subject_to(alt[0] == ALT_START)
    opti.subject_to(vz[0] == 0.0)
    opti.subject_to(vz[N] == 0.0)  # hover at end too

    # --- Bounds ---
    opti.subject_to(opti.bounded(0.0, tilt, TILT_MAX))
    opti.subject_to(opti.bounded(-MAX_TILT_RATE, tilt_dot, MAX_TILT_RATE))
    opti.subject_to(opti.bounded(ALT_MIN, alt, ALT_START + 10.0))
    opti.subject_to(opti.bounded(-5.0, vz, 5.0))

    # --- Solver options ---
    opti.solver(
        "ipopt",
        {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 500}
    )

    # --- Initial guess ---
    opti.set_initial(tilt, np.linspace(0, TILT_MAX, N + 1))
    opti.set_initial(tilt_dot, np.full(N, TILT_MAX / T_TOTAL))
    opti.set_initial(vz, np.zeros(N + 1))
    opti.set_initial(alt, np.full(N + 1, ALT_START))
    opti.set_initial(max_loss, 5.0)

    print("Solving with IPOPT (this may take 10-30 seconds)...")
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"\nIPOPT failed: {e}")
        print("Falling back to solution from last iterate (still useful)...")
        sol = opti.debug

    # --- Extract solution ---
    tilt_opt = np.array(sol.value(tilt))
    tilt_dot_opt = np.array(sol.value(tilt_dot))
    alt_opt = np.array(sol.value(alt))
    vz_opt = np.array(sol.value(vz))
    max_loss_val = float(sol.value(max_loss))

    print("\n=== Optimization Result ===")
    print(f"Max altitude loss: {max_loss_val:.3f} m")
    print(f"Max tilt rate used: {np.rad2deg(np.max(np.abs(tilt_dot_opt))):.2f} deg/s")
    print(f"Min altitude: {alt_opt.min():.2f} m")
    print(f"Final altitude: {alt_opt[-1]:.2f} m")
    print(f"Final tilt: {np.rad2deg(tilt_opt[-1]):.1f} deg")
    
    print("\nOptimal tilt schedule (sampled):")
    sample_idx = np.linspace(0, N, 8).astype(int)
    for i in sample_idx:
        print(f"  t={i * DT:5.2f}s  tilt={np.rad2deg(tilt_opt[i]):6.2f} deg")

    # --- Plot ---
    t_arr = np.linspace(0, T_TOTAL, N + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(t_arr, alt_opt, 'b-', linewidth=2)
    axes[0].axhline(ALT_MIN, color='orange', linestyle='--', label=f'Min allowed ({ALT_MIN}m)')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].set_title("Optimized Altitude Profile")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_arr, np.rad2deg(tilt_opt), 'g-', linewidth=2, label='Tilt')
    axes[1].plot(t_arr[:-1], np.rad2deg(tilt_dot_opt), 'r--', label='Tilt rate')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angle (deg or deg/s)")
    axes[1].set_title("Optimized Tilt Profile")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t_arr, vz_opt, 'm-', linewidth=2)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Vertical velocity (m/s)")
    axes[2].set_title("Optimized Vertical Velocity")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("casadi_corridor.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()