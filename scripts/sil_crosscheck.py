"""SIL Cross-Check: Compare the main simulation against an independent dynamics implementation."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

# --- IMPORTS FROM YOUR EXISTING PROJECT (only the controller and reference) ---
from src.control.cascade import CascadeController
from src.control.trajectory import ReferenceTrajectory

# --- INDEPENDENT DYNAMICS (written from scratch, no src.dynamics imports) ---

class IndependentDynamics:
    """A minimal, independent 6-DOF integrator for cross-checking."""
    def __init__(self, mass=5.0, inertia=(0.1, 0.1, 0.2)):
        self.m = mass
        self.I = np.array(inertia)
        self.g = 9.81
        # State: pos(3), vel(3), euler(3), omega(3)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)  # roll, pitch, yaw
        self.omega = np.zeros(3)

    def step(self, thrust, tilt, delta_vtail, dt):
        # Simple Euler integration (independent of your semi-implicit implementation)
        roll, pitch, yaw = self.euler

        # 1. Thrust in body frame (tilt=0 -> -Z up, tilt=pi/2 -> +X forward)
        # Using the same sign convention as your main code for fair comparison
        Fx_body = thrust * np.sin(tilt)
        Fz_body = -thrust * np.cos(tilt)

        # 2. V-tail pitch moment (simplified linear model, independent derivation)
        # Positive delta -> nose down (same as your code)
        My = -50.0 * delta_vtail * (abs(self.vel[0]) + 1.0)  # crude speed-dependent moment

        # 3. Rotate body forces to world frame (standard rotation matrix)
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)

        # Rotation matrix body->world (ZYX convention, same as yours)
        R = np.array([
            [c_y*c_p, c_y*s_p*s_r - s_y*c_r, c_y*s_p*c_r + s_y*s_r],
            [s_y*c_p, s_y*s_p*s_r + c_y*c_r, s_y*s_p*c_r - c_y*s_r],
            [-s_p,    c_p*s_r,               c_p*c_r]
        ])

        F_body = np.array([Fx_body, 0.0, Fz_body])
        F_world = R @ F_body

        # 4. Add gravity (NED: +Z down)
        F_world[2] += self.m * self.g

        # 5. Linear acceleration
        accel = F_world / self.m

        # 6. Angular acceleration (simplified: only pitch moment matters here)
        alpha_pitch = My / self.I[1]

        # 7. Integrate (explicit Euler, independent of your method)
        self.vel += accel * dt
        self.pos += self.vel * dt
        self.omega[1] += alpha_pitch * dt
        # Simplified kinematics: euler rate = body rate (valid for small angles)
        self.euler += self.omega * dt

        return self

def run_main_sim(duration, dt, controller, ref):
    """Run the main simulation (uses your existing dynamics via a simplified wrapper)."""
    from src.dynamics.state import AircraftState
    from src.dynamics.rotor import RotorModel
    from src.dynamics.tiltrotor import TiltrotorDynamics
    from src.aero.vtail import VtailModel
    from src.aero.aerodynamics import AeroModel

    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1,0.1,0.2]), aero=aero)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    log = {"t": [], "alt": [], "pitch": []}
    for i in range(int(duration/dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["pitch"].append(np.rad2deg(state.att[1]))
    return {k: np.array(v) for k,v in log.items()}

def run_independent_sim(duration, dt, controller, ref):
    """Run the independent dynamics with the same controller commands."""
    dyn = IndependentDynamics()
    dyn.pos = np.array([0.0, 0.0, -100.0])

    log = {"t": [], "alt": [], "pitch": []}
    for i in range(int(duration/dt)):
        t = i * dt
        r = ref.at(t)
        # We need a 'state' object with the same attributes the controller expects
        class SimpleState:
            pass
        s = SimpleState()
        s.pos = dyn.pos
        s.vel = dyn.vel
        s.att = dyn.euler
        s.omega = dyn.omega

        cmd = controller.compute(t, s, r["altitude"], r["speed"], r["tilt"])

        # Convert omega_cmd to thrust for the independent model
        # (approximate: assume hover thrust at omega_hover, scale by (omega/omega_hover)^2)
        thrust = 1e-5 * (cmd["omega"] ** 2)

        dyn.step(thrust, r["tilt"], cmd["delta_left"], dt)
        log["t"].append(t)
        log["alt"].append(-dyn.pos[2])
        log["pitch"].append(np.rad2deg(dyn.euler[1]))
    return {k: np.array(v) for k,v in log.items()}

def main():
    print("Running SIL cross-check...")
    dt = 0.01
    duration = 15.0
    ref = ReferenceTrajectory("configs/robust_corridor.csv", speed_max=12.0)
    controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=np.sqrt(5.0*9.81/1e-5))

    main_log = run_main_sim(duration, dt, controller, ref)
    indep_log = run_independent_sim(duration, dt, controller, ref)

    # Comparison
    print("\n=== SIL Cross-Check Comparison ===")
    print(f"{'Metric':<25} {'Main Sim':>12} {'Independent':>12} {'Diff':>10}")
    print("-" * 62)
    print(f"{'Max Alt Loss (m)':<25} {100-main_log['alt'].min():>12.2f} {100-indep_log['alt'].min():>12.2f} {abs((100-main_log['alt'].min())-(100-indep_log['alt'].min())):>10.2f}")
    print(f"{'Max |Pitch| (deg)':<25} {np.max(np.abs(main_log['pitch'])):>12.2f} {np.max(np.abs(indep_log['pitch'])):>12.2f} {abs(np.max(np.abs(main_log['pitch']))-np.max(np.abs(indep_log['pitch']))):>10.2f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    ax1.plot(main_log["t"], main_log["alt"], 'b-', label="Main Sim")
    ax1.plot(indep_log["t"], indep_log["alt"], 'r--', label="Independent")
    ax1.set_title("Altitude Comparison")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Altitude (m)")
    ax1.grid(True); ax1.legend()

    ax2.plot(main_log["t"], main_log["pitch"], 'b-', label="Main Sim")
    ax2.plot(indep_log["t"], indep_log["pitch"], 'r--', label="Independent")
    ax2.set_title("Pitch Comparison")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Pitch (deg)")
    ax2.grid(True); ax2.legend()

    plt.tight_layout()
    plt.savefig("sil_crosscheck.png", dpi=100)
    print("\nSaved plot: sil_crosscheck.png")
    plt.show()

if __name__ == "__main__":
    main()
