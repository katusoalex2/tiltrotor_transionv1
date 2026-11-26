"""SIL Cross-Check: Compare the main simulation against an independent dynamics implementation."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

# --- IMPORTS FROM YOUR EXISTING PROJECT (only the controller and reference) ---
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.configuration.aircraft_models import build_aircraft_dynamics, build_collective_controller, step_collective
from src.control.trajectory import DEFAULT_CORRIDOR_PATH, ReferenceTrajectory

# --- INDEPENDENT DYNAMICS (written from scratch, no src.dynamics imports) ---

class IndependentDynamics:
    """A minimal, independent 6-DOF integrator for cross-checking."""
    def __init__(self, aircraft):
        self.aircraft = aircraft
        self.m = aircraft.mass
        self.I = np.array([aircraft.inertia_xx, aircraft.inertia_yy, aircraft.inertia_zz])
        self.g = aircraft.gravity
        # State: pos(3), vel(3), euler(3), omega(3)
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)  # roll, pitch, yaw
        self.omega = np.zeros(3)
        self.tilt = 0.0

    def step(self, omega, tilt, delta_vtail, dt):
        # Simple Euler integration (independent of your semi-implicit implementation)
        roll, pitch, yaw = self.euler

        # 1. Thrust in body frame (tilt=0 -> -Z up, tilt=pi/2 -> +X forward)
        # Using the same sign convention as your main code for fair comparison
        thrust = self.aircraft.k_thrust * min(max(omega, 0.0), self.aircraft.max_omega)**2
        Fx_body = 0.0
        Fz_body = 0.0
        rotor_pitch_moment = 0.0
        pivot_x, pivot_z = self.aircraft.tilt_pivot_xz_m
        for x, _y, z, tilts in self.aircraft.motor_positions:
            rotor_tilt = tilt if tilts else 0.0
            if tilts:
                rel_x, rel_z = x - pivot_x, z - pivot_z
                x = pivot_x + rel_x * np.cos(tilt) - rel_z * np.sin(tilt)
                z = pivot_z + rel_x * np.sin(tilt) + rel_z * np.cos(tilt)
            fx = thrust * np.sin(rotor_tilt)
            fz = -thrust * np.cos(rotor_tilt)
            Fx_body += fx
            Fz_body += fz
            rotor_pitch_moment += z * fx - x * fz

        # 2. V-tail pitch moment (simplified linear model, independent derivation)
        # Positive delta -> nose down (same as your code)
        My = rotor_pitch_moment - 50.0 * delta_vtail * (abs(self.vel[0]) + 1.0)

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
        self.tilt = float(tilt)

        return self

def run_main_sim(duration, dt, controller, ref):
    """Run the main simulation (uses your existing dynamics via a simplified wrapper)."""
    from src.dynamics.state import AircraftState
    dyn = build_aircraft_dynamics(load_aircraft_parameters())

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    log = {"t": [], "alt": [], "pitch": []}
    for i in range(int(duration/dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])
        state = step_collective(dyn, state, cmd, dt)
        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["pitch"].append(np.rad2deg(state.att[1]))
    return {k: np.array(v) for k,v in log.items()}

def run_independent_sim(duration, dt, controller, ref):
    """Run the independent dynamics with the same controller commands."""
    dyn = IndependentDynamics(load_aircraft_parameters())
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
        s.tilt = dyn.tilt

        cmd = controller.compute(t, s, r["altitude"], r["speed"], r["tilt"])

        dyn.step(cmd["omega"], cmd["tilt"], cmd["delta_left"], dt)
        log["t"].append(t)
        log["alt"].append(-dyn.pos[2])
        log["pitch"].append(np.rad2deg(dyn.euler[1]))
    return {k: np.array(v) for k,v in log.items()}

def main():
    print("Running SIL cross-check...")
    dt = 0.01
    duration = 15.0
    ref = ReferenceTrajectory(DEFAULT_CORRIDOR_PATH, speed_max=12.0)
    aircraft = load_aircraft_parameters()
    main_log = run_main_sim(duration, dt, build_collective_controller(aircraft), ref)
    indep_log = run_independent_sim(duration, dt, build_collective_controller(aircraft), ref)

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
