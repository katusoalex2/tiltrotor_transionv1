"""Master simulation: The final robust PID transition with wind gusts."""

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
from src.control.trajectory import ReferenceTrajectory
from src.control.pid import PIDController
from src.wind.gust import DrydenGust

class PIDCascade:
    """The final winning controller architecture."""
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

        return {"omega": omega_cmd, "tilt": ref_tilt, "delta_left": vtail_cmd,
                "delta_right": vtail_cmd}

def main():
    print("Starting MASTER SIMULATION (PID + Wind Gusts + Robust Corridor)...")
    dt = 0.01
    duration = 20.0

    # 1. Setup the aircraft
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)

    # 2. Setup the winning PID controller
    controller = PIDCascade(mass=5.0, k_thrust=1e-5)

    # 3. Load the optimized robust corridor
    ref = ReferenceTrajectory("configs/robust_corridor.csv", speed_max=12.0)

    # 4. Setup a 3.5 m/s Dryden wind gust
    wind = DrydenGust(sigma=(3.5, 3.5, 2.1), airspeed=12.0, seed=99)

    # 5. Initial state
    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    log = {"t": [], "alt": [], "ref_alt": [], "pitch": [], "wind": []}

    print("Flying with 3.5 m/s wind gusts...")
    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])

        # Wind injection
        wind_vec = wind.step(dt)
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
        state.vel = state.vel - wind_vec

        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["ref_alt"].append(r["altitude"])
        log["pitch"].append(np.rad2deg(state.att[1]))
        log["wind"].append(np.linalg.norm(wind_vec))

    # 6. Plot final results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(log["t"], log["alt"], 'b-', linewidth=2, label="Actual Altitude")
    ax1.plot(log["t"], log["ref_alt"], 'k--', label="Reference Corridor")
    ax1.set_title("Altitude Tracking (with 3.5 m/s gusts)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Altitude (m)")
    ax1.grid(True); ax1.legend()

    ax2.plot(log["t"], log["pitch"], 'r-', linewidth=2)
    ax2.axhline(0, color='k', linestyle=':')
    ax2.set_title("Pitch Excursion")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Pitch (deg)")
    ax2.grid(True)

    ax3.plot(log["t"], log["wind"], 'g-', linewidth=1.5)
    ax3.set_title("Dryden Wind Gust Magnitude")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Wind Speed (m/s)")
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("master_simulation_final.png", dpi=100)
    print("Saved final master plot to master_simulation_final.png")
    plt.show()

if __name__ == "__main__":
    main()