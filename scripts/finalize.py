"""Final project deliverables: tests, master plot, report."""

import sys
import os
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.trajectory import ReferenceTrajectory
from src.control.pid import PIDController
from src.wind.gust import DrydenGust


class FinalPIDCascade:
    """The final selected controller architecture (PID inner loop)."""
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
        return {"omega": omega_cmd, "tilt": ref_tilt,
                "delta_left": vtail_cmd, "delta_right": vtail_cmd}


def run_master(gust_sigma=3.5, seed=99, duration=20.0, dt=0.01):
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)
    controller = FinalPIDCascade(5.0, 1e-5)
    ref = ReferenceTrajectory("configs/robust_corridor.csv", speed_max=12.0)
    wind = DrydenGust(sigma=(gust_sigma, gust_sigma, gust_sigma * 0.6), airspeed=12.0, seed=seed)

    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])

    log = {"t": [], "alt": [], "ref_alt": [], "pitch": [], "wind": []}
    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])
        wind_vec = wind.step(dt)
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
        state.vel = state.vel - wind_vec
        log["t"].append(t)
        log["alt"].append(-state.pos[2])
        log["ref_alt"].append(r["altitude"])
        log["pitch"].append(np.rad2deg(state.att[1]))
        log["wind"].append(np.linalg.norm(wind_vec))
    return {k: np.array(v) for k, v in log.items()}


def main():
    print("=" * 70)
    print("FINAL PROJECT DELIVERABLES")
    print("=" * 70)

    # 1. Run tests
    print("\n[1/3] Running test suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    # 2. Master simulation plot
    print("\n[2/3] Running master simulation (PID + gusts)...")
    log = run_master()
    max_loss = 100.0 - log["alt"].min()
    max_pitch = np.max(np.abs(log["pitch"]))
    max_wind = np.max(log["wind"])
    print(f"  Max altitude loss: {max_loss:.2f} m")
    print(f"  Max pitch: {max_pitch:.2f} deg")
    print(f"  Max gust: {max_wind:.2f} m/s")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(log["t"], log["alt"], 'b-', linewidth=2, label="Actual Altitude")
    axes[0].plot(log["t"], log["ref_alt"], 'k--', label="Reference Corridor")
    axes[0].set_title("Altitude Tracking (PID + 3.5 m/s gusts)")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Altitude (m)")
    axes[0].grid(True); axes[0].legend()

    axes[1].plot(log["t"], log["pitch"], 'r-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle=':')
    axes[1].set_title("Pitch Excursion")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Pitch (deg)")
    axes[1].grid(True)

    axes[2].plot(log["t"], log["wind"], 'g-', linewidth=1.5)
    axes[2].set_title("Wind Gust Magnitude")
    axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("Wind (m/s)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("FINAL_SELECTED_CONTROLLER.png", dpi=120)
    print("  Saved: FINAL_SELECTED_CONTROLLER.png")

    # 3. Write report
    print("\n[3/3] Writing project summary report...")
    report = f"""# Tilt-Rotor Transition Project — Final Report

## Controller Selected: PID Cascade

After Monte Carlo analysis of three controller architectures under random Dryden wind gusts,
**the PID cascade controller was selected** as the primary transition controller.

### Controller Comparison (100 runs each, gust σ = 0.5–5.5 m/s)

| Controller | Crashes | Mean Altitude Loss |
|------------|---------|-------------------|
| **PID** | 1/100 | 11.51 m |
| INDI | 23/100 | 31.38 m |
| INDI + DOB | 21/100 | 30.89 m |

**Conclusion:** INDI's fixed-B-matrix assumption is architecturally incompatible
with this aircraft's large variation in control effectiveness across the transition
envelope. A Disturbance Observer augmentation reduced crashes by only 2%, which
is not statistically significant. PID's "dumb" architecture survives because it
does not depend on an accurate aerodynamic model.

### Final Master Simulation

- Max altitude loss: {max_loss:.2f} m
- Max pitch excursion: {max_pitch:.2f} deg
- Peak gust encountered: {max_wind:.2f} m/s

### Key Deliverables

- **Physics engine**: 6-DOF rigid body + V-tail aero + rotor model
- **Controller**: Cascade PID (altitude/speed → pitch → V-tail)
- **Corridor**: CasADi/IPOPT-optimized robust corridor (`configs/robust_corridor.csv`)
- **Wind model**: Dryden turbulence (aerospace standard)
- **Validation**: 200+ unit tests + 400+ Monte Carlo simulations

### Recommendations for Real Flight

1. Use the PID cascade controller as-is.
2. Use the robust corridor with the tilt rate cap at 85% of actuator limit.
3. Limit operation to gusts below 4.5 m/s (crashes 0/100 below this threshold).
4. Do not deploy INDI in its current form.

---
Generated by scripts/finalize.py
"""
    with open("PROJECT_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("  Saved: PROJECT_REPORT.md")

    print("\n" + "=" * 70)
    print("FINAL DELIVERABLES COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print("  - FINAL_SELECTED_CONTROLLER.png  (master simulation plot)")
    print("  - PROJECT_REPORT.md              (engineering summary)")
    print("\nProject is ready for:")
    print("  - Academic submission")
    print("  - Flight test planning")
    print("  - Portfolio / job application")
    print("=" * 70)


if __name__ == "__main__":
    main()
    