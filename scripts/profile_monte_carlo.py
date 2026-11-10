"""Performance profile for Monte Carlo simulations."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
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


def single_run(dt, duration, seed, with_profiling=False):
    """One full Monte Carlo run. Returns per-phase timings if with_profiling."""
    timers = {"setup": 0, "sim_loop": 0, "total": 0}
    t_start = time.time()

    # --- Setup ---
    t0 = time.time()
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)
    controller = PIDCascade(5.0, 1e-5)
    ref = ReferenceTrajectory("configs/robust_corridor.csv", speed_max=12.0)
    wind = DrydenGust(sigma=(2.5, 2.5, 1.5), airspeed=12.0, seed=seed)
    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])
    timers["setup"] = time.time() - t0

    # --- Sim loop ---
    t0 = time.time()
    n_steps = int(duration / dt)
    alt_hist = np.zeros(n_steps)
    for i in range(n_steps):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r["altitude"], r["speed"], r["tilt"])
        wind_vec = wind.step(dt)
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd["omega"], r["tilt"], cmd["delta_left"], cmd["delta_right"], dt)
        state.vel = state.vel - wind_vec
        alt_hist[i] = -state.pos[2]
    timers["sim_loop"] = time.time() - t0
    timers["total"] = time.time() - t_start

    if with_profiling:
        return timers
    return 100.0 - alt_hist.min()


def main():
    print("=" * 70)
    print("PERFORMANCE PROFILE ? MONTE CARLO")
    print("=" * 70)

    # --- Benchmark a single run ---
    print("\n[1/3] Profiling a single run (dt=0.01, duration=20s)...")
    timers = single_run(dt=0.01, duration=20.0, seed=42, with_profiling=True)
    print(f"  Setup:     {timers['setup']*1000:>8.1f} ms")
    print(f"  Sim loop:  {timers['sim_loop']*1000:>8.1f} ms  (2000 steps)")
    print(f"  Per step:  {timers['sim_loop']/2000*1e6:>8.1f} ?s")
    print(f"  Total:     {timers['total']*1000:>8.1f} ms")

    # --- Benchmark 10 runs (to reduce warm-up bias) ---
    print("\n[2/3] Benchmarking 10 runs...")
    n_bench = 10
    t0 = time.time()
    for i in range(n_bench):
        single_run(dt=0.01, duration=20.0, seed=i)
    bench_time = time.time() - t0
    per_run = bench_time / n_bench
    print(f"  Total 10 runs: {bench_time:.2f}s")
    print(f"  Per run:       {per_run*1000:.1f} ms")
    print(f"  Extrapolated to 200 runs: {per_run * 200:.1f}s ({per_run * 200 / 60:.2f} min)")

    # --- Verdict ---
    print("\n[3/3] Verdict:")
    target = 30 * 60  # 30 minutes = 1800 seconds
    projected = per_run * 200
    if projected < target:
        print(f"  [OK] 200-run Monte Carlo projects to {projected:.1f}s < {target}s target")
        print(f"       Margin: {target - projected:.1f}s ({100*(1-projected/target):.0f}% under budget)")
    else:
        print(f"  [WARN] 200-run Monte Carlo projects to {projected:.1f}s > {target}s target")
        print(f"         Need to reduce by {projected - target:.1f}s.")
        print(f"         Options:")
        print(f"           - Reduce dt from 0.01 to 0.02 (halves time, minor accuracy loss)")
        print(f"           - Reduce duration from 20s to 15s (25% faster)")
        print(f"           - Reduce n_runs from 200 to 100 (50% faster)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
