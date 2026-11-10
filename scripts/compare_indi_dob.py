import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.trajectory import ReferenceTrajectory
from src.control.pid import PIDController
from src.control.indi import INDIController
from src.control.indi_dob import INDIControllerWithDOB
from src.wind.gust import DrydenGust

class HybridCascade:
    def __init__(self, mass, k_thrust, mode='pid'):
        self.mass = mass
        self.k_thrust = k_thrust
        self.alt_pid = PIDController(kp=0.1, ki=0.005, kd=0.1, limit=0.25)
        self.speed_pid = PIDController(kp=0.03, ki=0.002, kd=0.01, limit=0.20)
        if mode == 'pid':
            self.inner = PIDController(kp=0.6, ki=0.05, kd=0.4, limit=0.2)
            self.inner_type = 'pid'
        elif mode == 'indi':
            self.inner = INDIController(B_eff=-50.0, pitch_gain=4.0, limit=0.2)
            self.inner_type = 'indi'
        elif mode == 'indi_dob':
            self.inner = INDIControllerWithDOB(B_eff=-50.0, pitch_gain=4.0, limit=0.2)
            self.inner_type = 'indi_dob'

    def compute(self, t, state, ref_alt, ref_speed, ref_tilt):
        current_alt = -state.pos[2]
        current_speed = state.vel[0]
        current_pitch = state.att[1]
        current_pitch_rate = state.omega[1]
        pitch_from_alt = self.alt_pid.compute(ref_alt, current_alt, 0.02)
        pitch_from_speed = -self.speed_pid.compute(ref_speed, current_speed, 0.02)
        desired_pitch = np.clip(pitch_from_alt + pitch_from_speed, -0.35, 0.35)
        if self.inner_type == 'pid':
            vtail_cmd = -self.inner.compute(desired_pitch, current_pitch, 0.01)
        else:
            vtail_cmd = self.inner.compute(desired_pitch, current_pitch, current_pitch_rate, 0.01)
        altitude_error = ref_alt - current_alt
        total_thrust = (self.mass * 9.81) + (altitude_error * 0.5)
        omega_cmd = np.sqrt(max(total_thrust, 0.0) / self.k_thrust)
        omega_cmd = np.clip(omega_cmd, 0.0, 3000.0)
        return {'omega': omega_cmd, 'tilt': ref_tilt, 'delta_left': vtail_cmd, 'delta_right': vtail_cmd}

def run_one(gust_sigma, seed, mode, dt=0.01, duration=20.0):
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)
    controller = HybridCascade(5.0, 1e-5, mode)
    ref = ReferenceTrajectory('configs/robust_corridor.csv', speed_max=12.0)
    wind = DrydenGust(sigma=(gust_sigma, gust_sigma, gust_sigma * 0.6), airspeed=12.0, seed=seed)
    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])
    alt_hist, pitch_hist = [], []
    crashed = False
    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r['altitude'], r['speed'], r['tilt'])
        wind_vec = wind.step(dt)
        state.vel = state.vel + wind_vec
        state = dyn.step(state, cmd['omega'], r['tilt'], cmd['delta_left'], cmd['delta_right'], dt)
        state.vel = state.vel - wind_vec
        alt_hist.append(-state.pos[2])
        pitch_hist.append(np.rad2deg(state.att[1]))
        if -state.pos[2] < 0:
            crashed = True
            break
    alt_hist = np.array(alt_hist) if alt_hist else np.array([100.0])
    return {'max_alt_loss': float(100.0 - alt_hist.min()) if not crashed else 999.0, 'crashed': crashed}

def main():
    n_runs = 100
    print(f'Running 3-way comparison ({n_runs} runs each)...')
    rng = np.random.default_rng(2024)
    results = {'pid': [], 'indi': [], 'indi_dob': []}
    for i in range(n_runs):
        if (i + 1) % 20 == 0: print(f'  Progress: {i + 1}/{n_runs}')
        seed = int(rng.integers(0, 100000))
        sigma = float(rng.uniform(0.5, 5.5))
        for mode in results:
            results[mode].append(run_one(sigma, seed, mode))
    print('\n=== 3-WAY COMPARISON RESULTS ===')
    for mode in ['pid', 'indi', 'indi_dob']:
        crashes = sum(r['crashed'] for r in results[mode])
        valid = [r for r in results[mode] if not r['crashed']]
        mean_loss = np.mean([r['max_alt_loss'] for r in valid]) if valid else 999.0
        print(f'{mode.upper():<10} Crashes: {crashes:>3}/{n_runs} | Mean Alt Loss: {mean_loss:.2f}m')

if __name__ == '__main__': main()
