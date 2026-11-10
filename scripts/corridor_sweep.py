import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.pid import PIDController

def simulate_transition(tilt_rate_deg_per_s, duration=15.0, dt=0.01):
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1,0.1,0.2]), aero=aero)
    state = AircraftState()
    state.pos = np.array([0.0, 0.0, 100.0])
    state.vel = np.array([5.0, 0.0, 0.0])
    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)
    alt_pid = PIDController(kp=0.5, ki=0.05, kd=0.1, limit=200.0)
    pitch_pid = PIDController(kp=2.0, ki=0.5, kd=0.1, limit=0.5)
    tilt_rate = np.deg2rad(tilt_rate_deg_per_s)
    log = {'t': [], 'alt': [], 'vx': [], 'lift': [], 'pitch': []}
    t = 0.0
    for _ in range(int(duration/dt)):
        tilt_cmd = min(t * tilt_rate, np.pi/2)
        alt = 100.0 - state.pos[2]
        omega_cmd = np.clip(omega_hover + alt_pid.compute(100.0, alt, dt), 0.0, 3000.0)
        vtail_def = pitch_pid.compute(0.0, state.att[1], dt)
        state = dyn.step(state, omega_cmd, tilt_cmd, vtail_def, vtail_def, dt)
        t += dt
        log['t'].append(t); log['alt'].append(alt); log['vx'].append(state.vel[0])
        log['lift'].append(state.lift); log['pitch'].append(np.rad2deg(state.att[1]))
    return {k: np.array(v) for k, v in log.items()}

def main():
    tilt_rates = [2, 4, 6, 8, 10, 12, 15, 20]
    results = {}
    print('Running sweep...')
    for rate in tilt_rates:
        r = simulate_transition(rate)
        results[rate] = r
        print(f'Rate: {rate:2d} deg/s | Max Alt Loss: {r["alt"].min():8.2f} m | Final Vx: {r["vx"][-1]:6.2f} m/s')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for i, (rate, r) in enumerate(results.items()):
        c = plt.cm.viridis(i / len(results))
        ax1.plot(r['t'], r['alt'], color=c, label=f'{rate} deg/s')
        ax2.plot(r['t'], r['lift'], color=c, label=f'{rate} deg/s')
    ax1.axhline(0, color='red', linestyle='--'); ax1.set_title('Altitude'); ax1.grid(True); ax1.legend()
    ax2.set_title('Lift'); ax2.grid(True); ax2.legend()
    plt.show()

if __name__ == '__main__': main()
