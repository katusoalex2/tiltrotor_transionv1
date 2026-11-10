import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.aero.vtail import VtailModel
from src.aero.aerodynamics import AeroModel
from src.control.cascade import CascadeController
from src.control.reference import TransitionReference

def run_transition(transition_time: float, duration: float = 30.0, dt: float = 0.01):
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    vtail = VtailModel(area=0.1, arm_length=0.5)
    # BIGGER WINGS: 1.5 m^2 instead of 0.5
    aero = AeroModel(wing_area=1.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=np.array([0.1, 0.1, 0.2]), aero=aero)
    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)
    controller = CascadeController(mass=5.0, k_thrust=1e-5, hover_omega=omega_hover)
    # SLOWER TARGET SPEED: 12.0 instead of 20.0
    ref = TransitionReference(start_altitude=100.0, target_speed=12.0, transition_time=transition_time)
    state = AircraftState()
    state.pos = np.array([0.0, 0.0, -100.0])
    state.vel = np.array([0.0, 0.0, 0.0])
    alt_log, pitch_log = [], []
    for i in range(int(duration / dt)):
        t = i * dt
        r = ref.at(t)
        cmd = controller.compute(t, state, r['altitude'], r['speed'], r['tilt'])
        state = dyn.step(state, cmd['omega'], cmd['tilt'], cmd['delta_left'], cmd['delta_right'], dt)
        alt_log.append(-state.pos[2])
        pitch_log.append(np.rad2deg(state.att[1]))
    alt_log = np.array(alt_log)
    pitch_log = np.array(pitch_log)
    return {'max_alt_loss': 100.0 - alt_log.min(), 'final_alt': alt_log[-1], 'max_pitch': np.max(np.abs(pitch_log))}

def main():
    transition_times = [3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    print(f"{'T (s)':>6} | {'Max Alt Loss (m)':>18} | {'Final Alt (m)':>14} | {'Max |Pitch| (deg)':>20}")
    print('-' * 70)
    for T in transition_times:
        r = run_transition(T, duration=max(T + 10.0, 20.0))
        print(f"{T:>6.1f} | {r['max_alt_loss']:>18.2f} | {r['final_alt']:>14.2f} | {r['max_pitch']:>20.2f}")

if __name__ == '__main__': main()
