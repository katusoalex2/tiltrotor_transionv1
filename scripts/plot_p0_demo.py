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
from src.simulation.runner import Simulation

def main():
    dt = 0.01
    duration = 10.0
    rotor = RotorModel(k_thrust=1e-5, max_omega=3000.0)
    inertia = np.array([0.1, 0.1, 0.2])
    vtail = VtailModel(area=0.1, arm_length=0.5)
    aero = AeroModel(wing_area=0.5, vtail=vtail)
    dyn = TiltrotorDynamics(mass=5.0, rotor=rotor, inertia=inertia, aero=aero)

    init_state = AircraftState()
    init_state.pos = np.array([0.0, 0.0, 50.0])
    init_state.vel = np.array([15.0, 0.0, 0.0])
    omega_hover = np.sqrt(dyn.mass * 9.81 / rotor.k_thrust)

    def cmd_func(t, state):
        tilt_angle = min(t * (np.pi/6) / 2.0, np.pi/6)
        return omega_hover, tilt_angle, -0.08, -0.08

    sim = Simulation(dt, duration, dyn)
    data = sim.run(init_state, cmd_func)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(data['t'], np.rad2deg(data['att'][:, 1]), 'r-', label='Pitch (deg)')
    ax1.set_title('Pitch Angle')
    ax1.grid(True)
    ax2.plot(data['t'], 50.0 - data['pos'][:, 2], 'b-', label='Altitude')
    ax2.set_title('Altitude')
    ax2.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
