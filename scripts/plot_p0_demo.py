import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from src.dynamics.state import AircraftState
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.configuration.aircraft_models import build_aircraft_dynamics, step_collective

def main():
    dt = 0.01
    duration = 10.0
    aircraft = load_aircraft_parameters()
    if len(aircraft.motor_positions) != 4:
        raise ValueError("this simulation requires exactly four rotors")
    dyn = build_aircraft_dynamics(aircraft)

    init_state = AircraftState()
    init_state.pos = np.array([0.0, 0.0, -50.0])
    init_state.vel = np.array([15.0, 0.0, 0.0])
    omega_hover = np.sqrt(dyn.mass * aircraft.gravity / (4 * aircraft.k_thrust))
    state = init_state
    data = {"t": [], "att": [], "pos": []}
    for step in range(int(duration / dt)):
        t = step * dt
        data["t"].append(t)
        data["att"].append(state.att.copy())
        data["pos"].append(state.pos.copy())
        tilt_angle = min(t * (np.pi/6) / 2.0, np.pi/6)
        state = step_collective(dyn, state, {
            "omega": omega_hover, "tilt": tilt_angle,
            "delta_left": -0.08, "delta_right": -0.08,
        }, dt)
    data = {key: np.asarray(value) for key, value in data.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(data['t'], np.rad2deg(data['att'][:, 1]), 'r-', label='Pitch (deg)')
    ax1.set_title('Pitch Angle')
    ax1.grid(True)
    ax2.plot(data['t'], -data['pos'][:, 2], 'b-', label='Altitude')
    ax2.set_title('Altitude')
    ax2.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
