import numpy as np
from src.dynamics.state import AircraftState
from src.dynamics.tiltrotor import TiltrotorDynamics
from src.simulation.logger import StateLogger

class Simulation:
    def __init__(self, dt: float, duration: float, dynamics: TiltrotorDynamics):
        self.dt = dt
        self.duration = duration
        self.dynamics = dynamics
        self.logger = StateLogger()
        self.n_steps = int(round(duration / dt))

    def run(self, initial_state: AircraftState, control_func):
        state = initial_state.copy()
        t = 0.0
        for _ in range(self.n_steps):
            omega, tilt, d_left, d_right = control_func(t, state)
            self.logger.log(t, state, extra={'omega_cmd': omega, 'tilt_cmd': tilt})
            state = self.dynamics.step(state, omega, tilt, d_left, d_right, self.dt)
            t += self.dt
        return self.logger.to_arrays()
