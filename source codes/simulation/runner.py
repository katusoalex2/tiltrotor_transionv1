"""Master simulation loop."""

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

    def run(self, initial_state: AircraftState, control_func) -> dict[str, np.ndarray]:
        """Run the simulation.
        
        control_func should return: (omega_cmd, tilt_cmd, moment_cmd)
        """
        state = initial_state.copy()
        t = 0.0

        for _ in range(self.n_steps):
            # The control function now provides motor RPM, tilt angle, and moments
            omega, tilt, moment = control_func(t, state)
            
            # Log the state
            self.logger.log(t, state, extra={"omega_cmd": omega, "tilt_cmd": tilt})
            
            # Step the physics using the new 6-DOF signature
                        # The control function returns: omega, tilt, delta_left, delta_right
            omega, tilt, d_left, d_right = control_func(t, state)
            self.logger.log(t, state, extra={"omega_cmd": omega, "tilt_cmd": tilt})
            
            # Pass the new V-tail angles
            state = self.dynamics.step(state, omega, tilt, d_left, d_right, self.dt)

        # Log final state
        omega, tilt, moment = control_func(t, state)
        self.logger.log(t, state, extra={"omega_cmd": omega, "tilt_cmd": tilt})

        return self.logger.to_arrays()