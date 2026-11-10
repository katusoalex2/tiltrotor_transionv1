"""Entry point for the tilt-rotor transition simulation.

Phase P0 demo: drop a body from 100 m with zero thrust, then hold hover
with thrust = m*g. Prints a small table so you can eyeball the physics.
"""

import numpy as np

from src.dynamics.state import AircraftState
from src.dynamics.tiltrotor import TiltrotorDynamics


def run_demo() -> None:
    dyn = TiltrotorDynamics(mass=5.0)
    dt = 0.1
    n_steps = 30

    # Scenario 1: free fall from 100 m (NED: +Z is down)
    print("\n=== Free fall from 100 m (thrust = 0) ===")
    print(f"{'t (s)':>8} {'z (m)':>12} {'vz (m/s)':>12} {'az (m/s^2)':>12}")
    s = AircraftState()
    s.pos = np.array([0.0, 0.0, 100.0])
    t = 0.0
    for _ in range(n_steps):
        print(f"{t:8.2f} {s.pos[2]:12.3f} {s.vel[2]:12.3f} {9.81:12.3f}")
        s = dyn.step(s, thrust=0.0, dt=dt)
        t += dt
    print(f"{t:8.2f} {s.pos[2]:12.3f} {s.vel[2]:12.3f} {9.81:12.3f}")

    # Scenario 2: hover at 50 m
    print("\n=== Hover at 50 m (thrust = m*g) ===")
    print(f"{'t (s)':>8} {'z (m)':>12} {'vz (m/s)':>12}")
    s = AircraftState()
    s.pos = np.array([0.0, 0.0, 50.0])
    s.vel = np.zeros(3)
    t = 0.0
    for _ in range(n_steps):
        print(f"{t:8.2f} {s.pos[2]:12.3f} {s.vel[2]:12.3f}")
        s = dyn.step(s, thrust=dyn.hover_thrust(), dt=dt)
        t += dt

    print("\nIf free-fall velocity grows by ~9.81 m/s per second and hover")
    print("stays flat, the physics is behaving correctly.")


if __name__ == "__main__":
    run_demo()
    