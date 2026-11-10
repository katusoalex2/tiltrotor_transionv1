"""Dynamics for the tilt-rotor UAV using 6-DOF physics."""

from __future__ import annotations
import numpy as np
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.kinematics import rotation_body_to_world, kinematic_euler_rates
from src.aero.aerodynamics import AeroModel

GRAVITY = 9.81

class TiltrotorDynamics:
    def __init__(self, mass: float, rotor: RotorModel, inertia: np.ndarray, aero: AeroModel) -> None:
        self.mass = float(mass)
        self.rotor = rotor
        self.inertia = np.asarray(inertia, dtype=float)
        self.aero = aero

    def step(
        self, 
        state: AircraftState, 
        omega_cmd: float, 
        tilt_cmd: float, 
        delta_left: float,
        delta_right: float,
        dt: float
    ) -> AircraftState:
        
        if dt <= 0: raise ValueError("dt must be > 0")

        new_state = state.copy()
        roll, pitch, yaw = state.att

        # 1. Rotor Thrust
        thrust_mag = self.rotor.compute_thrust(omega_cmd)
        force_body = np.array([thrust_mag * np.sin(tilt_cmd), 0.0, -thrust_mag * np.cos(tilt_cmd)])

        # 2. Get Body Velocity for Aero
        R_b2w = rotation_body_to_world(roll, pitch, yaw)
        vel_body = R_b2w.T @ state.vel

        # 3. Wing Lift & Drag
        lift, drag = self.aero.compute_wing_forces(vel_body)
        new_state.lift = lift   # Store for logging
        new_state.drag = drag
        force_body[0] -= drag
        force_body[2] -= lift

        # 4. V-Tail Forces & Moments
        tail_force, tail_moment = self.aero.compute_tail_forces(vel_body, delta_left, delta_right)
        force_body += tail_force

        # 5. Translate to World Frame & Add Gravity
        force_world = R_b2w @ force_body
        gravity_world = np.array([0.0, 0.0, self.mass * GRAVITY])
        accel_world = (force_world + gravity_world) / self.mass

        # 6. Rotational Dynamics (PDF Page 24)
        p, q, r = state.omega
        Ix, Iy, Iz = self.inertia
        omega_cross_Iomega = np.array([(Iz - Iy)*q*r, (Ix - Iz)*r*p, (Iy - Ix)*p*q])
        
        # The only moments we have right now come from the V-Tail!
        total_moment = tail_moment 
        angular_accel = (total_moment - omega_cross_Iomega) / self.inertia

        # 7. Integration
        new_state.vel = state.vel + accel_world * dt
        new_state.pos = state.pos + new_state.vel * dt
        new_state.omega = state.omega + angular_accel * dt
        new_state.att = state.att + kinematic_euler_rates(roll, pitch, new_state.omega) * dt
        new_state.tilt = tilt_cmd

        return new_state
    