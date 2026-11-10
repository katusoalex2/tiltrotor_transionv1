import numpy as np
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.kinematics import rotation_body_to_world, kinematic_euler_rates
from src.aero.aerodynamics import AeroModel

GRAVITY = 9.81

class TiltrotorDynamics:
    def __init__(self, mass: float, rotor: RotorModel, inertia: np.ndarray, aero: AeroModel):
        self.mass = float(mass)
        if mass <= 0: raise ValueError('mass must be positive')
        self.rotor = rotor
        self.inertia = np.asarray(inertia, dtype=float)
        self.aero = aero

    def step(self, state: AircraftState, omega_cmd: float, tilt_cmd: float, delta_left: float, delta_right: float, dt: float) -> AircraftState:
        if dt <= 0: raise ValueError('dt must be > 0')
        new_state = state.copy()
        roll, pitch, yaw = state.att
        
        thrust_mag = self.rotor.compute_thrust(omega_cmd)
        force_body = np.array([thrust_mag * np.sin(tilt_cmd), 0.0, -thrust_mag * np.cos(tilt_cmd)])
        
        R_b2w = rotation_body_to_world(roll, pitch, yaw)
        vel_body = R_b2w.T @ state.vel
        
        lift, drag = self.aero.compute_wing_forces(vel_body)
        new_state.lift = lift
        new_state.drag = drag
        
        force_body[0] -= drag
        force_body[2] -= lift
        
        tail_force, tail_moment = self.aero.compute_tail_forces(vel_body, delta_left, delta_right)
        force_body += tail_force
        
        force_world = R_b2w @ force_body
        gravity_world = np.array([0.0, 0.0, self.mass * GRAVITY])
        accel_world = (force_world + gravity_world) / self.mass
        
        p, q, r = state.omega
        Ix, Iy, Iz = self.inertia
        omega_cross_Iomega = np.array([(Iz - Iy)*q*r, (Ix - Iz)*r*p, (Iy - Ix)*p*q])
        angular_accel = (tail_moment - omega_cross_Iomega) / self.inertia
        
        new_state.vel = state.vel + accel_world * dt
        new_state.pos = state.pos + new_state.vel * dt
        new_state.omega = state.omega + angular_accel * dt
        new_state.att = state.att + kinematic_euler_rates(roll, pitch, new_state.omega) * dt
        new_state.tilt = tilt_cmd
        return new_state
