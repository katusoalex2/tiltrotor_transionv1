from __future__ import annotations
import numpy as np
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.frames import rotation_body_to_earth, body_rates_to_euler_rates
from src.aero.aerodynamics import AeroModel

GRAVITY = 9.81

class TiltrotorDynamics:
    def __init__(self, mass: float, rotor: RotorModel, inertia: np.ndarray, aero: AeroModel):
        self.mass = float(mass)
        if mass <= 0: raise ValueError('mass must be positive')
        self.rotor = rotor
        self.inertia = np.asarray(inertia, dtype=float)
        self.aero = aero

    def step(self, state: AircraftState, wind_earth: np.ndarray, omega_fl: float, omega_fr: float, omega_rl: float, omega_rr: float, tilt_cmd: float, delta_left: float, delta_right: float, dt: float) -> AircraftState:
        if dt <= 0: raise ValueError('dt must be > 0')

        new_state = state.copy()
        roll, pitch, yaw = state.att

        t_fl = self.rotor.compute_thrust(omega_fl)
        t_fr = self.rotor.compute_thrust(omega_fr)
        t_rl = self.rotor.compute_thrust(omega_rl)
        t_rr = self.rotor.compute_thrust(omega_rr)

        force_x = (t_fl + t_fr) * np.sin(tilt_cmd)
        force_z = -((t_fl + t_fr) * np.cos(tilt_cmd) + t_rl + t_rr)
        force_body = np.array([force_x, 0.0, force_z])

        arm = 0.4
        moment_roll = ((t_rl + t_fl) - (t_rr + t_fr)) * arm
        moment_yaw = ((t_fl + t_rr) - (t_fr + t_rl)) * 0.1

        lift_l = 0.5 * 1.225 * (state.vel[0]**2) * 0.1 * delta_left
        lift_r = 0.5 * 1.225 * (state.vel[0]**2) * 0.1 * delta_right
        moment_pitch_control = (lift_l + lift_r) * 0.5
        
        alpha = state.att[1]
        q_bar = 0.5 * 1.225 * (state.vel[0]**2)
        moment_pitch_stability = q_bar * self.aero.wing_area * 0.4 * (-1.0 * alpha) 
        
        tail_moment = np.array([moment_roll, moment_pitch_control + moment_pitch_stability, moment_yaw])

        total_thrust = t_fl + t_fr + t_rl + t_rr
        lift, drag, V_T, alpha_calc = self.aero.compute_forces(state.vel, wind_earth, state.att, total_thrust, tilt_cmd)
        
        new_state.lift = lift
        new_state.drag = drag
        new_state.V_T = V_T
        new_state.alpha = alpha_calc
        
        force_body[0] -= drag
        force_body[2] -= lift

        R_b2w = rotation_body_to_earth(roll, pitch, yaw)
        force_world = R_b2w @ force_body
        gravity_world = np.array([0.0, 0.0, self.mass * GRAVITY])
        accel_world = (force_world + gravity_world) / self.mass
        new_state.vel = state.vel + accel_world * dt
        new_state.pos = state.pos + new_state.vel * dt

        p, q, r = state.omega
        Ix, Iy, Iz = self.inertia
        omega_cross_Iomega = np.array([(Iz - Iy)*q*r, (Ix - Iz)*r*p, (Iy - Ix)*p*q])
        angular_accel = (tail_moment - omega_cross_Iomega) / self.inertia
        new_state.omega = state.omega + angular_accel * dt
        new_state.att = state.att + body_rates_to_euler_rates(roll, pitch, new_state.omega) * dt
        new_state.tilt = tilt_cmd

        return new_state
