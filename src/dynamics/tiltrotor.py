from __future__ import annotations
import numpy as np
from src.dynamics.state import AircraftState
from src.dynamics.rotor import RotorModel
from src.dynamics.frames import rotation_body_to_earth, body_rates_to_euler_rates
from src.aero.aerodynamics import AeroModel

GRAVITY = 9.81

class TiltrotorDynamics:
    def __init__(self, mass: float, rotor: RotorModel, mass_props, aero: AeroModel):
        self.mass = float(mass)
        if mass <= 0: raise ValueError('mass must be positive')
        self.rotor = rotor
        self.mass_props = mass_props
        self.aero = aero

    def step(self, state: AircraftState, wind_earth: np.ndarray, omega_fl: float, omega_fr: float, omega_rl: float, omega_rr: float, tilt_cmd: float, delta_left: float, delta_right: float, dt: float) -> AircraftState:
        if dt <= 0: raise ValueError('dt must be > 0')

        new_state = state.copy()
        roll, pitch, yaw = state.att

        # 1. Motor Thrusts
                # Compute height above ground (NED: pos[2] is altitude downward, so negative)
        # The ground is at altitude 0
        height_above_ground = max(-state.pos[2], 0.01)  # Prevent division by zero
        
        t_fl = self.rotor.compute_thrust(omega_fl, height_above_ground)
        t_fr = self.rotor.compute_thrust(omega_fr, height_above_ground)
        t_rl = self.rotor.compute_thrust(omega_rl, height_above_ground)
        t_rr = self.rotor.compute_thrust(omega_rr, height_above_ground)
        # Calculate angular momentum for all 4 props
        # Note: Front right and rear left spin CW (-), Front left and rear right spin CCW (+)
        h_fl = self.rotor.get_angular_momentum(omega_fl)   # CCW (+)
        h_fr = self.rotor.get_angular_momentum(omega_fr)   # CW (-)
        h_rl = self.rotor.get_angular_momentum(omega_rl)   # CW (-)
        h_rr = self.rotor.get_angular_momentum(omega_rr)   # CCW (+)
        
        # The total angular momentum vector of the props in the body frame
        # The front props tilt forward, so their spin axis is tilted by tilt_cmd
        # The rear props are fixed, so they point straight up (Z-body)
        h_total = np.array([
            h_fl * np.sin(tilt_cmd) + h_fr * np.sin(tilt_cmd), # X (Forward) component
            h_fl - h_fr + h_rl - h_rr,                          # Y (Right) component (spin axis)
            -h_fl * np.cos(tilt_cmd) - h_fr * np.cos(tilt_cmd) - h_rl - h_rr # Z (Down) component
        ])
        
        # Gyroscopic moment: M_gyro = omega_body x h_total
        p, q, r = state.omega
        gyro_moment = np.cross(np.array([p, q, r]), h_total)
        
        # 2. Forces (Body Frame)
        force_x = (t_fl + t_fr) * np.sin(tilt_cmd)
        force_z = -((t_fl + t_fr) * np.cos(tilt_cmd) + t_rl + t_rr)
        force_body = np.array([force_x, 0.0, force_z])

        # 3. Moments
        arm = 0.4
        moment_roll = ((t_rl + t_fl) - (t_rr + t_fr)) * arm
        moment_yaw = ((t_fl + t_rr) - (t_fr + t_rl)) * 0.1

        total_thrust = t_fl + t_fr + t_rl + t_rr
        lift_l = 0.5 * 1.225 * (state.vel[0]**2) * 0.1 * delta_left
        lift_r = 0.5 * 1.225 * (state.vel[0]**2) * 0.1 * delta_right
        moment_pitch_control = (lift_l + lift_r) * 0.5
        
        alpha_wing = state.att[1]
        q_bar = 0.5 * 1.225 * (state.vel[0]**2)
        moment_pitch_stability = q_bar * self.aero.wing_area * 0.4 * (-1.0 * alpha_wing) 
        
        total_moment = np.array([moment_roll, moment_pitch_control + moment_pitch_stability, moment_yaw]) + gyro_moment
        
        # 4. Aerodynamic Forces
        total_thrust = t_fl + t_fr + t_rl + t_rr
        lift, drag, V_T, alpha_calc = self.aero.compute_forces(state.vel, wind_earth, state.att, total_thrust, tilt_cmd)
        
        new_state.lift = lift
        new_state.drag = drag
        new_state.V_T = V_T
        new_state.alpha = alpha_calc
        
        force_body[0] -= drag
        force_body[2] -= lift

        # 5. Linear Dynamics
        R_b2w = rotation_body_to_earth(roll, pitch, yaw)
        force_world = R_b2w @ force_body
        gravity_world = np.array([0.0, 0.0, self.mass * GRAVITY])
        accel_world = (force_world + gravity_world) / self.mass
        new_state.vel = state.vel + accel_world * dt
        new_state.pos = state.pos + new_state.vel * dt

        # 6. Rotational Dynamics (FULL 3x3 Euler Equations)
        omega = state.omega
        
        # Get the dynamic inertia matrix at this exact tilt angle
        I = self.mass_props.calculate_inertia(np.rad2deg(tilt_cmd))
        
        # Gyroscopic term: omega x (I * omega)
        omega_cross_Iomega = np.cross(omega, I @ omega)
        
        # Angular acceleration: I_inv * (tau - omega x I*omega)
        I_inv = np.linalg.inv(I)
        angular_accel = I_inv @ (total_moment - omega_cross_Iomega)
        
        new_state.omega = state.omega + angular_accel * dt
        new_state.att = state.att + body_rates_to_euler_rates(roll, pitch, new_state.omega) * dt
        new_state.tilt = tilt_cmd

        return new_state
