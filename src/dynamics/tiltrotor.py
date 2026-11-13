from __future__ import annotations

import numpy as np

from src.aero.aerodynamics import AeroModel
from src.dynamics.frames import body_rates_to_euler_rates, rotation_body_to_earth
from src.dynamics.rotor import RotorModel
from src.dynamics.state import AircraftState

GRAVITY = 9.81


class TiltrotorDynamics:
    """6-DOF rigid-body dynamics using NED states and FRD body forces."""

    def __init__(self, mass: float, rotor: RotorModel, mass_props=None, aero: AeroModel | None = None, inertia=None):
        self.mass = float(mass)
        if self.mass <= 0.0:
            raise ValueError('mass must be positive')

        self.rotor = rotor
        self.mass_props = mass_props
        self.inertia = inertia
        self.aero = aero
        self.last_diagnostics: dict[str, np.ndarray] = {}

    def _inertia_matrix(self, tilt_angle: float) -> np.ndarray:
        if self.mass_props is not None:
            return np.asarray(self.mass_props.calculate_inertia(np.rad2deg(tilt_angle)), dtype=float)
        if self.inertia is not None:
            inertia = np.asarray(self.inertia, dtype=float)
            if inertia.shape == (3,):
                return np.diag(inertia)
            return inertia
        return np.eye(3, dtype=float)

    def _parse_inputs(self, args, kwargs):
        if 'omega_cmd' in kwargs or 'tilt_cmd' in kwargs:
            omega_cmd = float(kwargs.get('omega_cmd', 0.0))
            tilt_cmd = float(kwargs.get('tilt_cmd', 0.0))
            delta_left = float(kwargs.get('delta_left', 0.0))
            delta_right = float(kwargs.get('delta_right', 0.0))
            dt = float(kwargs.get('dt', 0.01))
            wind_earth = np.asarray(kwargs.get('wind_earth', np.zeros(3)), dtype=float)
            return wind_earth, np.array([omega_cmd], dtype=float), tilt_cmd, delta_left, delta_right, dt

        if len(args) == 5:
            omega_cmd, tilt_cmd, delta_left, delta_right, dt = args
            return np.zeros(3), np.array([float(omega_cmd)], dtype=float), float(tilt_cmd), float(delta_left), float(delta_right), float(dt)

        if len(args) == 9:
            wind_earth, omega_fl, omega_fr, omega_rl, omega_rr, tilt_cmd, delta_left, delta_right, dt = args
            return np.asarray(wind_earth, dtype=float), np.array([float(omega_fl), float(omega_fr), float(omega_rl), float(omega_rr)], dtype=float), float(tilt_cmd), float(delta_left), float(delta_right), float(dt)

        raise TypeError('Unsupported step signature; expected either (state, omega_cmd, tilt_cmd, delta_left, delta_right, dt) or (state, wind_earth, omega_fl, omega_fr, omega_rl, omega_rr, tilt_cmd, delta_left, delta_right, dt).')

    def _rotor_forces_and_moments(
        self,
        rotor_cmds: np.ndarray,
        tilt_cmd: float,
        height_above_ground: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if rotor_cmds.size == 1:
            thrust = self.rotor.compute_thrust(float(rotor_cmds[0]), height_above_ground)
            force_body = thrust * np.array([np.sin(tilt_cmd), 0.0, -np.cos(tilt_cmd)])
            return force_body, np.zeros(3, dtype=float), float(thrust)

        default_positions = [
            [0.3, -0.4, 0.0, True],
            [0.3, 0.4, 0.0, True],
            [-0.3, -0.4, 0.0, False],
            [-0.3, 0.4, 0.0, False],
        ]
        positions = getattr(self.mass_props, "motor_positions", default_positions)
        if len(positions) != rotor_cmds.size:
            raise ValueError("motor_positions must match the number of rotor commands")
        position_vectors = [np.asarray(position, dtype=float) for position in positions]
        if any(position.size not in (3, 4) for position in position_vectors):
            raise ValueError("each motor position must contain x, y, z, and optionally a tilt flag")
        position_vectors.sort(key=lambda position: (-position[0], position[1]))

        force_body = np.zeros(3, dtype=float)
        moment_body = np.zeros(3, dtype=float)
        total_thrust = 0.0
        pivot_x = 0.1

        for omega_cmd, position_values in zip(rotor_cmds, position_vectors):
            x, y, z = position_values[:3]
            tilts = bool(position_values[3]) if position_values.size == 4 else False
            rotor_tilt = tilt_cmd if tilts else 0.0

            if tilts:
                relative_x = x - pivot_x
                x = pivot_x + relative_x * np.cos(rotor_tilt) + z * np.sin(rotor_tilt)
                z = -relative_x * np.sin(rotor_tilt) + z * np.cos(rotor_tilt)

            thrust = self.rotor.compute_thrust(float(omega_cmd), height_above_ground)
            rotor_force = thrust * np.array([np.sin(rotor_tilt), 0.0, -np.cos(rotor_tilt)])
            force_body += rotor_force
            moment_body += np.cross(np.array([x, y, z]), rotor_force)
            total_thrust += thrust

        return force_body, moment_body, total_thrust

    def step(self, state: AircraftState, *args, **kwargs):
        """Advance the 6-DOF state using NED velocities and FRD forces/moments.

        Integrates ``pos_dot = vel``, ``vel_dot = R_b2n @ force_body / mass + g_ned``,
        3-2-1 Euler kinematics, and ``I @ omega_dot = moment - omega x (I @ omega)
        - I_dot @ omega``.
        """
        if args and isinstance(args[0], dict):
            cfg = args[0]
            kwargs = {**cfg, **kwargs}
            args = ()

        wind_earth, rotor_cmds, tilt_cmd, delta_left, delta_right, dt = self._parse_inputs(args, kwargs)
        if dt <= 0:
            raise ValueError('dt must be > 0')

        new_state = state.copy()
        roll, pitch, yaw = state.att

        height_above_ground = max(-state.pos[2], 0.01)
        force_body, rotor_moment, total_thrust = self._rotor_forces_and_moments(
            rotor_cmds,
            tilt_cmd,
            height_above_ground,
        )

        if self.aero is not None:
            aero_force, aero_moment, V_T, alpha, _ = self.aero.compute_forces_and_moments(state.vel, wind_earth, state.att, total_thrust, tilt_cmd)
            tail_force, tail_moment = self.aero.compute_tail_forces(
                state.vel,
                wind_earth,
                state.att,
                delta_left,
                delta_right,
            )
            new_state.lift = float(-aero_force[2])
            new_state.drag = float(-aero_force[0])
            new_state.V_T = float(V_T)
            new_state.alpha = float(alpha)
            force_body += aero_force + tail_force
        else:
            new_state.lift = 0.0
            new_state.drag = 0.0
            new_state.V_T = 0.0
            new_state.alpha = 0.0
            aero_moment = np.zeros(3, dtype=float)
            tail_moment = np.zeros(3, dtype=float)

        R_b2w = rotation_body_to_earth(roll, pitch, yaw)
        force_world = R_b2w @ force_body
        gravity_world = np.array([0.0, 0.0, self.mass * GRAVITY], dtype=float)
        accel_world = (force_world + gravity_world) / self.mass

        new_state.vel = state.vel + accel_world * dt
        new_state.pos = state.pos + 0.5 * (state.vel + new_state.vel) * dt

        previous_inertia = self._inertia_matrix(state.tilt)
        inertia = self._inertia_matrix(tilt_cmd)
        inertia_rate = (inertia - previous_inertia) / dt
        omega = np.asarray(state.omega, dtype=float)
        omega_cross_Iomega = np.cross(omega, inertia @ omega)
        total_moment = rotor_moment + aero_moment + tail_moment

        angular_accel = np.linalg.solve(
            inertia,
            total_moment - omega_cross_Iomega - inertia_rate @ omega,
        )
        new_state.omega = state.omega + angular_accel * dt
        euler_rate = body_rates_to_euler_rates(roll, pitch, omega)
        attitude_midpoint = state.att + 0.5 * euler_rate * dt
        omega_midpoint = 0.5 * (omega + new_state.omega)
        euler_rate_midpoint = body_rates_to_euler_rates(
            float(attitude_midpoint[0]),
            float(attitude_midpoint[1]),
            omega_midpoint,
        )
        new_state.att = state.att + euler_rate_midpoint * dt
        new_state.tilt = float(tilt_cmd)
        self.last_diagnostics = {
            "force_body": force_body.copy(),
            "force_ned": force_world.copy(),
            "acceleration_ned": accel_world.copy(),
            "moment_body": total_moment.copy(),
            "angular_acceleration_body": angular_accel.copy(),
            "inertia_body": inertia.copy(),
        }

        return new_state
