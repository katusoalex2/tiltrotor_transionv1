"""Bounded control allocation for four rotors and two V-tail servos."""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


DEFAULT_MOTOR_POSITIONS = (
    (0.3, -0.4, 0.0, True),
    (0.3, 0.4, 0.0, True),
    (-0.3, -0.4, 0.0, False),
    (-0.3, 0.4, 0.0, False),
)


class ControlAllocator:
    """Map total thrust and body moments to bounded rotor and tail commands."""

    actuator_labels = (
        "Rotor FL",
        "Rotor FR",
        "Rotor RL",
        "Rotor RR",
        "V-tail left",
        "V-tail right",
    )
    output_labels = (
        "Total thrust (N)",
        "Roll moment (N·m)",
        "Pitch moment (N·m)",
        "Yaw moment (N·m)",
    )

    def __init__(
        self,
        k_thrust: float = 1e-5,
        max_omega: float = 3000.0,
        arm_length: float = 0.4,
        motor_positions=None,
    ):
        self.k_thrust = float(k_thrust)
        self.max_omega = float(max_omega)
        self.arm_length = float(arm_length)
        if self.k_thrust <= 0.0 or self.max_omega <= 0.0 or self.arm_length <= 0.0:
            raise ValueError("thrust coefficient, maximum speed, and arm length must be positive")

        positions = DEFAULT_MOTOR_POSITIONS if motor_positions is None else motor_positions
        position_vectors = [np.asarray(position, dtype=float) for position in positions]
        if len(position_vectors) != 4:
            raise ValueError("control allocation requires exactly four motor positions")
        if any(position.size not in (3, 4) for position in position_vectors):
            raise ValueError("each motor position must contain x, y, z, and optionally a tilt flag")
        if any(not np.all(np.isfinite(position[:3])) for position in position_vectors):
            raise ValueError("motor positions must contain finite coordinates")
        self.motor_positions = tuple(
            tuple(position.tolist())
            for position in sorted(position_vectors, key=lambda position: (-position[0], position[1]))
        )

    def allocation_matrix(
        self,
        tilt_cmd: float = 0.0,
        tail_effectiveness: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the local matrix from rotor thrust/tail deflection to [T, L, M, N]."""
        if not np.isfinite(tilt_cmd) or not 0.0 <= tilt_cmd <= np.pi / 2.0:
            raise ValueError("tilt command must be between 0 and 90 degrees")

        matrix = np.zeros((4, 6), dtype=float)
        pivot_x = 0.1
        for index, position_values in enumerate(self.motor_positions):
            x, y, z = position_values[:3]
            tilts = bool(position_values[3]) if len(position_values) == 4 else False
            rotor_tilt = tilt_cmd if tilts else 0.0
            if tilts:
                relative_x = x - pivot_x
                x = pivot_x + relative_x * np.cos(rotor_tilt) + z * np.sin(rotor_tilt)
                z = -relative_x * np.sin(rotor_tilt) + z * np.cos(rotor_tilt)

            force_per_newton = np.array([np.sin(rotor_tilt), 0.0, -np.cos(rotor_tilt)])
            matrix[0, index] = 1.0
            matrix[1:, index] = np.cross(np.array([x, y, z]), force_per_newton)

        if tail_effectiveness is not None:
            tail_effectiveness = np.asarray(tail_effectiveness, dtype=float)
            if tail_effectiveness.shape != (3, 2) or not np.all(np.isfinite(tail_effectiveness)):
                raise ValueError("tail effectiveness must be a finite 3-by-2 matrix")
            matrix[1:, 4:] = tail_effectiveness

        return matrix

    def allocate(
        self,
        total_thrust: float,
        roll_moment: float = 0.0,
        pitch_moment: float = 0.0,
        yaw_moment: float = 0.0,
        tilt_cmd: float = 0.0,
        thrust_bias: float = 0.0,
        *,
        tail_effectiveness: np.ndarray | None = None,
        moment_bias: np.ndarray | None = None,
        tail_limit: float = 0.2,
        axis_weights: np.ndarray | None = None,
    ) -> dict[str, float | bool | np.ndarray]:
        """Allocate a bounded actuator solution and report any unachieved command."""
        requested = np.asarray(
            [total_thrust + thrust_bias, roll_moment, pitch_moment, yaw_moment],
            dtype=float,
        )
        if not np.all(np.isfinite(requested)):
            raise ValueError("thrust and moment requests must be finite")
        requested[0] = max(requested[0], 0.0)
        if not np.isfinite(tail_limit) or tail_limit <= 0.0:
            raise ValueError("tail deflection limit must be positive")
        if axis_weights is None:
            axis_weights = np.ones(4, dtype=float)
        else:
            axis_weights = np.asarray(axis_weights, dtype=float)
            if axis_weights.shape != (4,) or not np.all(np.isfinite(axis_weights)) or np.any(axis_weights <= 0.0):
                raise ValueError("axis weights must be four positive finite values")

        if moment_bias is None:
            baseline_moment = np.zeros(3, dtype=float)
        else:
            baseline_moment = np.asarray(moment_bias, dtype=float)
            if baseline_moment.shape != (3,) or not np.all(np.isfinite(baseline_moment)):
                raise ValueError("moment bias must be a finite 3-vector")

        matrix = self.allocation_matrix(tilt_cmd, tail_effectiveness)
        target = requested.copy()
        target[1:] -= baseline_moment

        maximum_rotor_thrust = self.k_thrust * self.max_omega**2
        actuator_scales = np.array(
            [maximum_rotor_thrust] * 4 + [tail_limit] * 2,
            dtype=float,
        )
        output_scales = np.array(
            [
                max(requested[0], 1.0),
                max(requested[0] * self.arm_length, 1.0),
                max(requested[0] * self.arm_length, 1.0),
                max(requested[0] * self.arm_length, 1.0),
            ],
            dtype=float,
        )
        normalized_matrix = (
            matrix
            * actuator_scales[np.newaxis, :]
            / output_scales[:, np.newaxis]
            * axis_weights[:, np.newaxis]
        )
        normalized_target = target / output_scales * axis_weights
        bounds = (
            np.array([0.0] * 4 + [-1.0, -1.0]),
            np.ones(6, dtype=float),
        )
        solution = lsq_linear(
            normalized_matrix,
            normalized_target,
            bounds=bounds,
            tol=1e-8,
            max_iter=100,
        )
        actuator_commands = solution.x * actuator_scales
        rotor_thrusts = actuator_commands[:4]
        tail_deflections = actuator_commands[4:]
        achieved = matrix @ actuator_commands
        achieved[1:] += baseline_moment
        residual = requested - achieved
        normalized_residual = residual / output_scales

        omega_commands = np.sqrt(rotor_thrusts / self.k_thrust)
        saturated = bool(
            np.any(solution.active_mask != 0)
            or np.max(np.abs(normalized_residual)) > 0.02
            or not solution.success
        )
        return {
            "omega_fl": float(omega_commands[0]),
            "omega_fr": float(omega_commands[1]),
            "omega_rl": float(omega_commands[2]),
            "omega_rr": float(omega_commands[3]),
            "delta_left": float(tail_deflections[0]),
            "delta_right": float(tail_deflections[1]),
            "tilt": float(tilt_cmd),
            "total_thrust": float(requested[0]),
            "allocation_matrix": matrix.copy(),
            "allocation_achieved": achieved.copy(),
            "allocation_residual": residual.copy(),
            "allocation_residual_normalized": normalized_residual.copy(),
            "allocation_saturated": saturated,
        }

    def compute(self, total_thrust: float, roll_moment: float = 0.0, pitch_moment: float = 0.0,
                yaw_moment: float = 0.0, tilt_cmd: float = 0.0, thrust_bias: float = 0.0, **kwargs):
        return self.allocate(
            total_thrust,
            roll_moment,
            pitch_moment,
            yaw_moment,
            tilt_cmd,
            thrust_bias,
            **kwargs,
        )

    def allocate_commands(self, total_thrust: float, roll_moment: float = 0.0, pitch_moment: float = 0.0,
                          yaw_moment: float = 0.0, tilt_cmd: float = 0.0, thrust_bias: float = 0.0, **kwargs):
        return self.allocate(
            total_thrust,
            roll_moment,
            pitch_moment,
            yaw_moment,
            tilt_cmd,
            thrust_bias,
            **kwargs,
        )
