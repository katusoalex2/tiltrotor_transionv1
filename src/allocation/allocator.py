"""Controller allocation for mapping desired forces and moments into rotor commands."""

import numpy as np


class ControlAllocator:
    """Allocate a collective thrust and moment request to the four rotor motors."""

    def __init__(self, k_thrust: float = 1e-5, max_omega: float = 3000.0, arm_length: float = 0.4):
        self.k_thrust = float(k_thrust)
        self.max_omega = float(max_omega)
        self.arm_length = float(arm_length)

    def _omega_from_thrust(self, thrust: float) -> float:
        return float(np.sqrt(max(thrust, 0.0) / max(self.k_thrust, 1e-12)))

    def allocate(self, total_thrust: float, roll_moment: float = 0.0, pitch_moment: float = 0.0,
                 yaw_moment: float = 0.0, tilt_cmd: float = 0.0, thrust_bias: float = 0.0):
        """Return per-rotor RPMs and auxiliary control commands.

        The model uses a four-rotor symmetric layout with front/rear differential thrust
        and left/right differential thrust for roll and yaw trimming.
        """
        total_thrust = float(total_thrust) + float(thrust_bias)
        if total_thrust < 0.0:
            total_thrust = 0.0

        front = 0.5 * total_thrust + 0.5 * pitch_moment
        rear = 0.5 * total_thrust - 0.5 * pitch_moment
        left = 0.5 * total_thrust + 0.5 * roll_moment
        right = 0.5 * total_thrust - 0.5 * roll_moment

        # Yaw moment is handled as front/rear asymmetry while keeping total thrust constant.
        yaw_split = 0.5 * yaw_moment
        fl = front + left + yaw_split
        fr = front - right - yaw_split
        rl = rear + left - yaw_split
        rr = rear - right + yaw_split

        rotor_thrusts = np.array([fl, fr, rl, rr], dtype=float)
        rotor_thrusts = np.clip(rotor_thrusts, 0.0, None)

        # Re-normalize while preserving the commanded total thrust if possible.
        rotor_total = float(np.sum(rotor_thrusts))
        if rotor_total > 0.0 and not np.isclose(rotor_total, total_thrust):
            rotor_thrusts *= total_thrust / rotor_total

        omega_fl = np.clip(self._omega_from_thrust(rotor_thrusts[0]), 0.0, self.max_omega)
        omega_fr = np.clip(self._omega_from_thrust(rotor_thrusts[1]), 0.0, self.max_omega)
        omega_rl = np.clip(self._omega_from_thrust(rotor_thrusts[2]), 0.0, self.max_omega)
        omega_rr = np.clip(self._omega_from_thrust(rotor_thrusts[3]), 0.0, self.max_omega)

        return {
            "omega_fl": float(omega_fl),
            "omega_fr": float(omega_fr),
            "omega_rl": float(omega_rl),
            "omega_rr": float(omega_rr),
            "tilt": float(tilt_cmd),
            "roll_moment": float(roll_moment),
            "pitch_moment": float(pitch_moment),
            "yaw_moment": float(yaw_moment),
            "total_thrust": float(total_thrust),
        }

    def compute(self, total_thrust: float, roll_moment: float = 0.0, pitch_moment: float = 0.0,
                yaw_moment: float = 0.0, tilt_cmd: float = 0.0, thrust_bias: float = 0.0):
        return self.allocate(total_thrust, roll_moment, pitch_moment, yaw_moment, tilt_cmd, thrust_bias)

    def allocate_commands(self, total_thrust: float, roll_moment: float = 0.0, pitch_moment: float = 0.0,
                          yaw_moment: float = 0.0, tilt_cmd: float = 0.0, thrust_bias: float = 0.0):
        return self.allocate(total_thrust, roll_moment, pitch_moment, yaw_moment, tilt_cmd, thrust_bias)
