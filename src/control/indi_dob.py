"""INDI controller with Disturbance Observer (DOB) augmentation.

The DOB fixes the failure mode we saw in Monte Carlo: INDI crashes at high gusts
because it misattributes disturbance acceleration to its own actuator.
"""

import numpy as np


class INDIControllerWithDOB:
    """INDI + Disturbance Observer."""

    def __init__(
        self,
        B_eff: float = -50.0,
        pitch_gain: float = 4.0,
        filter_alpha: float = 0.3,
        dob_filter_alpha: float = 0.15,
        limit: float = 0.2,
        dob_enabled: bool = True,
    ):
        self.B_eff = B_eff
        self.pitch_gain = pitch_gain
        self.filter_alpha = filter_alpha
        self.dob_filter_alpha = dob_filter_alpha
        self.limit = limit
        self.dob_enabled = dob_enabled

        self.prev_vtail = 0.0
        self.prev_pitch_rate = 0.0
        self.filtered_accel = 0.0
        self.disturbance_estimate = 0.0

    def compute(self, desired_pitch, current_pitch, current_pitch_rate, dt):
        if dt <= 0:
            return self.prev_vtail

        # 1. Measure angular acceleration
        raw_accel = (current_pitch_rate - self.prev_pitch_rate) / dt
        self.filtered_accel = (
            self.filter_alpha * raw_accel + (1 - self.filter_alpha) * self.filtered_accel
        )

        # 2. DOB: estimate disturbance acceleration
        if self.dob_enabled:
            # What accel did the last V-tail SHOULD have produced?
            accel_from_my_command = self.B_eff * self.prev_vtail
            # The difference is the disturbance
            raw_disturbance = self.filtered_accel - accel_from_my_command
            # Low-pass filter the disturbance
            self.disturbance_estimate = (
                self.dob_filter_alpha * raw_disturbance
                + (1 - self.dob_filter_alpha) * self.disturbance_estimate
            )

        # 3. Desired rate and rate-dot
        pitch_error = desired_pitch - current_pitch
        desired_pitch_rate = self.pitch_gain * pitch_error
        desired_rate_dot = (desired_pitch_rate - current_pitch_rate) / 0.1

        # 4. INDI with DOB correction
        # We want: desired_rate_dot = B_eff * new_vtail + disturbance
        # So: delta_vtail = (desired_rate_dot - filtered_accel + disturbance) / B_eff
        # But if DOB is enabled, we don't double-count disturbance:
        correction = self.disturbance_estimate if self.dob_enabled else 0.0
        delta_vtail = (desired_rate_dot - self.filtered_accel + correction) / self.B_eff

        new_vtail = self.prev_vtail + delta_vtail
        new_vtail = np.clip(new_vtail, -self.limit, self.limit)

        self.prev_vtail = new_vtail
        self.prev_pitch_rate = current_pitch_rate
        return new_vtail

    def reset(self):
        self.prev_vtail = 0.0
        self.prev_pitch_rate = 0.0
        self.filtered_accel = 0.0
        self.disturbance_estimate = 0.0