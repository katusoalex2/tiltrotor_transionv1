"""Cascaded guidance, attitude, rate, scheduling, and actuator allocation."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.allocation.allocator import ControlAllocator
from src.control.envelope_protection import EnvelopeProtection
from src.control.pid import PIDController
from src.control.scheduler import GainScheduler
from src.control.transition_observer import TransitionStateObserver
from src.dynamics.frames import euler_rates_to_body_rates, rotation_earth_to_body
from src.dynamics.tiltrotor import GRAVITY


class ControlLawArchitecture:
    """Run the five Phase 6 control layers for one simulation step."""

    def __init__(
        self,
        mass: float,
        k_thrust: float,
        max_omega: float,
        motor_positions,
        vtail_model=None,
        *,
        aero_model=None,
        altitude_gains: dict[str, float] | None = None,
        speed_gains: dict[str, float] | None = None,
        attitude_gains: dict[str, float] | None = None,
        rate_gains: dict[str, float] | None = None,
        hover_rate_scale: float = 1.5,
        cruise_rate_scale: float = 0.75,
        max_tail_deflection_rad: float = 0.2,
        max_moment_nm: float = 5.0,
        transition_observer: TransitionStateObserver | None = None,
        envelope_protection: EnvelopeProtection | None = None,
    ):
        self.mass = float(mass)
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass must be positive and finite")
        if max_tail_deflection_rad <= 0.0 or max_moment_nm <= 0.0:
            raise ValueError("tail and moment limits must be positive")

        altitude_gains = altitude_gains or {"kp": 1.5, "ki": 0.05, "kd": 0.1}
        speed_gains = speed_gains or {"kp": 0.1, "ki": 0.0, "kd": 0.01}
        attitude_gains = attitude_gains or {"kp": 1.0, "ki": 0.0, "kd": 0.05}
        rate_gains = rate_gains or {"kp": 0.5, "ki": 0.0, "kd": 0.01}
        for gains in (altitude_gains, speed_gains, attitude_gains, rate_gains):
            if set(gains) != {"kp", "ki", "kd"} or not np.all(
                np.isfinite([gains["kp"], gains["ki"], gains["kd"]])
            ):
                raise ValueError("each PID gain set must contain finite kp, ki, and kd values")

        self.max_tail_deflection_rad = float(max_tail_deflection_rad)
        self.max_moment_nm = float(max_moment_nm)
        self.altitude_controller = PIDController(**altitude_gains, limit=2.5)
        self.speed_controller = PIDController(**speed_gains, limit=0.35)
        self.attitude_controllers = [
            PIDController(**attitude_gains, limit=1.5),
            PIDController(**attitude_gains, limit=1.5),
            PIDController(**attitude_gains, limit=0.8),
        ]

        hover_gains = {key: float(value) * hover_rate_scale for key, value in rate_gains.items()}
        cruise_gains = {key: float(value) * cruise_rate_scale for key, value in rate_gains.items()}
        self.gain_scheduler = GainScheduler(
            hover_gains=hover_gains,
            cruise_gains=cruise_gains,
        )
        self.rate_controllers = [
            PIDController(**hover_gains, limit=self.max_moment_nm)
            for _ in range(3)
        ]

        self.motor_positions = tuple(tuple(position) for position in motor_positions)
        self._tilt_flags = tuple(
            bool(position[3]) if len(position) == 4 else False
            for position in self.motor_positions
        )
        self.aero_model = aero_model
        self.transition_observer = transition_observer or TransitionStateObserver()
        self.envelope_protection = envelope_protection
        self._last_control_authority = 1.0
        self.vtail_model = (
            aero_model.vtail
            if aero_model is not None
            else vtail_model
        )
        self.allocator = ControlAllocator(
            k_thrust=k_thrust,
            max_omega=max_omega,
            motor_positions=self.motor_positions,
        )
        self.last_allocation_matrix = np.zeros((4, 6), dtype=float)
        self.last_allocation_residual = np.zeros(4, dtype=float)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _guidance(
        self,
        state,
        reference: dict[str, Any],
        air_velocity_earth: np.ndarray,
        wind_earth: np.ndarray,
        actuator_tilt_rad: float,
        dt: float,
    ) -> dict[str, float]:
        current_altitude = -float(state.pos[2])
        target_altitude = float(reference.get("altitude", current_altitude))
        target_speed = max(float(reference.get("speed", 0.0)), 0.0)
        current_speed = float(np.linalg.norm(air_velocity_earth[:2]))
        speed_error = target_speed - current_speed
        altitude_error = target_altitude - current_altitude

        vertical_acceleration = self.altitude_controller.compute(
            target_altitude,
            current_altitude,
            dt,
        )
        desired_pitch = float(np.clip(
            -self.speed_controller.compute(target_speed, current_speed, dt),
            -np.deg2rad(25.0),
            np.deg2rad(25.0),
        ))

        current_yaw = float(state.att[2])
        current_north, current_east = float(state.pos[0]), float(state.pos[1])
        if "north" in reference and "east" in reference:
            north_error = float(reference["north"]) - current_north
            east_error = float(reference["east"]) - current_east
            path_distance = float(np.hypot(north_error, east_error))
            if path_distance > 5.0:
                target_heading = float(np.arctan2(east_error, north_error))
            else:
                target_heading = float(reference.get("heading", current_yaw))
        else:
            north_error = 0.0
            east_error = 0.0
            path_distance = 0.0
            target_heading = float(reference.get("heading", current_yaw))

        heading_error = self._wrap_angle(target_heading - current_yaw)
        desired_yaw_rate = float(np.clip(heading_error, -0.8, 0.8))
        desired_roll = float(np.clip(
            np.arctan2(current_speed * desired_yaw_rate, GRAVITY),
            -np.deg2rad(30.0),
            np.deg2rad(30.0),
        ))
        target_tilt = float(np.clip(
            float(reference.get("tilt", state.tilt)),
            0.0,
            np.pi / 2.0,
        ))

        vertical_support_fraction = float(np.mean([
            np.cos(actuator_tilt_rad if tilts else 0.0)
            for tilts in self._tilt_flags
        ]))
        support_fraction = max(vertical_support_fraction, 0.25)
        requested_thrust = self.mass * (GRAVITY + vertical_acceleration) / support_fraction
        if self.aero_model is not None:
            for _ in range(3):
                aero_force, _, _, _, _ = self.aero_model.compute_forces_and_moments(
                    state.vel,
                    wind_earth,
                    state.att,
                    requested_thrust,
                    actuator_tilt_rad,
                )
                tail_force, _ = self.aero_model.compute_tail_forces(
                    state.vel,
                    wind_earth,
                    state.att,
                    0.0,
                    0.0,
                )
                body_to_earth = rotation_earth_to_body(*state.att).T
                aerodynamic_down_force = float(
                    (body_to_earth @ (aero_force + tail_force))[2]
                )
                compensated_thrust = (
                    self.mass * (GRAVITY + vertical_acceleration)
                    + aerodynamic_down_force
                ) / support_fraction
                compensated_thrust = max(compensated_thrust, 0.0)
                if abs(compensated_thrust - requested_thrust) < 1e-3:
                    requested_thrust = compensated_thrust
                    break
                requested_thrust = compensated_thrust

        return {
            "target_altitude": target_altitude,
            "target_speed": target_speed,
            "current_speed": current_speed,
            "altitude_error": altitude_error,
            "speed_error": speed_error,
            "north_error": north_error,
            "east_error": east_error,
            "path_distance": path_distance,
            "heading_error": heading_error,
            "target_heading": target_heading,
            "desired_roll": desired_roll,
            "desired_pitch": desired_pitch,
            "target_tilt": target_tilt,
            "vertical_acceleration": float(vertical_acceleration),
            "requested_thrust": float(max(requested_thrust, 0.0)),
        }

    def _tail_effectiveness(
        self,
        state,
        wind_earth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        effectiveness = np.zeros((3, 2), dtype=float)
        bias_moment = np.zeros(3, dtype=float)
        if self.vtail_model is None:
            return effectiveness, bias_moment

        air_velocity_earth = np.asarray(state.vel, dtype=float) - wind_earth
        _, baseline_moment = self.vtail_model.compute_forces_and_moments(
            air_velocity_earth,
            state.att,
            0.0,
            0.0,
        )
        bias_moment = np.asarray(baseline_moment, dtype=float)
        epsilon = 1e-4
        for index in range(2):
            plus = [0.0, 0.0]
            minus = [0.0, 0.0]
            plus[index] = epsilon
            minus[index] = -epsilon
            _, plus_moment = self.vtail_model.compute_forces_and_moments(
                air_velocity_earth,
                state.att,
                plus[0],
                plus[1],
            )
            _, minus_moment = self.vtail_model.compute_forces_and_moments(
                air_velocity_earth,
                state.att,
                minus[0],
                minus[1],
            )
            effectiveness[:, index] = (
                np.asarray(plus_moment, dtype=float) - np.asarray(minus_moment, dtype=float)
            ) / (2.0 * epsilon)
        return effectiveness, bias_moment

    def _aerodynamic_moment_bias(
        self,
        state,
        wind_earth: np.ndarray,
        total_thrust: float,
        tilt_rad: float,
    ) -> np.ndarray:
        if self.aero_model is None:
            return np.zeros(3, dtype=float)
        _, moment, _, _, _ = self.aero_model.compute_forces_and_moments(
            state.vel,
            wind_earth,
            state.att,
            total_thrust,
            tilt_rad,
        )
        return np.asarray(moment, dtype=float)

    def compute(
        self,
        state,
        reference: dict[str, Any],
        dt: float,
        wind_earth: np.ndarray | None = None,
        *,
        actuator_tilt_rad: float | None = None,
        available_control_authority: float | None = None,
    ) -> dict[str, Any]:
        """Compute guidance, attitude/rate feedback, scheduled gains, and actuator commands."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("time step must be positive and finite")
        wind = np.zeros(3, dtype=float) if wind_earth is None else np.asarray(wind_earth, dtype=float)
        if wind.shape != (3,) or not np.all(np.isfinite(wind)):
            raise ValueError("wind must be a finite 3-vector")

        actuator_tilt = float(state.tilt if actuator_tilt_rad is None else actuator_tilt_rad)
        actuator_tilt = float(np.clip(actuator_tilt, 0.0, np.pi / 2.0))
        air_velocity_earth = np.asarray(state.vel, dtype=float) - wind
        transition_estimate = self.transition_observer.update(state, wind, dt)
        control_authority = (
            self._last_control_authority
            if available_control_authority is None
            else float(available_control_authority)
        )
        if not np.isfinite(control_authority) or not 0.0 <= control_authority <= 1.0:
            raise ValueError("available control authority must be between 0 and 1")
        requested_speed = float(reference.get("speed", 0.0))
        requested_tilt = float(reference.get("tilt", state.tilt))
        effective_reference = dict(reference)
        protection = None
        if self.envelope_protection is not None:
            protection = self.envelope_protection.apply(
                requested_speed,
                requested_tilt,
                transition_estimate,
                available_control_authority=control_authority,
            )
            effective_reference["speed"] = protection.speed_mps
            effective_reference["tilt"] = protection.tilt_rad
        guidance = self._guidance(
            state,
            effective_reference,
            air_velocity_earth,
            wind,
            actuator_tilt,
            dt,
        )
        current_attitude = np.asarray(state.att, dtype=float)

        yaw_error = self._wrap_angle(guidance["target_heading"] - current_attitude[2])
        attitude_targets = np.array([
            guidance["desired_roll"],
            guidance["desired_pitch"],
            current_attitude[2] + yaw_error,
        ])
        desired_euler_rates = np.array([
            controller.compute(float(target), float(measurement), dt)
            for controller, target, measurement in zip(
                self.attitude_controllers,
                attitude_targets,
                current_attitude,
            )
        ])
        desired_rates = euler_rates_to_body_rates(
            float(current_attitude[0]),
            float(current_attitude[1]),
            desired_euler_rates,
        )

        current_speed = guidance["current_speed"]
        scheduled_gains = self.gain_scheduler.get_gains(current_speed, actuator_tilt)
        for controller in self.rate_controllers:
            controller.set_gains(**scheduled_gains)
        rate_moments = np.array([
            controller.compute(float(target), float(measurement), dt)
            for controller, target, measurement in zip(
                self.rate_controllers,
                desired_rates,
                state.omega,
            )
        ])

        tail_effectiveness, tail_bias_moment = self._tail_effectiveness(state, wind)

        def allocate_at_tilt(
            tilt: float,
            prioritize_pitch: bool = False,
        ) -> dict[str, Any]:
            aerodynamic_moment_bias = self._aerodynamic_moment_bias(
                state,
                wind,
                guidance["requested_thrust"],
                tilt,
            )
            return self.allocator.allocate(
                guidance["requested_thrust"],
                roll_moment=float(rate_moments[0]),
                pitch_moment=float(rate_moments[1]),
                yaw_moment=float(rate_moments[2]),
                tilt_cmd=tilt,
                tail_effectiveness=tail_effectiveness,
                moment_bias=tail_bias_moment + aerodynamic_moment_bias,
                tail_limit=self.max_tail_deflection_rad,
                axis_weights=np.array([1.0, 1.0, 3.0, 1.0])
                if prioritize_pitch
                else None,
            )

        pitch_reference_scale = max(
            guidance["requested_thrust"] * self.allocator.arm_length,
            1.0,
        )

        def pitch_residual_fraction(allocation_result: dict[str, Any]) -> float:
            return abs(float(allocation_result["allocation_residual"][2])) / pitch_reference_scale

        def allocate_with_pitch_priority(tilt: float) -> dict[str, Any]:
            best_allocation = allocate_at_tilt(tilt)
            if pitch_residual_fraction(best_allocation) > 0.01:
                candidate = allocate_at_tilt(tilt, prioritize_pitch=True)
                if pitch_residual_fraction(candidate) < pitch_residual_fraction(best_allocation):
                    best_allocation = candidate
            return best_allocation

        allocation = allocate_with_pitch_priority(actuator_tilt)
        tilt_command = guidance["target_tilt"]
        tilt_limited = False
        pitch_error = abs(
            self._wrap_angle(guidance["desired_pitch"] - current_attitude[1])
        )
        attitude_protection = (
            pitch_error > np.deg2rad(20.0)
            or abs(float(state.omega[1])) > np.deg2rad(35.0)
        )
        pitch_allocation_shortfall = (
            abs(float(allocation["allocation_residual_normalized"][2])) > 0.01
        )
        if pitch_allocation_shortfall or attitude_protection:
            current_tilt = float(np.clip(state.tilt, 0.0, np.pi / 2.0))
            current_allocation = allocate_with_pitch_priority(current_tilt)
            tilt_step = abs(actuator_tilt - current_tilt)
            if (
                not attitude_protection
                and pitch_residual_fraction(current_allocation) <= 0.01
            ):
                allocation = current_allocation
                tilt_command = current_tilt
            elif tilt_step > 0.0:
                tilt_command = max(current_tilt - tilt_step, 0.0)
                allocation = allocate_with_pitch_priority(tilt_command)
            else:
                allocation = current_allocation
                tilt_command = current_tilt
            tilt_limited = True

        self.last_allocation_matrix = np.asarray(allocation["allocation_matrix"], dtype=float)
        self.last_allocation_residual = np.asarray(allocation["allocation_residual"], dtype=float)
        residual_fraction = float(
            np.max(np.abs(allocation["allocation_residual_normalized"]))
        )
        self._last_control_authority = float(np.clip(1.0 - residual_fraction, 0.0, 1.0))

        return {
            **allocation,
            "desired_roll": guidance["desired_roll"],
            "desired_pitch": guidance["desired_pitch"],
            "desired_heading": guidance["target_heading"],
            "desired_p": float(desired_rates[0]),
            "desired_q": float(desired_rates[1]),
            "desired_r": float(desired_rates[2]),
            "moment_l_command": float(rate_moments[0]),
            "moment_m_command": float(rate_moments[1]),
            "moment_n_command": float(rate_moments[2]),
            "gain_blend": self.gain_scheduler.transition_blend(current_speed, actuator_tilt),
            "gain_kp": scheduled_gains["kp"],
            "gain_ki": scheduled_gains["ki"],
            "gain_kd": scheduled_gains["kd"],
            "airspeed_mps": current_speed,
            "transition_airspeed_mps": transition_estimate.airspeed_mps,
            "transition_alpha_rad": transition_estimate.alpha_rad,
            "transition_beta_rad": transition_estimate.beta_rad,
            "transition_blend": transition_estimate.transition_blend,
            "transition_phase": transition_estimate.phase,
            "available_control_authority": control_authority,
            "guidance_altitude_error": guidance["altitude_error"],
            "guidance_speed_error": guidance["speed_error"],
            "guidance_north_error": guidance["north_error"],
            "guidance_east_error": guidance["east_error"],
            "guidance_heading_error": guidance["heading_error"],
            "guidance_path_distance": guidance["path_distance"],
            "requested_thrust": guidance["requested_thrust"],
            "vertical_acceleration_command": guidance["vertical_acceleration"],
            "requested_speed": requested_speed,
            "protected_speed": guidance["target_speed"],
            "requested_tilt": requested_tilt,
            "protected_tilt": guidance["target_tilt"],
            "envelope_protection_active": protection is not None,
            "envelope_protected": protection.protected if protection is not None else False,
            "envelope_within": protection.within_envelope if protection is not None else True,
            "envelope_reasons": protection.reasons if protection is not None else (),
            "tilt_limited": tilt_limited,
            "tilt": tilt_command,
        }
