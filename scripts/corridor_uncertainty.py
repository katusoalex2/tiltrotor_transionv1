"""Stress the same longitudinal controller with uncertain plant and wind inputs."""

from dataclasses import asdict, dataclass, replace

import numpy as np

from scripts.dynamic_corridor import FLOW_CHECK_TOLERANCE_DEG, flow_guard_margins
from scripts.refined_corridor import coverage
from scripts.transition_study import DURATION, LOW_SPEED_FULL_LOAD, TARGET_SPEED, path_rows, replay_trajectory, study_limits
from src.aero.wing_polar import low_speed_aerodynamic_factor
from src.configuration.aircraft_models import build_aircraft_dynamics
from src.configuration.aircraft_parameters import load_aircraft_parameters
from src.dynamics.transition import assess_rotor_electrical_point
from src.dynamics.transition_trajectory import RPM_PER_RAD_S


SEED = 404
MONTE_CARLO_DRAWS = 16
INNER_MARGINS = {
    "altitude_deviation_m": 0.75,
    "pitch_deg": 18.0,
    "pitch_rate_deg_s": 18.0,
    "tilt_rate_fraction": 0.85,
    "motor_rpm_reserve_fraction": 0.15,
    "current_reserve_fraction": 0.20,
    "voltage_headroom_v": 0.50,
    "differential_pitch_reserve_nm": 1.0,
    "tail_travel_reserve_deg": 1.5,
    "wingborne_tail_moment_reserve_nm": 0.15,
    "wingborne_stall_margin_deg": 5.0,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    mass_scale: float = 1.0
    cg_forward_m: float = 0.0
    lift_slope_scale: float = 1.0
    drag_scale: float = 1.0
    thrust_scale: float = 1.0
    tilt_time_constant_s: float = 0.0
    gust_mps: float = 0.0
    gust_start_s: float = 10.0

    def gust(self, time_s):
        fraction = (time_s - self.gust_start_s) / 5.0
        return self.gust_mps * np.sin(np.pi * fraction)**2 if 0 < fraction < 1 else 0.0

    def aircraft(self, baseline):
        shift = self.cg_forward_m
        pivot_x, pivot_z = baseline.tilt_pivot_xz_m
        return replace(
            baseline, mass=baseline.mass * self.mass_scale,
            inertia_xx=baseline.inertia_xx * self.mass_scale,
            inertia_yy=baseline.inertia_yy * self.mass_scale,
            inertia_zz=baseline.inertia_zz * self.mass_scale,
            k_thrust=baseline.k_thrust * self.thrust_scale,
            wing_lift_slope=baseline.wing_lift_slope * self.lift_slope_scale,
            wing_drag_coeff=baseline.wing_drag_coeff * self.drag_scale,
            wing_root_le_x_m=baseline.wing_root_le_x_m - shift,
            wing_tip_le_x_m=baseline.wing_tip_le_x_m - shift,
            vtail_root_le_x_m=baseline.vtail_root_le_x_m - shift,
            vtail_tip_le_x_m=baseline.vtail_tip_le_x_m - shift,
            tilt_pivot_xz_m=(pivot_x - shift, pivot_z),
            motor_positions=tuple((x - shift, y, z, enabled) for x, y, z, enabled in baseline.motor_positions),
        )


def scenarios():
    rng = np.random.default_rng(SEED)
    draws = [Scenario(
        f"mc_{index + 1:02d}", float(rng.uniform(0.95, 1.05)), float(rng.uniform(-0.006, 0.006)),
        float(rng.uniform(0.92, 1.08)), float(rng.uniform(0.88, 1.12)), float(rng.uniform(0.92, 1.05)),
        float(rng.uniform(0.03, 0.20)), float(rng.uniform(-0.8, 0.8)), float(rng.uniform(5.0, 22.0)),
    ) for index in range(MONTE_CARLO_DRAWS)]
    return [
        Scenario("nominal"),
        Scenario("early_headwind", gust_mps=-0.8, gust_start_s=5),
        Scenario("handoff_tailwind", gust_mps=0.8, gust_start_s=10),
        Scenario("wingborne_headwind", gust_mps=-0.8, gust_start_s=17),
        Scenario("late_tailwind", gust_mps=0.8, gust_start_s=22),
        Scenario("heavy_low_thrust", 1.05, 0.006, 0.92, 1.12, 0.92, 0.20, 0.8, 10),
        *draws,
    ]


def fill_margins(aircraft, limits, rows):
    """Calculate measured-in-simulation actuator and aerodynamic reserves per sample."""
    parameters = aircraft.electrical_parameters
    tail_dihedral_cosine = np.cos(np.deg2rad(aircraft.vtail_dihedral_deg))
    for row in rows:
        front_rpm, rear_rpm = row["front_rpm_per_motor"], row["rear_rpm_per_motor"]
        electrical = assess_rotor_electrical_point([front_rpm] * 2 + [rear_rpm] * 2, parameters)
        row["battery_power_w"] = electrical["battery_power_w"]
        row["voltage_headroom_v"] = electrical["battery_voltage_v"] - electrical["maximum_required_motor_voltage_v"]
        row["current_reserve_fraction"] = 1 - max(
            electrical["motor_current_a"] / electrical["motor_current_limit_a"],
            electrical["battery_current_a"] / electrical["battery_current_limit_a"],
        )
        row["motor_rpm_reserve_fraction"] = 1 - max(
            front_rpm / (limits.front_omega * RPM_PER_RAD_S), rear_rpm / (limits.rear_omega * RPM_PER_RAD_S),
        )
        row["tail_travel_reserve_deg"] = np.rad2deg(limits.tail_rad) - abs(row["vtail_deg"])
        speed = row["airspeed_mps"]
        alpha = np.deg2rad(row["freestream_alpha_deg"]) if speed >= 0.5 else 0.0
        factor = low_speed_aerodynamic_factor(speed, LOW_SPEED_FULL_LOAD)
        q = 0.5 * aircraft.air_density * speed**2 * factor
        moment_scale = q * aircraft.vtail_area * aircraft.vtail_arm_length * tail_dihedral_cosine
        angles = np.array([-limits.tail_rad, np.deg2rad(row["vtail_deg"]), limits.tail_rad])
        coefficients = aircraft.vtail_lift_slope * (alpha + angles * tail_dihedral_cosine)
        moments = -moment_scale * np.clip(coefficients, -aircraft.vtail_max_lift_coefficient, aircraft.vtail_max_lift_coefficient)
        row["tail_pitch_reserve_nm"] = min(moments[0] - moments[1], moments[1] - moments[2])
        row["stall_margin_deg"] = aircraft.wing_stall_angle_deg - abs(row["effective_alpha_deg"])


def screen_route(aircraft, limits, rows, *, inner=False):
    def peak(field):
        return max(abs(row[field]) for row in rows)

    def minimum(field, eligible=rows):
        return min((row[field] for row in eligible), default=float("nan"))

    last = rows[-1]
    wingborne = [row for row in rows if row["airspeed_mps"] >= limits.wingborne_speed_mps]
    summary = {
        "completed_time_s": last["time_s"],
        "altitude_deviation_m": peak("altitude_change_m"),
        "pitch_deg": peak("pitch_deg"),
        "pitch_rate_deg_s": peak("pitch_rate_deg_s"),
        "tilt_rate_deg_s": peak("tilt_rate_deg_s"),
        "forward_acceleration_mps2": peak("forward_acceleration_mps2"),
        "vertical_acceleration_mps2": peak("upward_acceleration_mps2"),
        "climb_rate_mps": peak("climb_rate_mps"),
        "motor_slew_rpm_s": max(peak("front_slew_rpm_s"), peak("rear_slew_rpm_s")),
        "tail_rate_deg_s": peak("vtail_rate_deg_s"),
        "tail_travel_reserve_deg": minimum("tail_travel_reserve_deg"),
        "wingborne_tail_moment_reserve_nm": minimum("tail_pitch_reserve_nm", wingborne),
        "wingborne_stall_margin_deg": minimum("stall_margin_deg", wingborne),
        "differential_pitch_reserve_nm": minimum("differential_pitch_reserve_nm"),
        "motor_rpm_reserve_fraction": minimum("motor_rpm_reserve_fraction"),
        "current_reserve_fraction": minimum("current_reserve_fraction"),
        "voltage_headroom_v": minimum("voltage_headroom_v"),
        "energy_wh": np.trapezoid([r["battery_power_w"] for r in rows], [r["time_s"] for r in rows]) / 3600,
        "terminal_speed_mps": last["airspeed_mps"],
        "terminal_tilt_deg": last["front_tilt_deg"],
    }
    upper = {
        "altitude_deviation_m": limits.altitude_loss_m,
        "pitch_deg": limits.pitch_limit_deg,
        "pitch_rate_deg_s": limits.pitch_rate_deg_s,
        "tilt_rate_deg_s": np.rad2deg(aircraft.max_tilt_rate),
        "forward_acceleration_mps2": limits.longitudinal_accel_mps2,
        "vertical_acceleration_mps2": limits.vertical_accel_mps2,
        "climb_rate_mps": limits.climb_rate_mps,
        "motor_slew_rpm_s": limits.motor_slew_rpm_s,
        "tail_rate_deg_s": limits.tail_rate_deg_s,
    }
    lower = {
        "tail_travel_reserve_deg": 0.0, "wingborne_stall_margin_deg": limits.stall_reserve_deg,
        "differential_pitch_reserve_nm": limits.pitch_reserve_nm, "motor_rpm_reserve_fraction": 0.0,
        "current_reserve_fraction": 0.0, "voltage_headroom_v": 0.0,
    }
    if inner:
        for key in ("altitude_deviation_m", "pitch_deg", "pitch_rate_deg_s"):
            upper[key] = INNER_MARGINS[key]
        upper["tilt_rate_deg_s"] *= INNER_MARGINS["tilt_rate_fraction"]
        lower.update({key: INNER_MARGINS[key] for key in (*lower, "wingborne_tail_moment_reserve_nm")})
    events = []
    def crossing(key, samples, limit, below=False):
        for row, observed in samples:
            violated = observed < limit - 1e-5 if below else observed > limit + 1e-5
            if violated:
                events.append((row["time_s"], key, observed, limit, row))
                break

    fields = {"altitude_deviation_m": "altitude_change_m", "tail_rate_deg_s": "vtail_rate_deg_s",
              "vertical_acceleration_mps2": "upward_acceleration_mps2"}
    for key, limit in upper.items():
        samples = ((row, max(abs(row["front_slew_rpm_s"]), abs(row["rear_slew_rpm_s"]))) for row in rows) if key == "motor_slew_rpm_s" else (
            (row, abs(row[fields.get(key, key)])) for row in rows
        )
        crossing(key, samples, limit)
    for key, limit in lower.items():
        field = {"wingborne_stall_margin_deg": "stall_margin_deg",
                 "wingborne_tail_moment_reserve_nm": "tail_pitch_reserve_nm"}.get(key, key)
        crossing(key, ((row, row[field]) for row in (wingborne if key.startswith("wingborne_") else rows)), limit, below=True)
    crossing("flow_guard", zip(rows, flow_guard_margins(aircraft, limits, rows)), -FLOW_CHECK_TOLERANCE_DEG, below=True)
    failed = [event[1] for event in events]
    if last["time_s"] < DURATION - 1e-6:
        failed.append("incomplete")
    elif (abs(last["airspeed_mps"] - TARGET_SPEED) > (0.30 if inner else 0.15)
          or abs(last["front_tilt_deg"]) > (0.4 if inner else 1e-5)
          or abs(last["altitude_change_m"]) > (0.40 if inner else 0.15)
          or abs(last["climb_rate_mps"]) > 0.15
          or abs(last["pitch_rate_deg_s"]) > 1.0
          or abs(last["forward_acceleration_mps2"]) > 0.15):
        failed.append("terminal_tracking")
    if events:
        event_time, check, observed, limit, row = min(events, key=lambda event: event[0])
        summary |= {
            "first_violation_time_s": event_time, "first_violated_constraint": check,
            "first_violation_value": observed, "first_violation_limit": limit,
            **{f"first_violation_{field}": row[field] for field in (
                "airspeed_mps", "front_tilt_deg", "effective_alpha_deg", "vtail_deg",
                "front_vertical_thrust_n", "rear_vertical_thrust_n", "wing_vertical_lift_n",
                "vtail_pitch_moment_nm", "pitch_rate_deg_s",
            )},
        }
    elif failed:
        summary |= {"first_violation_time_s": last["time_s"], "first_violated_constraint": failed[0]}
    return summary, failed


def run_case(task, *, wind_at=None, tail_reserve_rad=0.0):
    route_id, plan, scenario = task
    aircraft = load_aircraft_parameters()
    plant = scenario.aircraft(aircraft)
    limits = study_limits(aircraft)
    dynamics = build_aircraft_dynamics(plant, low_speed_full_load_mps=LOW_SPEED_FULL_LOAD)
    wind = scenario.gust if wind_at is None else wind_at
    time, motion, controls, loads, discrepancy = replay_trajectory(
        aircraft, dynamics, limits, *plan, step_s=0.02, stop_on_limit=True,
        plant_aircraft=plant, wind_at=wind, tilt_response_s=scenario.tilt_time_constant_s,
        tail_reserve_rad=tail_reserve_rad,
    )
    rows = path_rows(plant, limits, time, motion, controls, loads, wind_at=wind)
    fill_margins(plant, limits, rows)
    summary, failed = screen_route(plant, limits, rows, inner=True)
    if discrepancy > 1e-6:
        failed.append("load_parity")
    cells, _ = coverage(rows)
    return {"route_id": route_id, **asdict(scenario), **summary, "passed": not failed,
            "failed_checks": " ".join(failed), "load_parity_max_difference": discrepancy}, set(cells)
