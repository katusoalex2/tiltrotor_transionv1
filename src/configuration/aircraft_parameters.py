"""Load aircraft inputs for backend calculations from one YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, isfinite, pi
from pathlib import Path

import yaml


DEFAULT_AIRCRAFT_PATH = Path(__file__).resolve().parents[2] / "UAM Configurations" / "aircraft.yaml"
ELECTRICAL_OPTIONAL_FIELDS = (
    "motor_resistance_ohm",
    "motor_no_load_current_a",
    "motor_continuous_current_a",
    "propeller_torque_coefficient_nm_s2_rad2",
    "battery_internal_resistance_ohm",
    "battery_capacity_ah",
    "battery_c_rating",
    "battery_continuous_current_a",
    "esc_efficiency",
    "esc_continuous_current_a",
)


@dataclass(frozen=True)
class AircraftParameters:
    mass: float
    inertia_xx: float
    inertia_yy: float
    inertia_zz: float
    battery_cells: int
    cell_voltage_v: float
    motor_kv_rpm_per_volt: float
    motor_speed_factor: float
    k_thrust: float
    motor_resistance_ohm: float | None
    motor_no_load_current_a: float | None
    motor_continuous_current_a: float | None
    rear_thrust_reserve_fraction: float
    prop_diameter_m: float
    prop_mass_kg: float
    propeller_torque_coefficient_nm_s2_rad2: float | None
    battery_internal_resistance_ohm: float | None
    battery_capacity_ah: float | None
    battery_c_rating: float | None
    battery_continuous_current_a: float | None
    esc_efficiency: float | None
    esc_continuous_current_a: float | None
    max_tilt_rate: float
    tilt_pivot_xz_m: tuple[float, float]
    motor_positions: tuple[tuple[float, float, float, bool], ...]
    wingspan: float
    root_chord: float
    tip_chord: float
    wing_root_le_x_m: float
    wing_tip_le_x_m: float
    fuselage_length_m: float
    wing_lift_coefficient_zero: float
    wing_lift_slope: float
    wing_stall_angle_deg: float
    wing_drag_coeff: float
    oswald_efficiency: float
    wing_aerodynamic_center_fraction_mac: float
    wing_pitch_moment_zero: float
    wing_pitch_moment_slope: float
    vtail_area: float
    vtail_root_le_x_m: float
    vtail_root_chord_m: float
    vtail_tip_le_x_m: float
    vtail_tip_y_m: float
    vtail_tip_z_m: float
    vtail_tip_chord_fraction: float
    vtail_aerodynamic_center_fraction_chord: float
    vtail_lift_slope: float
    vtail_drag_coeff: float
    vtail_aspect_ratio: float
    vtail_oswald_efficiency: float
    vtail_max_lift_coefficient: float
    vtail_max_deflection_deg: float
    air_density: float
    gravity: float

    @property
    def wing_area(self) -> float:
        """Trapezoidal planform area derived from full span and root/tip chords."""
        return self.wingspan * (self.root_chord + self.tip_chord) / 2.0

    @property
    def wing_aspect_ratio(self) -> float:
        """Geometric aspect ratio derived from full span and planform area."""
        return self.wingspan**2 / self.wing_area

    @property
    def wing_mean_chord_m(self) -> float:
        """Mean aerodynamic chord derived from the trapezoidal wing planform."""
        taper_ratio = self.tip_chord / self.root_chord
        return (
            2.0
            / 3.0
            * self.root_chord
            * (1.0 + taper_ratio + taper_ratio**2)
            / (1.0 + taper_ratio)
        )

    @property
    def wing_aerodynamic_center_x_m(self) -> float:
        """Estimate AC x from the configured fraction of MAC and wing geometry."""
        taper_ratio = self.tip_chord / self.root_chord
        y_mac = (
            self.wingspan
            / 6.0
            * (1.0 + 2.0 * taper_ratio)
            / (1.0 + taper_ratio)
        )
        x_le_mac = self.wing_root_le_x_m + (
            self.wing_tip_le_x_m - self.wing_root_le_x_m
        ) * y_mac / (self.wingspan / 2.0)
        return (
            x_le_mac
            - self.wing_aerodynamic_center_fraction_mac * self.wing_mean_chord_m
        )

    @property
    def vtail_tip_chord_m(self) -> float:
        """Tip chord from the configured fraction of the root chord."""
        return self.vtail_root_chord_m * self.vtail_tip_chord_fraction

    @property
    def vtail_dihedral_deg(self) -> float:
        """Panel inclination above horizontal from its lateral and vertical offsets."""
        return degrees(atan2(-self.vtail_tip_z_m, self.vtail_tip_y_m))

    @property
    def vtail_arm_length(self) -> float:
        """CG-to-area-weighted quarter-chord arm for the trapezoidal V-tail."""
        root_chord = self.vtail_root_chord_m
        tip_chord = self.vtail_tip_chord_m
        chord_integral = (root_chord + tip_chord) / 2.0
        span_fraction_chord_integral = (
            root_chord + 2.0 * tip_chord
        ) / 6.0
        squared_chord_integral = (
            root_chord**2 + root_chord * tip_chord + tip_chord**2
        ) / 3.0
        leading_edge_delta = self.vtail_tip_le_x_m - self.vtail_root_le_x_m
        aerodynamic_center_x = (
            self.vtail_root_le_x_m * chord_integral
            + leading_edge_delta * span_fraction_chord_integral
            - self.vtail_aerodynamic_center_fraction_chord
            * squared_chord_integral
        ) / chord_integral
        return abs(aerodynamic_center_x)

    @property
    def vtail_max_deflection_rad(self) -> float:
        return self.vtail_max_deflection_deg * pi / 180.0

    @property
    def battery_voltage_v(self) -> float:
        return self.battery_cells * self.cell_voltage_v

    @property
    def electrical_parameters(self) -> dict[str, float | None]:
        """Return inputs in the format expected by the electrical screening model."""
        return {
            "motor_kv_rpm_per_volt": self.motor_kv_rpm_per_volt,
            "motor_resistance_ohm": self.motor_resistance_ohm,
            "motor_no_load_current_a": self.motor_no_load_current_a,
            "motor_continuous_current_a": self.motor_continuous_current_a,
            "propeller_torque_coefficient_nm_s2_rad2": (
                self.propeller_torque_coefficient_nm_s2_rad2
            ),
            "battery_nominal_voltage_v": self.battery_voltage_v,
            "battery_internal_resistance_ohm": self.battery_internal_resistance_ohm,
            "battery_capacity_ah": self.battery_capacity_ah,
            "battery_c_rating": self.battery_c_rating,
            "battery_continuous_current_a": self.battery_continuous_current_a,
            "esc_efficiency": self.esc_efficiency,
            "esc_continuous_current_a": self.esc_continuous_current_a,
        }

    @property
    def estimated_motor_rpm(self) -> float:
        return self.motor_kv_rpm_per_volt * self.battery_voltage_v * self.motor_speed_factor

    @property
    def max_omega(self) -> float:
        return self.estimated_motor_rpm * 2.0 * pi / 60.0

    @property
    def max_thrust_per_rotor(self) -> float:
        return self.k_thrust * self.max_omega**2


def load_aircraft_parameters(path: str | Path = DEFAULT_AIRCRAFT_PATH) -> AircraftParameters:
    """Read fresh values on each call; reject missing, invalid or unknown inputs."""
    with Path(path).open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file)
    if not isinstance(values, dict):
        raise ValueError("aircraft configuration must be a mapping")

    fields = AircraftParameters.__dataclass_fields__
    missing = fields.keys() - values.keys()
    extra = values.keys() - fields.keys()
    if missing or extra:
        raise ValueError(f"aircraft configuration fields: missing={sorted(missing)}, unknown={sorted(extra)}")

    scalar_fields = (
        fields.keys()
        - {"motor_positions", "tilt_pivot_xz_m", "battery_cells"}
        - set(ELECTRICAL_OPTIONAL_FIELDS)
    )
    for name in scalar_fields:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    for name in ELECTRICAL_OPTIONAL_FIELDS:
        value = values[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(f"{name} must be null or a finite number")
    positive = scalar_fields - {
        "wing_lift_coefficient_zero",
        "wing_aerodynamic_center_fraction_mac",
        "vtail_aerodynamic_center_fraction_chord",
        "wing_pitch_moment_zero", "wing_pitch_moment_slope",
        "wing_root_le_x_m", "wing_tip_le_x_m", "vtail_root_le_x_m",
        "vtail_tip_le_x_m", "vtail_tip_z_m", "rear_thrust_reserve_fraction",
    }
    for name in positive:
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if values["motor_speed_factor"] > 1:
        raise ValueError("motor_speed_factor must be at most 1")
    if not 0.0 <= values["wing_aerodynamic_center_fraction_mac"] <= 1.0:
        raise ValueError(
            "wing_aerodynamic_center_fraction_mac must be between 0 and 1"
        )
    if not 0.0 < values["vtail_tip_chord_fraction"] <= 1.0:
        raise ValueError("vtail_tip_chord_fraction must be greater than 0 and at most 1")
    if not 0.0 <= values["vtail_aerodynamic_center_fraction_chord"] <= 1.0:
        raise ValueError(
            "vtail_aerodynamic_center_fraction_chord must be between 0 and 1"
        )
    if not 0.0 <= values["rear_thrust_reserve_fraction"] < 1.0:
        raise ValueError("rear_thrust_reserve_fraction must be at least 0 and less than 1")
    if values["oswald_efficiency"] > 1:
        raise ValueError("oswald_efficiency must be at most 1")
    if values["vtail_oswald_efficiency"] > 1:
        raise ValueError("vtail_oswald_efficiency must be at most 1")
    if values["vtail_max_deflection_deg"] > 90.0:
        raise ValueError("vtail_max_deflection_deg must be at most 90 degrees")
    nonnegative_electrical = {
        "motor_resistance_ohm",
        "motor_no_load_current_a",
        "battery_internal_resistance_ohm",
    }
    for name in ELECTRICAL_OPTIONAL_FIELDS:
        value = values[name]
        if value is None:
            continue
        if name == "esc_efficiency":
            if not 0.0 < value <= 1.0:
                raise ValueError("esc_efficiency must be greater than 0 and at most 1")
        elif name in nonnegative_electrical:
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        elif value <= 0.0:
            raise ValueError(f"{name} must be positive")
    cells = values["battery_cells"]
    if isinstance(cells, bool) or not isinstance(cells, int) or cells <= 0:
        raise ValueError("battery_cells must be a positive integer")

    pivot = values["tilt_pivot_xz_m"]
    if not isinstance(pivot, list) or len(pivot) != 2 or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in pivot
    ):
        raise ValueError("tilt_pivot_xz_m must contain two finite coordinates")
    positions = values["motor_positions"]
    if not isinstance(positions, list) or not positions:
        raise ValueError("motor_positions must be a non-empty list")
    for position in positions:
        if (
            not isinstance(position, list)
            or len(position) != 4
            or not isinstance(position[3], bool)
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in position[:3])
        ):
            raise ValueError("each motor position must contain three finite coordinates and a tilt flag")
    values["tilt_pivot_xz_m"] = tuple(pivot)
    values["motor_positions"] = tuple(tuple(position) for position in positions)
    return AircraftParameters(**values)
