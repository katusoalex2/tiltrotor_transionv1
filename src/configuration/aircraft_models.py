"""Construct simulation models from a loaded aircraft configuration."""

import numpy as np

from src.aero.aerodynamics import AeroModel
from src.aero.vtail import VtailModel
from src.configuration.aircraft_parameters import AircraftParameters
from src.control.cascade import CascadeController
from src.dynamics.rotor import RotorModel
from src.dynamics.tiltrotor import TiltrotorDynamics


def build_aircraft_dynamics(aircraft: AircraftParameters) -> TiltrotorDynamics:
    rotor = RotorModel(
        aircraft.k_thrust, aircraft.max_omega,
        prop_mass_kg=aircraft.prop_mass_kg, prop_radius_m=aircraft.prop_diameter_m / 2,
    )
    tail = VtailModel(
        aircraft.vtail_area, aircraft.vtail_arm_length,
        dihedral_deg=aircraft.vtail_dihedral_deg, rho=aircraft.air_density,
    )
    tail.CL_alpha_tail = aircraft.vtail_lift_slope
    tail.CD_0_tail = aircraft.vtail_drag_coeff
    tail.AR_tail = aircraft.vtail_aspect_ratio
    tail.oswald_efficiency = aircraft.vtail_oswald_efficiency
    tail.max_lift_coefficient = aircraft.vtail_max_lift_coefficient
    aero = AeroModel(
        aircraft.wing_area, tail, rho=aircraft.air_density,
        num_rotors=len(aircraft.motor_positions), prop_diameter_m=aircraft.prop_diameter_m,
    )
    aero.CL_0 = aircraft.wing_lift_coefficient_zero
    aero.CL_alpha = aircraft.wing_lift_slope
    aero.CD_0 = aircraft.wing_drag_coeff
    aero.AR = aircraft.wing_aspect_ratio
    aero.oswald_efficiency = aircraft.oswald_efficiency
    aero.stall_angle = np.deg2rad(aircraft.wing_stall_angle_deg)
    aero.chord = aircraft.wing_mean_chord_m
    aero.x_ac = aircraft.wing_aerodynamic_center_x_m
    aero.Cm_0 = aircraft.wing_pitch_moment_zero
    aero.Cm_alpha = aircraft.wing_pitch_moment_slope
    return TiltrotorDynamics(
        aircraft.mass, rotor, aero=aero,
        inertia=np.array([aircraft.inertia_xx, aircraft.inertia_yy, aircraft.inertia_zz]),
        motor_positions=aircraft.motor_positions, tilt_pivot_xz=aircraft.tilt_pivot_xz_m,
        gravity=aircraft.gravity,
    )


def build_collective_controller(aircraft: AircraftParameters) -> CascadeController:
    """Equal-RPM collective controller; not an allocation or trim solution."""
    if len(aircraft.motor_positions) != 4:
        raise ValueError("the simulation step supports exactly four rotors")
    collective_coefficient = len(aircraft.motor_positions) * aircraft.k_thrust
    hover_omega = np.sqrt(aircraft.mass * aircraft.gravity / collective_coefficient)
    return CascadeController(
        aircraft.mass, collective_coefficient, hover_omega,
        max_omega=aircraft.max_omega, gravity=aircraft.gravity,
    )


def step_collective(dynamics, state, command, dt):
    """Apply one collective RPM to all four configured rotors."""
    return dynamics.step(
        state, np.zeros(3), *([float(command["omega"])] * 4),
        command["tilt"], command["delta_left"], command["delta_right"], dt,
    )
