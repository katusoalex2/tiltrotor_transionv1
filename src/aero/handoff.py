"""Lift handoff and control authority during transition."""
import numpy as np

RHO = 1.225
GRAVITY = 9.81


def compute_lift_breakdown(state, tilt, V_T, alpha, rotor_thrust, wing_area, 
                           mass, CL_alpha=4.0, CL_0=0.1, prop_radius=0.15):
    """Returns the full lift breakdown at this instant.
    
    Args:
        state: AircraftState (for altitude, vel)
        tilt: Rotor tilt angle (rad)
        V_T: True airspeed (m/s)
        alpha: Angle of Attack (rad)
        rotor_thrust: Total thrust from all 4 rotors (N)
        wing_area: Wing planform area (m^2)
        mass: Total mass (kg)
        CL_alpha: Lift curve slope (per rad)
        CL_0: Zero-alpha lift coefficient
        prop_radius: Rotor radius (m)
    
    Returns:
        dict with rotor_thrust, wing_lift, wash_lift, total_vertical, mg, 
        surplus, handoff_ratio
    """
    # 1. Rotor Vertical Thrust
    rotor_vertical = rotor_thrust * np.cos(tilt)
    
    # 2. Wing Lift (Free Stream)
    if V_T > 0.5:
        alpha_clamped = np.clip(alpha, -np.deg2rad(15), np.deg2rad(15))
        CL = CL_0 + CL_alpha * alpha_clamped
    else:
        CL = 0.0
    q_bar = 0.5 * RHO * (V_T ** 2)
    wing_lift = q_bar * wing_area * CL
    
    # 3. Rotor Wash Lift (Slipstream on wing)
    # Induced velocity
    thrust_per_rotor = rotor_thrust / 4.0
    A_rotor = np.pi * (prop_radius ** 2)
    v_i = np.sqrt(max(thrust_per_rotor, 0) / (2 * RHO * A_rotor))
    
    # The wash hits the wing at a downward angle (depends on tilt)
    wash_velocity = v_i * np.cos(tilt)  # Downward component
    q_bar_wash = 0.5 * RHO * (wash_velocity ** 2)
    alpha_wash = np.deg2rad(5)  # Approximate wash AoA on wing
    CL_wash = CL_0 + CL_alpha * alpha_wash
    wash_lift = q_bar_wash * wing_area * CL_wash
    
    # 4. Total Vertical Force
    total_vertical = rotor_vertical + wing_lift + wash_lift
    
    # 5. Weight
    mg = mass * GRAVITY
    
    # 6. Surplus (positive = climbing, negative = dropping)
    surplus = total_vertical - mg
    
    # 7. Handoff Ratio (0 = rotor, 1 = wing)
    if total_vertical > 0.1:
        handoff_ratio = wing_lift / total_vertical
    else:
        handoff_ratio = 0.0
    
    return {
        "rotor_thrust": rotor_vertical,
        "wing_lift": wing_lift,
        "wash_lift": wash_lift,
        "total_vertical": total_vertical,
        "weight": mg,
        "surplus": surplus,
        "handoff_ratio": handoff_ratio,
    }


def control_authority(tilt, V_T, rotor_thrust_per_rotor, arm=0.4, 
                     wing_area=0.6, CL_alpha=4.0):
    """Returns pitch and roll control authority from rotor vs V-tail at this tilt."""
    # Rotor pitch authority (differential RPM)
    rotor_pitch = (rotor_thrust_per_rotor * np.cos(tilt)) * arm
    
    # V-Tail pitch authority (aerodynamic)
    q_bar = 0.5 * RHO * (V_T ** 2)
    vtail_pitch = q_bar * wing_area * 0.1 * CL_alpha
    
    # Normalize to 0-1
    total = rotor_pitch + vtail_pitch
    if total > 0.01:
        rotor_frac = rotor_pitch / total
        vtail_frac = vtail_pitch / total
    else:
        rotor_frac = 1.0
        vtail_frac = 0.0
    
    return {
        "rotor_pitch_authority": rotor_pitch,
        "vtail_pitch_authority": vtail_pitch,
        "rotor_authority_frac": rotor_frac,
        "vtail_authority_frac": vtail_frac,
    }


def control_blend(tilt):
    """Returns a blending factor: 0 = rotor-only, 1 = V-tail-only."""
    # Smooth S-curve between 30? and 60? tilt
    tilt_deg = np.rad2deg(tilt)
    if tilt_deg < 30:
        return 0.0
    elif tilt_deg > 60:
        return 1.0
    else:
        # S-curve in the 30-60 range
        t = (tilt_deg - 30) / 30.0
        return 3 * t**2 - 2 * t**3  # Smoothstep
