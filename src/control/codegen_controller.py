"""CasADi codegen controller using CodeGenerator (safer than Function.generate)."""

import casadi as ca

state_sym = ca.SX.sym('state', 12)
ref_sym = ca.SX.sym('ref', 3)
integral_sym = ca.SX.sym('integral', 2)

current_alt = -state_sym[2]
current_speed = state_sym[3]
current_pitch = state_sym[7]
current_pitch_rate = state_sym[10]

ref_alt = ref_sym[0]
ref_speed = ref_sym[1]

# Altitude PID
alt_error = ref_alt - current_alt
alt_p = 0.1 * alt_error
alt_integral_new = integral_sym[0] + alt_error * 0.02
alt_i = ca.fmin(ca.fmax(0.005 * alt_integral_new, -0.25), 0.25)
alt_d = 0.1 * (0 - state_sym[5])
pitch_from_alt = alt_p + alt_i + alt_d

# Speed PID
speed_error = ref_speed - current_speed
pitch_from_speed = -0.03 * speed_error

desired_pitch = ca.fmin(ca.fmax(pitch_from_alt + pitch_from_speed, -0.35), 0.35)

# Pitch PID
pitch_error = desired_pitch - current_pitch
pitch_p = 0.6 * pitch_error
pitch_integral_new = integral_sym[1] + pitch_error * 0.01
pitch_i = ca.fmin(ca.fmax(0.05 * pitch_integral_new, -0.2), 0.2)
pitch_d = -0.4 * (0 - current_pitch_rate)
vtail_raw = -(pitch_p + pitch_i + pitch_d)
vtail_cmd = ca.fmin(ca.fmax(vtail_raw, -0.2), 0.2)

# Thrust
altitude_error = ref_alt - current_alt
total_thrust = (5.0 * 9.81) + (altitude_error * 0.5)
omega_cmd = ca.sqrt(ca.fmax(total_thrust, 0.0) / 1e-5)
omega_cmd = ca.fmin(ca.fmax(omega_cmd, 0.0), 3000.0)

integral_new = ca.vertcat(alt_integral_new, pitch_integral_new)

ctrl_fn = ca.Function(
    'pid_controller_step',
    [state_sym, ref_sym, integral_sym],
    [omega_cmd, vtail_cmd, integral_new],
    ['state', 'ref', 'integral'],
    ['omega', 'vtail', 'integral_new'],
)
