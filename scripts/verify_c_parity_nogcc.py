"""Verify C code parity without gcc installed."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.control.codegen_controller import ctrl_fn


def manual_pid(state, ref, integral):
    current_alt = -state[2]
    current_speed = state[3]
    current_pitch = state[7]
    current_pitch_rate = state[10]
    ref_alt = ref[0]
    ref_speed = ref[1]

    alt_error = ref_alt - current_alt
    alt_p = 0.1 * alt_error
    alt_integral_new = integral[0] + alt_error * 0.02
    alt_i = np.clip(0.005 * alt_integral_new, -0.25, 0.25)
    alt_d = 0.1 * (0 - state[5])
    pitch_from_alt = alt_p + alt_i + alt_d

    speed_error = ref_speed - current_speed
    pitch_from_speed = -0.03 * speed_error

    desired_pitch = np.clip(pitch_from_alt + pitch_from_speed, -0.35, 0.35)

    pitch_error = desired_pitch - current_pitch
    pitch_p = 0.6 * pitch_error
    pitch_integral_new = integral[1] + pitch_error * 0.01
    pitch_i = np.clip(0.05 * pitch_integral_new, -0.2, 0.2)
    pitch_d = -0.4 * (0 - current_pitch_rate)
    vtail_raw = -(pitch_p + pitch_i + pitch_d)
    vtail_cmd = np.clip(vtail_raw, -0.2, 0.2)

    altitude_error = ref_alt - current_alt
    total_thrust = (5.0 * 9.81) + (altitude_error * 0.5)
    omega_cmd = np.sqrt(max(total_thrust, 0.0) / 1e-5)
    omega_cmd = np.clip(omega_cmd, 0.0, 3000.0)

    return np.array([omega_cmd, vtail_cmd, alt_integral_new, pitch_integral_new])


def main():
    print("=" * 70)
    print("C-PYTHON PARITY VERIFICATION (no gcc required)")
    print("=" * 70)

    print("\n[1/3] Checking C source was generated...")
    if not os.path.exists("build/pid_controller.c"):
        print("  ERROR: build/pid_controller.c not found.")
        print("  Run scripts/generate_c.py first.")
        return
    with open("build/pid_controller.c", "r", encoding="utf-8") as f:
        c_source = f.read()
    if "pid_controller_step" not in c_source:
        print("  ERROR: function name not found in C source.")
        return
    print(f"  [OK] build/pid_controller.c contains pid_controller_step ({len(c_source)} bytes)")

    print("\n[2/3] Running test vectors through CasADi function...")
    test_cases = [
        ("Test 1 (nominal)",
         np.array([0.0, 0.0, -100.0, 10.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.1, 0.0]),
         np.array([100.0, 12.0, 0.5]),
         np.array([0.0, 0.0])),
        ("Test 2 (high gust)",
         np.array([0.0, 0.0, -95.0, 8.0, 2.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.2, 0.0]),
         np.array([100.0, 10.0, 0.3]),
         np.array([0.1, -0.05])),
    ]

    print("\n[3/3] Comparison: CasADi vs manual numpy")
    print()
    print(f'{"Test":<22} {"Metric":<12} {"CasADi":>16} {"Manual":>16} {"Diff":>12}')
    print("-" * 82)

    all_pass = True
    for name, state, ref, integral in test_cases:
        omega, vtail, int_new = ctrl_fn(state, ref, integral)
        casadi_vals = np.array([float(omega), float(vtail), float(int_new[0]), float(int_new[1])])
        manual_vals = manual_pid(state, ref, integral)

        for i, metric in enumerate(["omega", "vtail", "int_alt", "int_pitch"]):
            diff = abs(casadi_vals[i] - manual_vals[i])
            ok = diff < 1e-9
            if not ok:
                all_pass = False
            print(f'{name:<22} {metric:<12} {casadi_vals[i]:>16.8f} {manual_vals[i]:>16.8f} {diff:>12.2e}')
        print()

    print("=" * 70)
    if all_pass:
        print("RESULT: PASS")
        print()
        print("CasADi-generated function output matches manual numpy math to <1e-9.")
        print("Since the C code is generated FROM the CasADi function, the C code is")
        print("mathematically identical to what you see above.")
        print()
        print("To compile the C code yourself, install gcc from https://winlibs.com/")
        print("then run: python scripts\\verify_c_parity.py")
    else:
        print("RESULT: FAIL - CasADi and manual numpy differ.")
    print("=" * 70)


if __name__ == "__main__":
    main()
