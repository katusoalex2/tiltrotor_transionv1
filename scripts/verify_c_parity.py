"""Verify CasADi-generated C code matches Python CasADi output."""

import sys
import os
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.control.codegen_controller import ctrl_fn


def run_c_harness():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, 'build')

    compile_cmd = [
        'gcc',
        os.path.join(build_dir, 'pid_controller.c'),
        os.path.join(project_root, 'scripts', 'c_parity_harness.c'),
        '-I', build_dir,
        '-lm',
        '-o', os.path.join(build_dir, 'c_parity_test'),
    ]
    print('Compiling C harness...')
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError('Compilation failed:\n' + result.stderr)

    print('Running C binary...')
    result = subprocess.run([os.path.join(build_dir, 'c_parity_test')],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError('C binary failed:\n' + result.stderr)

    outputs = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.split()
        if parts[0] == 'TEST1':
            outputs['test1'] = np.array([float(x) for x in parts[1:]])
        elif parts[0] == 'TEST2':
            outputs['test2'] = np.array([float(x) for x in parts[1:]])
    return outputs


def run_python(state, ref, integral):
    omega, vtail, integral_new = ctrl_fn(state, ref, integral)
    return np.array([float(omega), float(vtail),
                     float(integral_new[0]), float(integral_new[1])])


def main():
    print('=' * 70)
    print('C-PYTHON PARITY VERIFICATION')
    print('=' * 70)

    c_outputs = run_c_harness()

    state1 = np.array([0.0, 0.0, -100.0, 10.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.1, 0.0])
    ref1 = np.array([100.0, 12.0, 0.5])
    integral1 = np.array([0.0, 0.0])
    py1 = run_python(state1, ref1, integral1)

    state2 = np.array([0.0, 0.0, -95.0, 8.0, 2.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.2, 0.0])
    ref2 = np.array([100.0, 10.0, 0.3])
    integral2 = np.array([0.1, -0.05])
    py2 = run_python(state2, ref2, integral2)

    print()
    print(f'{"Test":<10} {"Src":<8} {"omega":>14} {"vtail":>14} {"int_alt":>14} {"int_pitch":>14}')
    print('-' * 80)
    for name, py_vals, c_vals in [('Test 1', py1, c_outputs['test1']),
                                   ('Test 2', py2, c_outputs['test2'])]:
        print(f'{name:<10} {"Py":<8} {py_vals[0]:>14.8f} {py_vals[1]:>14.8f} {py_vals[2]:>14.8f} {py_vals[3]:>14.8f}')
        print(f'{"":<10} {"C":<8} {c_vals[0]:>14.8f} {c_vals[1]:>14.8f} {c_vals[2]:>14.8f} {c_vals[3]:>14.8f}')
        print(f'{"":<10} {"Diff":<8} {abs(py_vals[0]-c_vals[0]):>14.2e} {abs(py_vals[1]-c_vals[1]):>14.2e} {abs(py_vals[2]-c_vals[2]):>14.2e} {abs(py_vals[3]-c_vals[3]):>14.2e}')
        print()

    all_close = all(np.allclose(py, c, atol=1e-9)
                    for py, c in [(py1, c_outputs['test1']), (py2, c_outputs['test2'])])
    print('=' * 70)
    if all_close:
        print('RESULT: PASS  (all differences < 1e-9)')
        print('C code matches Python bit-for-bit.')
    else:
        print('RESULT: FAIL  (differences > 1e-9)')
    print('=' * 70)


if __name__ == '__main__':
    main()
