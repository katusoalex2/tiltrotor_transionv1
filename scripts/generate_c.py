"""Generate C code using CodeGenerator (more reliable than Function.generate)."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import casadi as ca
from src.control.codegen_controller import ctrl_fn

os.makedirs('build', exist_ok=True)

# Use CodeGenerator instead of Function.generate
cg = ca.CodeGenerator('pid_controller', {
    'with_header': True,
    'mex': False,
    'main': False,
    'casadi_real': 'double',
    'casadi_int': 'long long int',
})
cg.add(ctrl_fn)
cg.generate('build/')

print('C code generated: build/pid_controller.c')
print('C header generated: build/pid_controller.h')
print(f'Function name in generated C: pid_controller_step')
