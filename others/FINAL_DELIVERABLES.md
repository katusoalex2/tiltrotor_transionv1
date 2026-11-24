# Final Deliverables

## Project Summary

Offline 6-DOF simulation, control design, and validation tools for a V-tail
tilt-rotor UAV transitioning from hover (0 deg tilt) to cruise (90 deg tilt),
including wind-gust analysis.

**Safety status:** This is a simulation and development project, not a flight-ready
autopilot. It has not been integrated with PX4, validated on target hardware, or
cleared for flight. Do not copy its controller code or profile exports to a
Pixhawk as flight software or parameters.

The V-tail model uses one shared longitudinal moment arm. It does not model
individual panel moment arms or V-tail roll moment; its lateral stability
results are therefore incomplete and must not be used to establish flight safety.

For timed AUTO missions the dashboard reports the requested transition targets
against recorded samples. The pitch check starts 3 s after transition entry;
angle of attack is assessed only at airspeeds of at least 15 m/s, a requested
analysis threshold rather than a measured stall speed. The altitude-loss band is
0.5-1.234 m. A transition that does not reach approximately 90 deg by the
scheduled end cannot pass. These checks do not change the controller or establish
flight safety, and the static margin and gust recovery remain unverified.

The dashboard's optional **Show reference-only transition animation** displays
the default timed 0–90 degree tilt reference using the active profile geometry.
It is a stationary schematic, not simulated motion or flight evidence; it does
not enforce actuator or aerodynamic limits and is excluded from simulation
reports. Physical simulation results and warnings remain separate.

## Additional interactive transition analyses

Use the **Additional interactive transition analyses** section in Phase 5, or
the jump link near the top of the dashboard. The selector provides:

- Cosine and linear reference schedules, tilt-rate limits, and both angle
  conventions. These controls do not change the main simulation.
- Fixed-hover-speed force and acceleration sweeps using the current dashboard
  aircraft; the sweeps exclude rotor wash and tail forces and are not trim.
- An analytical tilt-speed screen with either tilt convention. Its separate
  maximum-thrust checks do not establish simultaneous trim or flight safety.
- An on-demand reduced longitudinal optimization. This uses its own fixed
  5 kg, 1.5 m², 15 s scenario, not the dashboard aircraft; it omits moments,
  rotor wash, gusts and electrical limits.
- An on-demand exploratory PID diagnostic with altitude-gain and gust inputs.
  It uses the illustrative KADA profile and an unverified 3000 rad/s motor
  limit, not the dashboard aircraft or PID controls. Results stop at the pitch
  or ground limit, and changing inputs requires a new run.

The additional charts are interactive but are not included in the downloadable
simulation report. They are simulation studies, not validated flight controls.

## Test Status

Latest local run: `python -m pytest -q -p no:cacheprovider` — **121 passed**.

## Deliverables Inventory

### Physics Engine (`src/dynamics/`)
- `state.py` ? AircraftState dataclass (pos, vel, att, omega, tilt, lift, drag)
- `tiltrotor.py` ? 6-DOF Newton-Euler dynamics with rotor + aero + V-tail coupling
- `rotor.py` ? Physical motor model: T = k*omega^2 (PDF Page 9)
- `kinematics.py` ? 3D rotation matrices (PDF Pages 5-8) + Euler kinematic equations

### Aerodynamics (`src/aero/`)
- `aerodynamics.py` ? Wing lift/drag with stall protection + tail coupling
- `vtail.py` ? V-tail symmetric/antisymmetric deflection ? pitch/yaw moments

### Control (`src/control/`)
- `pid.py` ? Robust PID with anti-windup + derivative filtering
- `cascade.py` ? Cascade architecture (altitude/speed ? pitch ? V-tail)
- `scheduler.py` ? Gain scheduling (hover ? cruise interpolation)
- `trajectory.py` ? ReferenceTrajectory (loads corridor CSV) + LinearReference
- `indi.py` ? INDI controller (baseline comparison)
- `indi_dob.py` ? INDI + Disturbance Observer augmentation
- `codegen_controller.py` ? CasADi symbolic controller for C codegen

### Wind Model (`src/wind/`)
- `gust.py` ? Dryden turbulence (1/f^2 PSD) + ConstantGust

### Simulation (`src/simulation/`)
- `runner.py` ? Simulation orchestrator
- `logger.py` ? State history logger

### Configuration (`configs/`)
- `robust_corridor.csv` ? IPOPT-optimized transition tilt schedule

### Scripts (`scripts/`)
- `generate_c.py` ? CasADi CodeGenerator ? C source
- `master_simulation.py` ? Full system demo (PID + gusts + corridor)
- `monte_carlo_gusts.py` ? 200-run wind gust study
- `statistical_comparison.py` ? PID vs INDI deep statistical analysis
- `compare_indi_dob.py` ? 3-way comparison (PID vs INDI vs INDI+DOB)
- `finalize.py` ? Project finalization (report + plot)
- `verify_c_parity_nogcc.py` ? C?Python parity proof (no gcc required)

### Tests (`tests/`)
- Automated tests are in `tests/`; see the test command above for current status.

### Generated C Code (`build/`)
- `pid_controller.c` ? Generated controller code for offline parity testing only;
  it is not PX4 firmware and is not flight-ready
- `pid_controller.h` ? C header

## Controller Selection Decision

Reported offline Monte Carlo analysis (100 runs each, gust sigma 0.5-5.5 m/s;
these results are not flight evidence):

| Controller | Crashes | Mean Altitude Loss |
|------------|---------|-------------------|
| **PID**    | 1/100   | 11.51 m           |
| INDI       | 23/100  | 31.38 m           |
| INDI + DOB | 21/100  | 30.89 m           |

**Selected: PID Cascade**

Reason: INDI's fixed-B-matrix assumption is incompatible with this aircraft's
large variation in control effectiveness across the transition envelope.
DOB augmentation reduced crashes by only 2% (not statistically significant).

## C Code Parity Verification

The recorded parity check reported **0.00e+00 error on its test vectors** between
the CasADi-generated controller and the manual NumPy implementation. This checks
implementation parity only; it does not establish closed-loop safety, PX4
compatibility, timing behavior on a flight controller, or flight readiness.
The C code at `build/pid_controller.c` is not integrated into PX4.

To compile with gcc (optional ? install MinGW from winlibs.com):
    python scripts/verify_c_parity.py

## Key Results

- Reported simulation-only max altitude loss during a 3.5 m/s gust transition: **12.90 m**
- Max pitch excursion: **13.65 deg**
- Peak gust survived: **8.66 m/s**
- Robust corridor tilt rate cap: **85% of actuator limit**

## PX4 and flight-development status

- [x] Dashboard can export a profile-based PX4 SITL mapping scaffold.
- [x] The mapping is explicitly marked non-flashable, assigns no physical outputs,
  and generates no PX4 parameter file or actuator commands.
- [ ] Select and verify a specific PX4 release, flight-controller board, airframe
  support, and measured actuator/output configuration.
- [ ] Validate the mapping in PX4 SITL, then complete hardware-specific bench and
  HITL checks with propellers removed where applicable.
- [ ] Independent safety review and an authorized, staged flight-test program.

The current profile is illustrative, and its PX4 release and controller board are
unknown. The mapping download is **not** a Pixhawk upload file. Simulation,
Monte Carlo results, and C/Python parity checks do not replace aircraft-specific
system identification, hardware validation, or flight-safety review.

## Downloading a simulation report

In the dashboard sidebar, open **Simulation report** and select **Generate full
simulation report**. This recomputes the constrained corridor and stability map
and runs the configured Monte Carlo trials, then offers Word and PDF downloads.
The files contain the displayed graphs, results, warnings, and active profile.
Generation can take several minutes. Install the Python requirements and make
Chrome or Chromium available on the machine running Streamlit so Plotly graphs
can be rendered. If settings change, generate a fresh report.

The transition-time plots compare recorded samples with mission targets; those
targets are not verified altitude, speed, or tilt safety boundaries.
The recorded tilt-versus-true-airspeed curve uses transition samples in time
order. It can reverse direction and ends when the simulation stops; it is not
a verified safe corridor.

The corridor trim screen follows the independent front/rear rotor-pair idea in
Zhao et al., *Development of Multimode Flight Transition Strategy for Tilt-Rotor
VTOL UAVs*, Drones 2023, 7, 580, https://doi.org/10.3390/drones7090580,
sections 3.1–3.2. This project uses the opposite tilt convention (0 degrees at
hover), and solves steady level trim rather than the paper's dynamic trajectory
optimization. Pair speeds propagate into stability, electrical, and average
motor-slew screens. Simplified unmeasured aerodynamics and propulsion, static
sign checks, and missing actuator/electrical inputs limit the results; they do
not demonstrate a feasible or safe flight transition.
Select **Compute constrained corridor** to display the filtered grid. Gray
crosses indicate trim-only points that have not passed all known filters;
purple and green points represent the more restrictive filter results.
The corridor has true airspeed on the horizontal axis and nacelle tilt on the
vertical axis. The recorded transition histories show airspeed, pitch, tilt,
angle of attack, motor commands, and V-tail commands. The 5–12 degree
angle-of-attack band is a target, assessed only at true airspeed of at least
15 m/s; unassessed or out-of-range values are not clipped. These plots do not
reproduce the paper's optimized trajectories or demonstrate flight safety.

--
