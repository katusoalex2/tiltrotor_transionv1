# Tiltrotor transition project: structure and flowcharts

These diagrams describe the code as implemented. They show simulation and
analysis workflows, not a flight-ready control system.

## 1. Overall project structure

```mermaid
flowchart TB
    U(["User"]) --> DASH["Dashboard<br/>scripts/dashboard.py"]
    U --> STAND["Standalone scripts<br/>scripts/"]
    U --> TEST["Tests<br/>tests/"]

    subgraph INPUTS["Inputs and configuration"]
        PROFILE["Aircraft profile<br/>YAML / JSON / Excel"]
        PARAM["Geometry, mass, inertia,<br/>propulsion and PID gains"]
        MISSION["Mission, tilt reference<br/>and operating limits"]
        WIND["Wind and gust settings"]
    end

    PROFILE --> DASH
    DASH --> PARAM
    DASH --> MISSION
    DASH --> WIND

    subgraph CORE["Main dashboard simulation"]
        CONTROL["Guidance and control<br/>src/control/"]
        ALLOC["Control allocation<br/>src/allocation/"]
        PHYSICS["6-DOF aircraft dynamics<br/>src/dynamics/"]
        AERO["Wing and V-tail aerodynamics<br/>src/aero/"]
        DIST["Wind models<br/>src/wind/"]
        STATE["AircraftState<br/>position, velocity, attitude, rates, tilt"]
        CONTROL --> ALLOC
        ALLOC -->|"rotor, tilt and tail commands"| PHYSICS
        AERO -->|"forces and moments"| PHYSICS
        DIST -->|"wind"| CONTROL
        DIST -->|"wind"| PHYSICS
        PHYSICS -->|"next state"| STATE
        STATE -->|"feedback"| CONTROL
    end

    PARAM --> CONTROL
    PARAM --> PHYSICS
    MISSION --> CONTROL
    WIND --> DIST
    DASH -->|"run_simulation"| CORE
    CORE --> LOG["Recorded samples and diagnostics"]
    LOG --> RESULTS["Dashboard graphs, targets,<br/>corridor and stability assessments"]
    RESULTS --> REPORT["Optional Word / PDF report"]

    subgraph STUDIES["Additional analyses: separate workflows"]
        SCHED["Cosine / linear tilt references"]
        FORCES["Fixed-speed force sweeps"]
        ENVELOPE["Analytical envelope,<br/>trim and constrained corridor"]
        OPT["Reduced longitudinal and<br/>multi-corridor optimization"]
        MC["Monte Carlo, gust and<br/>controller comparisons"]
        PID["Exploratory four-rotor PID diagnostic"]
    end

    DASH -. "interactive studies" .-> STUDIES
    STAND --> STUDIES
    PARAM -. "selected studies" .-> STUDIES
    STUDIES --> CURVES["Study-specific curves and data"]
    CURVES --> U
    RESULTS --> U
    REPORT --> U
    DASH -.-> PX4["Optional PX4 SITL mapping draft<br/>not actuator commands or firmware"]
    TEST -. "checks selected components" .-> CORE
    TEST -. "checks selected studies" .-> STUDIES
```

Solid arrows show a data or control flow; dotted arrows indicate optional use
or testing. The additional studies are not all part of one simulation run.

## 2. Dashboard closed-loop simulation

```mermaid
flowchart TD
    START(["Start dashboard"]) --> LOAD["Load aircraft profile and dashboard settings"]
    LOAD --> VALID{"Profile compatible<br/>with simulator?"}
    VALID -- "No" --> ERROR["Display compatibility errors"] --> END1(["Stop"])
    VALID -- "Yes" --> INIT["Create state, dynamics, controller,<br/>mission and optional wind / protection"]
    INIT --> LOOP["Begin simulation step"]

    LOOP --> REF["Get reference: timed mission,<br/>waypoint, hold or manual"]
    REF --> GUST["Sample wind / gust"]
    GUST --> TILT["Apply tilt-rate limit to candidate tilt"]
    TILT --> OBS["Update transition observer<br/>from simulated state and wind"]
    OBS --> PROT["Optionally limit reference<br/>using envelope protection"]
    PROT --> GUIDE["Guidance: altitude / speed PID,<br/>desired thrust and attitude"]
    GUIDE --> ATT["Attitude PID: desired body rates"]
    ATT --> RATE["Scheduled rate PID:<br/>desired body moments"]
    RATE --> ALLOC["Allocate bounded rotor and V-tail commands<br/>with pitch-priority handling if needed"]
    ALLOC --> FAIL["Apply optional actuator failure<br/>and final tilt-rate limit"]
    FAIL --> DYN["6-DOF dynamics step:<br/>rotor + wing + V-tail forces and moments"]
    DYN --> NEW["Integrate position, velocity,<br/>attitude and body rates"]
    NEW --> LIMIT{"Absolute pitch<br/>exceeds 60 degrees?"}
    LIMIT -- "Yes" --> STOP["Stop run; show reason<br/>and prior recorded samples"]
    LIMIT -- "No" --> RECORD["Record states, forces, commands,<br/>wind and controller diagnostics"]
    RECORD --> MORE{"Simulation time<br/>remaining?"}
    MORE -- "Yes: next step" --> LOOP
    MORE -- "No" --> DONE["Display recorded graphs, metrics<br/>and optional analyses"]
    STOP --> DONE
    DONE --> EXPORT["Optionally generate a report"]
    EXPORT --> END2(["End"])
```

In the current dashboard, the **60-degree pitch check happens before the new
sample is logged**. The chart therefore ends at the last recorded sample. The
dashboard controller receives the simulated state and wind directly; the
filtered transition observer is used for monitoring and optional protection,
not as a full sensor-estimation system. The diagram does not imply a general
ground-impact termination check in this dashboard loop.

## 3. Main object-oriented relationships

```mermaid
classDiagram
    class AircraftState {
        pos
        vel
        att
        omega
        tilt
        copy()
    }
    class TiltrotorDynamics {
        mass
        step(state, commands)
    }
    class RotorModel {
        k_thrust
        max_omega
        compute_thrust()
    }
    class AeroModel {
        wing_area
        compute_forces_and_moments()
        compute_tail_forces()
    }
    class VtailModel {
        area
        arm_length
        compute_forces_and_moments()
    }
    class MassProperties {
        motor_positions
        calculate_inertia()
    }
    class ControlLawArchitecture {
        compute(state, reference, dt, wind)
    }
    class PIDController {
        kp
        ki
        kd
        compute()
    }
    class GainScheduler {
        get_gains()
    }
    class ControlAllocator {
        allocate()
    }
    class TransitionStateObserver {
        update()
    }
    class EnvelopeProtection {
        apply()
    }
    class MissionProfile {
        at(time)
    }
    class MissionManager {
        update(state)
    }
    class WindField {
        step(dt)
    }
    class DrydenGust
    class DiscreteGust

    TiltrotorDynamics --> RotorModel : uses
    TiltrotorDynamics --> AeroModel : uses
    TiltrotorDynamics --> MassProperties : optional
    TiltrotorDynamics ..> AircraftState : advances
    AeroModel --> VtailModel : uses
    ControlLawArchitecture --> PIDController : contains several
    ControlLawArchitecture --> GainScheduler : uses
    ControlLawArchitecture --> ControlAllocator : uses
    ControlLawArchitecture --> TransitionStateObserver : uses
    ControlLawArchitecture --> EnvelopeProtection : optional
    ControlLawArchitecture ..> AircraftState : reads
    EnvelopeProtection ..> TransitionStateObserver : consumes estimate
    MissionManager ..> AircraftState : reads for waypoints
    WindField <|-- DrydenGust
    WindField <|-- DiscreteGust
```

This is a **selected class relationship diagram**, not an exhaustive UML class
inventory. The project is **mixed-paradigm**: it uses classes and objects for
state, models and controllers, while many analysis routines and scripts are
procedural. For example, `WindField` has subclasses, whereas the controller
primarily combines other objects rather than inheriting from them.

## 4. Separate study paths and outputs

| Study | Entry point | What it produces / important distinction |
|---|---|---|
| Dashboard simulation | [dashboard.py](scripts/dashboard.py) | Interactive 6-DOF state, force, control, mission and gust plots. |
| Additional dashboard studies | [dashboard_transition_analyses.py](scripts/dashboard_transition_analyses.py) | Interactive references, sweeps, analytical envelope, reduced optimizer and PID diagnostic. They do not all use the active dashboard model. |
| Force sweeps | [plot_force_model_curves.py](scripts/plot_force_model_curves.py) | Fixed-hover-speed, level-attitude force and acceleration sweeps; no rotor wash in the sweep; not a closed-loop trajectory. |
| Analytical tilt-speed envelope | [plot_tilt_angle_vs_velocity.py](scripts/plot_tilt_angle_vs_velocity.py) and [transition.py](src/dynamics/transition.py) | Approximate separate force-bound screening, not verified simultaneous trim or a safe corridor. |
| Reduced longitudinal optimization | [optimize_longitudinal_transition.py](scripts/optimize_longitudinal_transition.py) | Separate 2D point-mass trajectory and force curves; not full 6-DOF or gust validation. |
| Multiple corridors | [multi_corridor.py](scripts/multi_corridor.py) | Separate simplified vertical-model objectives: fast, safe and smooth labels; not certified corridors. |
| Master simulation | [master_simulation.py](scripts/master_simulation.py) | Separate PID cascade and CSV reference workflow with its own gust handling and single equivalent rotor command; not the dashboard control architecture. |
| Exploratory PID diagnostic | [plot_master_pid_diagnostics.py](scripts/plot_master_pid_diagnostics.py) | Separate four-rotor diagnostic using an illustrative profile and unverified motor-speed assumption; stops at its pitch or ground limit. |
| Dashboard Monte Carlo | [monte_carlo.py](src/simulation/monte_carlo.py) | Seeded scenarios and optional actuator failures run through a supplied simulation callback. |
| Standalone comparisons | [monte_carlo_gusts.py](scripts/monte_carlo_gusts.py), [compare_indi_dob.py](scripts/compare_indi_dob.py) | Separate controller-comparison scripts; their configurations and results are not interchangeable with dashboard Monte Carlo. |

The master simulation loads [robust_corridor.csv](configs/robust_corridor.csv);
that CSV is not automatically produced or consumed by the dashboard's timed
mission. Likewise, the reduced optimizer's curves do not automatically become
the main dashboard simulation's references.

## 5. Reading the diagrams

- **Reference:** desired altitude, speed and nacelle tilt, not the actual state.
- **Controller:** computes thrust, attitude and body-moment requests from the
  reference and simulated state.
- **Allocator:** maps requests to bounded motor and V-tail commands.
- **Dynamics:** computes forces and moments and integrates the next aircraft
  state; the feedback loop then repeats.
- **NED / FRD:** position and velocity use north-east-down; body forces and
  body rates use forward-right-down. Project nacelle convention is 0 degrees
  in hover and 90 degrees in cruise.
- **Safety:** model assumptions, control authority, sensor effects, actuator
  dynamics and aircraft-specific parameters remain unvalidated. Neither a
  successful plot nor an optimization establishes flight safety.
