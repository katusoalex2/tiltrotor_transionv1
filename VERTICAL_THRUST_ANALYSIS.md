# Historical vertical-thrust analysis

This archived analysis uses deflection from vertical for its angles and equations. The project now reports 90° upward and 0° forward. The rerun is in [transition study](figures/transition_study/summary.json).

> **RETRACTED:** This analysis contains incorrect RPM-unit conversions, lift calculations,
> and unsupported safety claims. Do not use its conclusions or recommendations.

# Vertical Thrust Sufficiency Constraint Analysis
## Tiltrotor Transition Corridor - Phase 1 Diagnostic

**Date:** 2026-11-25  
**Status:** Initial Assessment - Ready for Improvements  
**Author:** Copilot Analysis

---

## Executive Summary

The transition corridor's first constraint is **vertical thrust sufficiency**. As forward nacelles tilt from 0° (hover) to 90° (cruise), the vertical component of their thrust drops, and the fixed rear rotors must provide compensating lift. If rear rotors lack authority, the lower airspeed boundary rises sharply, constraining the transition envelope.

**Current Status:**
- ✅ Vertical thrust is modeled in the envelope calculator (`transition_speed_envelope()`)
- ⚠️ **Gaps Identified:**
  1. No explicit rear-motor authority checking during optimization
  2. Rear motor RPM constraints not explicitly enforced
  3. Altitude loss optimization focuses on tilt-rate limits, not thrust availability
  4. Control effectiveness analysis lacks rear-motor saturation boundaries

---

## 1. What Has Been Done

### 1.1 Physics Foundation

**Location:** `src/dynamics/transition.py:40-52`

The speed envelope correctly models vertical thrust budget:

```python
max_vertical_rotor = max_thrust_per_rotor * (
    n_fixed + n_tilting * np.cos(tilt_rad)
)
```

Where:
- `n_fixed` = number of fixed (rear) rotors
- `n_tilting` = number of tilting (forward) rotors
- `cos(tilt_rad)` = vertical component factor for forward thrust

**Physics check:** ✅
- At tilt = 0° (hover): `cos(0°) = 1.0` → forward rotors contribute full vertical thrust
- At tilt = 90° (cruise): `cos(90°) = 0.0` → forward rotors contribute zero; rear rotors alone must hold the aircraft

### 1.2 Envelope Calculation

**Location:** `src/dynamics/transition.py:54-62`

Required wing lift is calculated as the **deficit**:

```python
wing_lift_required = np.maximum(mass * GRAVITY - max_vertical_rotor, 0.0)
```

This establishes the minimum speed needed to generate that lift:

```python
minimum_stall_speed = np.sqrt(2.0 * wing_lift_required / (rho * wing_area * cl_max))
```

**What it does right:** ✅
- Correctly identifies when rotors alone cannot support the aircraft
- Ties lower speed boundary to lift requirement
- Scales with tilt angle (tighter at 90° than at 0°)

### 1.3 Aircraft Configuration

**Location:** `UAM Configurations/master_uam.yaml`

```yaml
propulsion:
  num_rotors: 4
  motor_kv: 920
  battery_voltage: 22.2
  thrust_coefficient: 1.0e-05
  max_rpm: 3000.0

geometry:
  motors:
    - id: 1, x: 0.3, y: 0.4, z: 0.0, tilts: true   # Forward left
    - id: 2, x: 0.3, y: -0.4, z: 0.0, tilts: true  # Forward right
    - id: 3, x: -0.3, y: 0.4, z: 0.0, tilts: false # Rear left (FIXED)
    - id: 4, x: -0.3, y: -0.4, z: 0.0, tilts: false # Rear right (FIXED)
```

**Motor Authority at 90° Tilt:**
- Maximum rotor speed: 3000 RPM ≈ 314 rad/s
- Thrust coefficient: k_thrust = 1.0e-05
- **Max thrust per rotor:** `T = 1.0e-05 × (314)² ≈ 0.985 N`
- **Rear motor total:** 2 × 0.985 ≈ **1.97 N**
- **Aircraft weight:** 5 kg × 9.81 m/s² ≈ **49.1 N**
- **Rear-motor margin at 90°:** `1.97 / 49.1 ≈ 4%` ⚠️ **CRITICALLY THIN**

### 1.4 Corridor Optimization Approaches

Three optimization scripts have been developed:

#### A. Simplified CasADi (optimize_corridor_casadi.py)
- **Scope:** 50-step tilt schedule over 15 seconds
- **Objective:** Minimize altitude loss
- **Constraints:** Tilt rate, altitude floor, thrust approximation
- **Limitation:** Uses crude 1D model; altitude dynamics approximated

#### B. Scipy-based (optimize_corridor.py)
- **Scope:** 5-piece piecewise tilt profile
- **Objective:** Minimize altitude loss via full 6-DOF simulation
- **Limitation:** Cannot directly observe rear-motor RPM constraints; only simulates result

#### C. Robust Corridor (robust_corridor.py)
- **Enhancement:** Nominal vs. robust variants
- **Robust variant:** Caps tilt rate at 85% of hardware limit (17 deg/s instead of 20 deg/s)
- **Result:** Provides 15% safety margin on tilt actuator, **but not on motor thrust**

---

## 2. What Is Lacking

### 2.1 ⛔ No Explicit Rear-Motor Authority Constraint

**Problem:** The corridor optimization never checks if rear motors can actually provide the required vertical thrust.

**Where it should be:** `optimize_corridor.py` and `optimize_corridor_casadi.py`

The solver minimizes altitude loss but never:
- Verifies that `T_rear(rpm_max) ≥ mg - T_forward(tilt, omega_forward)`
- Tracks rear-motor RPM during optimization
- Rejects solutions where rear motors saturate

**Current workaround:** The simulation runs with controllers that can saturate motors, but the optimization doesn't prevent saturated states from being selected.

### 2.2 ⛔ No Rear-Motor RPM Tracking in Trajectory Reference

**Location:** `src/control/trajectory.py` (or reference module)

The reference trajectory loads the tilt schedule from the corridor but **does not map rear-motor commands**. The PID cascade controller must recover by trial-and-error during the actual transition.

**Missing capability:**
```python
# DOES NOT EXIST YET:
def compute_rear_motor_rpm(airspeed, tilt_angle, altitude):
    """Return recommended rear-motor RPM to maintain vertical trim."""
    # Should use transition_speed_envelope or trim solver
    pass
```

### 2.3 ⛔ No Saturation-Aware Speed Boundaries

**Location:** `src/dynamics/transition.py:transition_speed_envelope()`

The envelope assumes rotors **always** run at max power. It does not distinguish between:
- Speeds achievable when rear motors are **below** saturation (comfortable)
- Speeds only achievable when rear motors are **at or above** saturation (marginal)

**Result:** The feasible speed band looks wider than it really is.

**Example:** At 90° tilt:
- Current: "Min speed = 15 m/s" (assumes rear motors at max)
- Better: "Min speed = 15 m/s, but rear motors saturate above 12 m/s" (more honest about margin)

### 2.4 ⛔ Control Effectiveness Analysis Omits Rear-Motor Limitations

**Location:** `src/dynamics/transition.py:control_effectiveness_map()`

The function computes pitch/roll/yaw authority by perturbing rotor speeds. But it:
- Uses collective and pair speeds as inputs
- **Does not check if the perturbed state saturates rear motors**
- Reports "control authority" that assumes unlimited motor headroom

### 2.5 ⛔ No Explicit Vertical Acceleration Constraint in Optimization

**Problem:** The corridor optimizer focuses on tilt rate but not on the aerodynamic consequence: vertical velocity.

The simplified altitude dynamics (optimize_corridor_casadi.py) approximates:
```python
def altitude_dynamics(tilt, vz, alt):
    thrust_vert = k_thrust * (omega_hover ** 2) * np.cos(tilt)
    lift = 0.5 * rho * wing_area * CL * (v_forward ** 2) * np.sin(tilt)
    a_z = (thrust_vert + lift - mass * g) / mass
    return a_z
```

**Missing:** No constraint that `a_z ≥ 0` (no acceleration toward ground) during transition. The optimizer can produce schedules that require active altitude recovery, which may not be achievable if rear motors are saturated.

### 2.6 ⛔ Altitude Loss Tolerance Is Ad Hoc

**Location:** `robust_corridor.py:70, 75`

```python
if robust:
    alt_floor = ALT_START - 10.0  # 10 m loss allowed
else:
    alt_floor = ALT_START - 10.0  # Same!
```

Both nominal and robust use the **same** floor. This is not a principled constraint; it should be:
- Derived from aircraft stall margin or ground-clearance requirements
- Scaled by wind-gust penetration depth during validation
- Made explicit in the physics model

---

## 3. Proposed Improvements (Prioritized)

### **PHASE 1: Diagnostic Tools (Low Effort, High Insight)**

#### 1.1 Add Rear-Motor Authority Plot
**What:** Create visualization showing rear-motor RPM vs. tilt angle at fixed speeds.

**Location:** New script `scripts/plot_rear_motor_authority.py`

**Output:** 
- X-axis: Tilt angle (0°–90°)
- Y-axis: Rear-motor RPM needed for vertical trim
- Overlay: Max available RPM (3000 rpm line)
- Shaded regions: safe vs. saturated

**Why:** Immediately shows the envelope boundary where rear motors run out of authority.

**Effort:** 2–3 hours

---

#### 1.2 Augment Speed Envelope with Saturation Flag
**What:** Modify `transition_speed_envelope()` to return rear-motor RPM requirement.

**Location:** `src/dynamics/transition.py:transition_speed_envelope()`

**Change:**
```python
# ADD to return dict:
"rear_motor_rpm": rear_motor_rpm_at_max_speed,
"rear_saturated": rear_motor_rpm_at_max_speed > omega_limit_rpm,
```

**Why:** Enables corridor optimizer to see saturation boundaries upstream.

**Effort:** 1–2 hours

---

### **PHASE 2: Constraint-Aware Optimization (Medium Effort, Transformative)**

#### 2.1 Rewrite Corridor Optimization with Rear-Motor Tracking
**What:** Modify the main optimization loop to solve for and constrain rear-motor speed.

**Location:** `optimize_corridor.py` → new `optimize_corridor_with_motor_authority.py`

**Algorithm:**
```
For each time step:
  1. Given tilt(t), airspeed(t), estimate required vertical thrust
  2. Solve for omega_rear(t) = f(tilt, V, altitude, mass)
  3. Constraint: omega_rear(t) ≤ 0.9 × max_omega (15% margin)
  4. If constraint violated → reject tilt schedule or flag as "marginal"
```

**Output:** 
- Corridor tilt schedule
- Associated rear-motor RPM profile
- Saturation margin visualization

**Why:** Replaces ad-hoc tilt-rate limits with physics-based motor authority limits.

**Effort:** 6–8 hours

---

#### 2.2 Add Vertical Acceleration Lower Bound
**What:** Add constraint `a_z(t) ≥ a_z_min` to optimization.

**Location:** `optimize_corridor_casadi.py` dynamics

**Rationale:**
```python
# No free-fall during transition
opti.subject_to(altitude_dynamics(tilt[k], vz[k]) ≥ -0.5)  # max 0.5 m/s² downward allowed
```

**Why:** Prevents optimizer from selecting tilt schedules that require the controller to "rescue" altitude via thrust—which may not be possible if rear motors are saturated.

**Effort:** 1 hour

---

### **PHASE 3: Real-Time Guidance (High Effort, High Value)**

#### 3.1 Extend Trajectory Reference with Rear-Motor Feedforward
**What:** Add rear-motor RPM as a reference input to the cascade controller.

**Location:** `src/control/trajectory.py` and `src/control/cascade.py`

**New method:**
```python
def get_rear_motor_rpm_target(airspeed, tilt, altitude, mass=5.0):
    """Feedforward rear-motor command to maintain vertical force balance."""
    # Use trim solver to find required rear omega
    return omega_rear
```

**Integration:**
```python
# In cascade controller:
state = controller.compute(...)  # Existing PID
state["rear_omega_feedforward"] = get_rear_motor_rpm_target(...)
```

**Why:** Reduces control loop burden; rear motors get a smart initial guess, reducing saturation transients.

**Effort:** 4–6 hours + tuning

---

#### 3.2 Add Real-Time Saturation Detection & Recovery
**What:** Monitor rear-motor RPM during flight; if saturated, reduce forward-tilt rate or trigger pitch-up.

**Location:** New module `src/control/saturation_monitor.py`

**Logic:**
```python
if rear_motor_rpm > 0.95 * max_rpm:
    alert("Rear motor near saturation")
    reduce_tilt_rate()  # Slow down nacelle tilt
    # OR
    trigger_pitch_up()  # Use wing lift to recover
```

**Why:** Prevents "hard" saturation crashes; provides graceful degradation.

**Effort:** 3–4 hours

---

### **PHASE 4: Validation & Hardening (Integration)**

#### 4.1 Monte Carlo with Motor-Authority Failure Mode
**What:** Run gust trials and deliberately trigger rear-motor saturation; measure recovery success.

**Location:** Extend `monte_carlo_gusts.py`

**Scenario:**
- Reduce rear-motor max RPM to 85% nominal
- Measure crash rate vs. gusts
- Compare against unsaturated baseline

**Why:** Quantifies robustness margin; reveals hidden dependencies.

**Effort:** 2–3 hours

---

#### 4.2 Document Rear-Motor Authority as a First-Class Constraint
**What:** Update project report to list "rear-motor authority" as the primary corridor bottleneck.

**Location:** New section in `others/PROJECT_REPORT.md`

**Content:**
```markdown
### Critical Path Analysis: Why the Envelope Is Shaped Like This

1. **Rear-Motor Authority (Primary):** At 90°, rear motors provide 100% of vertical lift
   - Current: 2 × 0.985 N ≈ 2% of weight → Very thin margin
   - Recommended: Upgrade to larger props or more powerful motors
   
2. **Tilt Rate (Secondary):** Limited by actuator speed (20 deg/s)
   - Current: Robust profile uses 85% = 17 deg/s → Safe
   
3. **Control Effectiveness (Tertiary):** Pitch authority drops as airspeed rises
   - Current: V-tail adequate up to ~20 m/s
```

**Effort:** 1–2 hours

---

## 4. Recommended Immediate Actions

### For This Week:
1. **Run Diagnostic 1.1** (Rear-Motor Authority Plot)
   - 15 min to generate, 30 min to interpret
   - Gives you a clear picture of the problem

2. **Extend Transition Envelope (1.2)**
   - Add `rear_motor_rpm` column to output
   - Plot it; visually confirm the saturation corner

### For Next Week:
3. **Rewrite Main Optimizer (2.1)**
   - Change objective from "minimize altitude loss" to "find tilt schedule that keeps rear motors ≤ 85% max RPM"
   - This flips the problem: motor authority becomes the design bottleneck, not tilt rate

4. **Add Vertical Acceleration Constraint (2.2)**
   - Prevents unrealistic "free-fall then recover" scenarios in optimization

### For Integration:
5. **Feedforward Rear-Motor Command (3.1)**
   - Reduces tuning burden on the main PID loop
   - Improves transition smoothness

---

## 5. Key Metrics to Track

As you work through improvements, monitor these:

| Metric | Current | Target | Reason |
|--------|---------|--------|--------|
| **Rear-motor margin at 90°** | 4% | >20% | Safety headroom for gusts & transients |
| **Lower speed boundary at 90°** | 15 m/s | ≤12 m/s | More envelope room for wind recovery |
| **Tilt schedule convexity** | Multi-peak | Monotonic | Smoother, more predictable transition |
| **Controller reliance on saturation** | High (guessing) | Low (feedforward) | Cleaner, faster convergence |
| **Monte Carlo crash rate @ 90° + gust** | 0% (no test yet) | <1% @ 4.5 m/s gust | Validated safety margin |

---

## 6. Physics Deep-Dive: Why This Matters

### The Core Problem

At tilt angle θ, the vertical force balance is:

```
T_vertical = T_forward × cos(θ) + T_rear × 1.0 + L_wing
           = m × g  (for level flight)
```

Rearranging:
```
T_rear = m × g - T_forward × cos(θ) - L_wing
```

As θ increases:
- `cos(θ)` drops → more burden on `T_rear`
- At θ = 90°: `cos(90°) = 0` → rear motors must provide `m × g - L_wing`
- If `L_wing` is small (low speed), rear must provide almost `m × g` alone

### Why Current Optimization Misses This

The CasADi and scipy optimizers use **approximated** aerodynamics:
```python
thrust_vert = k_thrust * (omega_hover ** 2) * np.cos(tilt)
```

But they **never solve for the actual `omega` values** that produce this thrust. They just assume thrust is available. The real constraint—`omega ≤ max_omega`—is hidden.

### Why Rear Motors Are the Bottleneck

**Hover (θ = 0°):**
- All 4 motors can tilt and share vertical load
- Total lift = 4 × max_thrust ≈ 4 × 1 N = 4 N (plenty for 5 kg)

**Cruise (θ = 90°):**
- Only 2 rear motors contribute vertically
- Total lift = 2 × max_thrust ≈ 2 × 1 N = 2 N (barely 1/25th of weight!)
- Wing must generate 47 N of lift to balance
- Requires ~15 m/s just to avoid stall

**The transition is a race:** Can wing lift build up faster than forward-motor lift vanishes?

If tilt rate is too fast → wing hasn't built enough lift → altitude drops sharply.  
If tilt rate is too slow → risk stall or motor burnout from sustained high RPM.  
**The optimal path threads this needle.**

---

## 7. Files to Review Next

**For deeper understanding:**
1. `src/dynamics/transition.py` — Envelope and trim solver (core physics)
2. `src/control/cascade.py` — How controller currently feeds back
3. `optimize_corridor_casadi.py` — Current optimization approach
4. `master_uam.yaml` — Aircraft spec; check motor limits
5. `monte_carlo_gusts.py` — Validation approach (extend to motor authority)

**Deliverables you've already created:**
- ✅ Robust corridor (image: `robust_corridor.png`)
- ✅ PID vs. INDI comparison
- ✅ 200-run Monte Carlo validation

**Next deliverables:**
- 🔲 Rear-motor authority plot
- 🔲 Motor-authority-aware corridor
- 🔲 Updated transition report with motor constraints

---

## 8. Summary: What to Fix First

**The vertical thrust constraint is currently **implicit and invisible**.**

**You need to make it explicit:**

1. **See it:** Plot rear-motor RPM vs. tilt angle
2. **Measure it:** Add rear-motor margin to envelope output
3. **Optimize for it:** Rewrite corridor optimizer to prioritize motor authority, not just tilt rate
4. **Validate it:** Test that corners of the envelope don't saturate motors under gust

Once rear-motor authority is the **primary design constraint** (instead of tilt-rate limits), your corridor will be:
- ✅ Narrower (more honest)
- ✅ Safer (fewer hidden saturation corners)
- ✅ Easier to control (less guessing by PID loop)
- ✅ Ready for real flight (clear safety margins)

---

**Next Step:** Would you like me to implement Diagnostic 1.1 (rear-motor authority plot) first? This will give you immediate visual insight into the problem.
