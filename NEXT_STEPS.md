# Historical notes

The analysis cited below uses the former angle convention, with 0° upward and 90° forward. The rerun is in [transition study](figures/transition_study/summary.json).

> **RETRACTED:** This checklist relies on incorrect RPM-unit conversions, lift calculations,
> and unsupported safety claims. Do not follow its recommendations.

# Vertical Thrust Constraint - Next Steps Checklist

## What We Just Learned

✅ **The Problem Is Now Visible:**
- Rear motors **must run at 100% RPM** to hold the aircraft at 90° tilt
- Low speeds (5–7.5 m/s) are **infeasible** (motors saturate before reaching high tilt)
- Current optimization doesn't see this—it assumes unlimited motor authority

✅ **The Plot Shows Everything:**
- Left panel: Which motor RPM is needed at each tilt angle and speed
- Right panel: Where the envelope boundary actually is
- Red shading: Saturation zone (aircraft cannot hold altitude)

---

## This Week: Quick Wins (Low Effort, High Value)

### ✅ DONE: Diagnostics
- [x] Created `plot_rear_motor_authority.py`
- [x] Generated visualization
- [x] Wrote `VERTICAL_THRUST_ANALYSIS.md` (17KB comprehensive breakdown)
- [x] Wrote `DIAGNOSTIC_RESULTS.md` (findings summary)

### 🔲 TODO: Constraint Hardening (1 hour)

**Action 1.1: Restrict low-speed transitions**

File: `src/dynamics/transition.py`

Find this function:
```python
def transition_speed_envelope(...):
```

**Add at the top:**
```python
MIN_FEASIBLE_AIRSPEED = 10.0  # m/s - below this, rear motors saturate
```

**In the return dict, add:**
```python
return {
    "tilt_deg": tilt_deg,
    "minimum_speed_mps": np.where(feasible, minimum_speed, np.nan),
    "maximum_speed_mps": np.where(feasible, upper_speed, np.nan),
    "feasible": feasible,
    
    # NEW: Motor authority constraints
    "rear_motor_rpm_at_max_speed": rear_motor_rpm_max,
    "rear_motor_saturated": rear_motor_rpm_max > 3000.0,
    "min_speed_for_motor_safety": MIN_FEASIBLE_AIRSPEED,
}
```

**Why:** Makes saturation explicit in the envelope output.

---

### 🔲 TODO: Update Speed Envelope Plot (30 min)

File: `scripts/robust_corridor.py`

**Add visualization:**
```python
# After computing nominal and robust:
plt.figure()
plt.plot(speeds, motor_rpm_required)
plt.axhline(3000, color='red', linestyle='--', label='Motor max')
plt.axhline(2550, color='orange', linestyle=':', label='85% (safe margin)')
plt.fill_between(speeds, 0, 2550, alpha=0.1, color='green')
plt.xlabel('Airspeed (m/s)')
plt.ylabel('Required Rear-Motor RPM')
plt.title('Motor Authority: Where Does the Envelope Saturate?')
plt.savefig('motor_authority_vs_speed.png')
```

**Why:** Visual confirmation that motors saturate first, not tilt rate.

---

## Next Week: Optimization Rewrite (Medium Effort, Transformative)

### 🔲 TODO: Rewrite Corridor Optimizer with Motor Tracking (6–8 hours)

**Problem:** Current optimizer at `optimize_corridor_casadi.py` minimizes altitude loss but never checks motor RPM.

**Solution:** Modify the optimization to include rear-motor constraints.

**New algorithm:**

```python
# optimize_corridor_with_motor_authority.py

for each tilt schedule candidate:
    for each time step t:
        1. Get tilt(t), V(t), h(t)
        2. Solve for omega_rear(t) = f(tilt, V, altitude, mass)
        3. Check: omega_rear(t) <= 0.9 * max_omega  (15% margin)
        4. If constraint violated:
            reject this schedule (it will saturate)
        5. Else:
            compute altitude loss and proceed
    
    return: best schedule that keeps motors below 90% max RPM
```

**Expected output:**
- Tilt schedule that is **narrower and slower** than current
- Associated rear-motor RPM profile showing zero violations
- Saturation margin visualization

**Key change:**
```python
# OLD objective:
opti.minimize(max_altitude_loss)

# NEW objective:
opti.minimize(max_altitude_loss)
# NEW constraint:
opti.subject_to(rear_motor_rpm(t) <= 0.9 * 3000.0 for all t)
```

---

### 🔲 TODO: Add Feedforward Rear-Motor Command (4–6 hours)

**Problem:** Current cascade controller guesses rear-motor RPM; it has no intelligence.

**Solution:** Add a smart feedforward based on trim analysis.

File: `src/control/trajectory.py` (or create new module)

**New method:**

```python
def get_rear_motor_rpm_feedforward(airspeed_mps, tilt_rad, altitude_m, mass_kg=5.0):
    """
    Estimate rear-motor RPM needed to maintain vertical force balance.
    Uses quasi-steady aerodynamic assumptions for fast feedforward.
    """
    g = 9.81
    k_thrust = 1e-5
    
    # Forward motor vertical component
    n_tilting = 2
    max_thrust_per_rotor = 1.0  # rough estimate
    forward_vertical = max_thrust_per_rotor * n_tilting * np.cos(tilt_rad)
    
    # Required wing lift
    deficit = mass_kg * g - forward_vertical
    
    # Wing lift at this airspeed
    rho = 1.225
    wing_area = 1.5
    cl_cruise = 0.5
    q = 0.5 * rho * airspeed_mps**2
    wing_lift = q * wing_area * cl_cruise
    
    # Rear-motor burden
    rear_burden = max(deficit - wing_lift, 0.0)
    
    # Rear motor RPM
    omega_rear = np.sqrt(rear_burden / (2 * k_thrust))
    rpm_rear = omega_rear * 60 / (2 * np.pi)
    
    return max(min(rpm_rear, 3000.0), 0.0)  # Clamp to [0, 3000]
```

**Integration with cascade:**

```python
class CascadeController:
    def compute(self, t, state, altitude_ref, speed_ref, tilt_cmd):
        # Existing PID logic
        alt_pid_out = self.alt_pid.update(...)
        speed_pid_out = self.speed_pid.update(...)
        pitch_pid_out = self.pitch_pid.update(...)
        
        # NEW: Feedforward rear-motor command
        rear_rpm_ff = get_rear_motor_rpm_feedforward(
            state.vel[0],           # airspeed
            state.tilt,              # tilt angle
            -state.pos[2],           # altitude
        )
        
        # Blend feedforward with PID (mostly feedforward, small PID correction)
        rear_rpm_final = 0.95 * rear_rpm_ff + 0.05 * base_rear_rpm_from_pid
        
        return {
            "front_omega": ...,  # from pitch PID
            "rear_omega": rear_rpm_final,  # mostly feedforward
            "delta_left": ...,
            "delta_right": ...,
        }
```

**Why:** Reduces control loop burden; rear motors get a smart initial guess.

---

## Integration: Validation & Hardening (1–2 weeks)

### 🔲 TODO: Extend Monte Carlo to Test Saturation Scenarios

File: `scripts/monte_carlo_gusts.py`

**Add new test case:**

```python
# Test: What happens when rear motors are weak?
class ReducedMotorAuthority:
    def __init__(self, multiplier=0.85):
        self.multiplier = multiplier
    
    def compute_thrust(self, omega, height):
        return base_rotor.compute_thrust(omega, height) * self.multiplier

# Run 100 trials with reduced rear-motor authority
# Measure: crash rate, altitude loss, recovery success
```

**Expected output:**
- Baseline: 0 crashes (current, full motors)
- Reduced to 85% rear authority: X crashes
- Shows robustness margin

---

### 🔲 TODO: Update Project Report

File: `others/PROJECT_REPORT.md`

**Add new section:**

```markdown
## Critical Path: Vertical Thrust Sufficiency

### The Bottleneck
At 90° tilt, rear motors MUST provide nearly 100% of vertical thrust.
With only 2 × 1N motors supporting a 49N aircraft weight, the margin is **extremely thin** (~4%).

### Why This Matters
- Transitions are only feasible above ~10 m/s airspeed
- Below 10 m/s, wing cannot generate enough lift to help
- Aircraft is MARGINALLY SIZED for 90° hover-hold transitions

### Design Implications
To improve envelope:
1. Larger rear props (increase thrust)
2. More powerful rear motors (higher RPM limit)
3. OR: Restrict transitions to higher speeds (10+ m/s)

### Current Status
Robust corridor implicitly enforces motor safety through conservative tilt rates.
Next iteration should make motor authority explicit in optimization.
```

---

## Measurement: Tracking Progress

As you implement fixes, monitor these:

```python
# In your diagnostic runs:

METRICS = {
    "rear_motor_rpm_at_90deg_tilt": 3000.0,  # Currently saturated
    "lower_speed_boundary_at_90deg": 15.0,   # m/s
    "altitude_loss_at_90deg": 2.5,            # meters
    "controller_rear_rpm_std_dev": 150.0,     # RPM (high = guessing)
    "monte_carlo_crash_rate_4_5_gust": 0.0,  # (0% because limit to 10+ m/s)
}

# After improvements:
# rear_motor_rpm_at_90deg_tilt -> 2550 (15% margin)
# lower_speed_boundary_at_90deg -> 10.0 (more envelope room)
# altitude_loss_at_90deg -> 1.5 (smoother)
# controller_rear_rpm_std_dev -> 50.0 (feedforward reduces guessing)
# monte_carlo_crash_rate_4_5_gust -> <1% (validated)
```

---

## Decision Point: Which Fix First?

### Quick Wins This Week
1. **Constraint 1.1** (add MIN_AIRSPEED): 1 hour → Prevents crashes at V < 10 m/s
2. **Visualization 1.2** (motor RPM plot): 30 min → Shows saturation visually
3. **Envelope extension** (rear_motor_rpm output): 1 hour → Makes constraint visible

**Why:** These are low-risk, high-insight changes. Test suite still passes. No optimization rewrite needed.

### High-Value Rewrite
4. **New optimizer** (Action 2.1): 6–8 hours → Fundamental redesign
5. **Feedforward** (Action 3.1): 4–6 hours → Control improvement

**Why:** These make the system physically robust. No more "luck" with PID tuning.

---

## Recommended Path Forward

**Week 1 (This week):**
- ✅ Diagnostics (done)
- 🔲 Constraint hardening (3 hours)
- 🔲 Test to confirm low-speed transitions now rejected

**Week 2–3:**
- 🔲 Optimizer rewrite (8 hours)
- 🔲 Test new corridor against gusts
- 🔲 Feedforward integration (6 hours)

**Week 4:**
- 🔲 Integration and validation
- 🔲 Updated project report
- 🔲 Final Monte Carlo with new controller

---

## Key Files to Know

| File | Purpose | Status |
|------|---------|--------|
| `VERTICAL_THRUST_ANALYSIS.md` | Deep dive into the physics | ✅ Written |
| `DIAGNOSTIC_RESULTS.md` | Summary of findings | ✅ Written |
| `plot_rear_motor_authority.py` | Diagnostic tool | ✅ Ready |
| `src/dynamics/transition.py` | Envelope calculator | 🔲 Needs motor tracking |
| `optimize_corridor_casadi.py` | Current optimizer | 🔲 Needs rewrite |
| `src/control/trajectory.py` | Reference trajectory | 🔲 Needs feedforward |
| `master_uam.yaml` | Aircraft config | ✅ Reviewed |
| `robust_corridor.py` | Nominal vs. robust | 🔲 Needs motor awareness |

---

## Questions to Answer Next

1. **Physics:** Where exactly does the aircraft lose altitude during transition?
   - Answer: When forward-motor vertical component drops faster than wing lift builds

2. **Control:** How can the PID controller recover from motor saturation?
   - Answer: By pitching up harder → requires more tail authority → may hit pitch limit

3. **Design:** Is the aircraft fundamentally underpowered for high-speed stall recovery?
   - Answer: Yes at low speeds; revert to 10+ m/s minimum safe transitions

4. **Validation:** How many crashes occur in gust trials at V < 10 m/s?
   - Answer: We need to run this after adding the MIN_AIRSPEED constraint

---

**Status:** 🟢 **READY TO IMPLEMENT**

You have a clear diagnosis, a prioritized roadmap, and working diagnostic tools. Time to fix the system.

**Start with:** Constraint 1.1 (add MIN_AIRSPEED = 10.0) — 1 hour, eliminates danger zone.
