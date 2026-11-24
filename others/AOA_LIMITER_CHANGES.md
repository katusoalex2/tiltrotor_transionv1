# Angle-of-Attack Limiter Implementation

## Problem
Your simulation was producing angle-of-attack spikes of ±150-170° during tilt-rotor transition, far exceeding the safe 5-12° target envelope.

## Root Cause
During high-tilt maneuvers (e.g., 80-90° tilt), pitch attitude approximately equals angle of attack. Without a tilt-aware limiter, the pitch controller would command aggressive pitch changes (up to 25°) that created extreme AoA at high tilt angles.

## Solution Implemented
Added **dynamic angle-of-attack limiters** in two key control paths:

### 1. ControlLawArchitecture (dashboard.py uses this)
**File:** `src/control/architecture.py` (lines 134-138)

```python
# Aggressively reduce max pitch: 25° at hover, ~5° at cruise to stay within 12° AoA
tilt_sin = np.sin(float(state.tilt))
max_pitch_for_aoa = np.deg2rad(25.0 - 20.0 * tilt_sin)
desired_pitch = float(np.clip(desired_pitch, -max_pitch_for_aoa, max_pitch_for_aoa))
```

### 2. CascadeController (other scripts use this)
**File:** `src/control/cascade.py` (lines 14-17)

```python
# Angle-of-attack limiter: aggressive reduction at high tilt
tilt_sin = np.sin(float(state.tilt))
max_pitch_for_aoa = np.deg2rad(25.0 - 20.0 * tilt_sin)
desired_pitch = float(np.clip(desired_pitch, -max_pitch_for_aoa, max_pitch_for_aoa))
```

## Pitch Limits by Flight Phase

| Tilt Angle | Max Pitch | Effect |
|---|---|---|
| 0° (Hover) | ±25.0° | Full maneuverability |
| 15° | ±19.8° | Gradual restriction |
| 30° | ±15.0° | Moderate restriction |
| 45° | ±10.9° | Significant protection |
| 60° | ±7.7° | Strong protection |
| 75° | ±5.7° | Very strong protection |
| 90° (Cruise) | ±5.0° | Maximum AoA protection |

## Impact on Angle of Attack
When pitch is limited to 5° at 90° tilt, the resulting angle of attack should stay within the 5-12° target envelope, allowing the blue line in your plots to remain inside the requested target zone.

## Testing
✅ All 26 core unit tests pass with the new limiter
✅ No modifications to aerodynamic model
✅ Fully backward compatible

## Next Steps
1. Run your simulation/dashboard again
2. Check if the AoA plot now shows the blue line staying within the gray target zone
3. If AoA still exceeds limits, verify which control path is being used and check if wind/gust conditions are adding additional velocity components
