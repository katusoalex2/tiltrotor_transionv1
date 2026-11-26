"""Evaluate recorded transition outcomes without altering simulation results."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def assess_transition_targets(
    data: Mapping[str, np.ndarray],
    *,
    transition_start_s: float,
    transition_end_s: float,
    pitch_settling_s: float = 3.0,
    minimum_alpha_airspeed_mps: float = 15.0,
) -> list[dict[str, str]]:
    """Assess user-requested simulation targets; incomplete runs cannot pass."""
    bounds = [transition_start_s, transition_end_s, pitch_settling_s, minimum_alpha_airspeed_mps]
    if not np.all(np.isfinite(bounds)) or transition_start_s < 0 or transition_end_s <= transition_start_s:
        raise ValueError("transition times and target thresholds must be finite and ordered")
    if pitch_settling_s < 0 or minimum_alpha_airspeed_mps <= 0:
        raise ValueError("pitch settling time must be non-negative and minimum airspeed positive")

    names = ("t", "alt", "ref_alt", "pitch", "tilt", "V_T", "u_body", "w_body")
    arrays = {name: np.asarray(data[name], dtype=float) for name in names}
    times = arrays["t"]
    if times.ndim != 1 or any(values.shape != times.shape for values in arrays.values()):
        raise ValueError("target assessment inputs must be matching one-dimensional arrays")
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("target assessment inputs must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("simulation times must be strictly increasing")

    transition = (times >= transition_start_s) & (times < transition_end_s)
    if not np.any(transition):
        return [
            {"Target": name, "Result": "Unassessed", "Observed": "No transition samples"}
            for name in ("Transition completion", "Peak altitude loss", "Pitch", "Angle of attack")
        ]

    final_tilt = float(arrays["tilt"][transition][-1])
    complete_time = bool(times[-1] >= transition_end_s)
    complete = complete_time and abs(final_tilt) <= 0.1
    rows = [{
        "Target": "Transition completion",
        "Result": "Met" if complete else "Not met",
        "Observed": f"Final transition tilt {final_tilt:.2f}°; recorded through {times[-1]:.2f} s",
    }]

    peak_loss = float(np.max(np.maximum(
        arrays["ref_alt"][transition] - arrays["alt"][transition], 0.0
    )))
    loss_result = (
        "Not met" if peak_loss > 1.234 + 1e-9 or (complete and peak_loss < 0.5 - 1e-9)
        else "Met" if complete
        else "Unassessed (incomplete)"
    )
    rows.append({
        "Target": "Peak altitude loss: 0.5–1.234 m",
        "Result": loss_result,
        "Observed": f"{peak_loss:.3f} m",
    })

    pitch_mask = transition & (times >= transition_start_s + pitch_settling_s)
    if np.any(pitch_mask):
        pitch = arrays["pitch"][pitch_mask]
        pitch_result = (
            "Not met" if np.any((pitch < 2.0 - 1e-9) | (pitch > 2.8 + 1e-9))
            else "Met" if complete
            else "Unassessed (incomplete)"
        )
        pitch_observed = f"{np.min(pitch):.2f}–{np.max(pitch):.2f}°"
    else:
        pitch_result, pitch_observed = "Unassessed", "No samples after settling allowance"
    rows.append({
        "Target": f"Pitch after {pitch_settling_s:g} s: +2.0–+2.8°",
        "Result": pitch_result,
        "Observed": pitch_observed,
    })

    alpha_mask = (
        transition
        & (arrays["V_T"] >= minimum_alpha_airspeed_mps)
        & (np.hypot(arrays["u_body"], arrays["w_body"]) > 1e-6)
    )
    if np.any(alpha_mask):
        alpha = np.rad2deg(np.arctan2(arrays["w_body"][alpha_mask], arrays["u_body"][alpha_mask]))
        alpha_result = (
            "Not met" if np.any((alpha < 5.0 - 1e-9) | (alpha > 12.0 + 1e-9))
            else "Met" if complete
            else "Unassessed (incomplete)"
        )
        alpha_observed = f"{np.min(alpha):.2f}–{np.max(alpha):.2f}° ({alpha.size} samples)"
    else:
        alpha_result, alpha_observed = "Unassessed", "No qualifying airspeed samples"
    rows.append({
        "Target": f"Angle of attack at ≥{minimum_alpha_airspeed_mps:g} m/s: 5–12°",
        "Result": alpha_result,
        "Observed": alpha_observed,
    })
    return rows
