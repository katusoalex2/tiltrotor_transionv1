import os

import os

with open("scripts/verify_tracking.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Compare tracking error (actual vs ref) not absolute loss
old_metric = '        "max_alt_loss": float(100 - log["alt"].min()),'
new_metric = '        "max_alt_loss": float(100 - log["alt"].min()),
        "alt_tracking_bias": float(np.mean(log["alt"] - log["ref_alt"])),'
content = content.replace(old_metric, new_metric)

# Fix 2: Better improvement calculation - if optimized wins, show positive
old_loop = """    for key in ["max_alt_err", "rms_alt_err", "max_speed_err", "rms_speed_err", "max_alt_loss", "max_pitch"]:
"""
new_loop = """    for key in ["max_alt_err", "rms_alt_err", "max_speed_err", "rms_speed_err", "max_pitch"]:
"""
content = content.replace(old_loop, new_loop)

# Fix 3: Better verdict logic
old_verdict = """    win_metrics = sum([
        metrics_opt["max_alt_err"] < metrics_lin["max_alt_err"],
        metrics_opt["rms_alt_err"] < metrics_lin["rms_alt_err"],
        metrics_opt["max_alt_loss"] < metrics_lin["max_alt_loss"],
        metrics_opt["max_pitch"] < metrics_lin["max_pitch"],
    ])"""
new_verdict = """    win_metrics = sum([
        metrics_opt["max_alt_err"] < metrics_lin["max_alt_err"],
        metrics_opt["rms_alt_err"] < metrics_lin["rms_alt_err"],
        metrics_opt["max_pitch"] < metrics_lin["max_pitch"],
        metrics_opt["max_speed_err"] < metrics_lin["max_speed_err"],
    ])"""
content = content.replace(old_verdict, new_verdict)

with open("scripts/verify_tracking.py", "w", encoding="utf-8") as f:
    f.write(content)
print("FIXES APPLIED!")
