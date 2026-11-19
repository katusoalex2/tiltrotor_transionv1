"""Generate plots for the cosine tilt reference and a closed-loop run."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.control.reference import TransitionReference


def main():
    output_dir = os.path.join("figures")
    os.makedirs(output_dir, exist_ok=True)

    transition_time = 10.0
    hold_time = 2.0
    duration = 15.0
    dt = 0.01
    times = np.arange(0.0, duration + dt, dt)
    smooth_reference = TransitionReference(
        start_altitude=100.0,
        target_speed=20.0,
        transition_time=transition_time,
        tilt_hold_time=hold_time,
    )
    smooth_tilt = np.rad2deg(
        np.array([smooth_reference.at(t)["tilt"] for t in times])
    )
    linear_tilt = np.rad2deg(
        np.pi / 2 * np.clip(times / transition_time, 0.0, 1.0)
    )
    smooth_rate = np.gradient(smooth_tilt, dt)
    linear_rate = np.gradient(linear_tilt, dt)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, linear_tilt, "--", label="Linear reference")
    axes[0].plot(times, smooth_tilt, linewidth=2, label="Cosine reference")
    axes[0].set_ylabel("Tilt command (deg)")
    axes[0].set_title("Linear vs cosine tilt schedule (project convention)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(times, linear_rate, "--", label="Linear rate")
    axes[1].plot(times, smooth_rate, linewidth=2, label="Cosine rate")
    axes[1].axhline(20.0, color="r", linestyle=":", label="20 deg/s configured limit")
    axes[1].axhline(-20.0, color="r", linestyle=":")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Tilt rate (deg/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "tilt_schedule_comparison.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    image_convention_tilt = 90.0 - smooth_tilt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, image_convention_tilt, color="tab:orange", linewidth=2)
    axes[0].set_ylabel("Image angle (deg)")
    axes[0].set_title("Image convention: 90° hover → 0° cruise")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(times, smooth_tilt, color="tab:purple", linewidth=2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Project angle (deg)")
    axes[1].set_title("Project convention: 0° hover → 90° cruise")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "tilt_convention_conversion.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Saved figures/tilt_schedule_comparison.png")
    print("Saved figures/tilt_convention_conversion.png")


if __name__ == "__main__":
    main()
