"""Aircraft state container.

Frames (NED - North-East-Down, matching the project PDF):
    - Position, velocity, Euler angles: world frame (z DOWN)
    - Angular velocity: body frame (p, q, r)
    - Tilt: rotor tilt angle, 0 = hover (thrust up), pi/2 = cruise (thrust forward)

Units: SI (meters, seconds, radians, kilograms, newtons).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AircraftState:
    """Complete aircraft state at one instant.

    Attributes:
        pos: Position of CG in world frame, meters. Shape (3,).
        vel: Velocity of CG in world frame, m/s. Shape (3,).
        att: Euler angles (roll, pitch, yaw) in world frame, rad. Shape (3,).
        omega: Angular velocity in body frame (p, q, r), rad/s. Shape (3,).
        tilt: Rotor tilt angle, rad. 0 = thrust up, pi/2 = thrust forward.
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tilt: float = 0.0
    lift: float = 0.0
    drag: float = 0.0

    def copy(self) -> "AircraftState":
        """Return a deep copy. Useful before integration to preserve history."""
        return AircraftState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            att=self.att.copy(),
            omega=self.omega.copy(),
            tilt=float(self.tilt),
            lift=float(self.lift),
            drag=float(self.drag),
        )