from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

@dataclass
class AircraftState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tilt: float = 0.0
    lift: float = 0.0
    drag: float = 0.0
    V_T: float = 0.0
    alpha: float = 0.0

    def copy(self):
        return AircraftState(
            pos=self.pos.copy(), vel=self.vel.copy(), att=self.att.copy(), omega=self.omega.copy(),
            tilt=float(self.tilt), lift=float(self.lift), drag=float(self.drag),
            V_T=float(self.V_T), alpha=float(self.alpha)
        )