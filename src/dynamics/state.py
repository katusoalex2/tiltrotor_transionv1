from dataclasses import dataclass, field
import numpy as np

@dataclass
class AircraftState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    att: np.ndarray = field(default_factory=lambda: np.zeros(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tilt: float = 0.0
    lift: float = 0.0
    drag: float = 0.0

    def copy(self):
        return AircraftState(
            pos=self.pos.copy(), vel=self.vel.copy(), att=self.att.copy(), omega=self.omega.copy(),
            tilt=float(self.tilt), lift=float(self.lift), drag=float(self.drag)
        )
