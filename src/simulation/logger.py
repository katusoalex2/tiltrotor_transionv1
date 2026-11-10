"""Data logging for simulation histories."""

import numpy as np
from src.dynamics.state import AircraftState


class StateLogger:
    """Records the history of an AircraftState during a simulation."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def log(self, t: float, state: AircraftState, extra: dict | None = None) -> None:
        """Record the state at a specific time."""
        record = {
            "t": t,
            "pos": state.pos.copy(),
            "vel": state.vel.copy(),
            "att": state.att.copy(),
            "omega": state.omega.copy(),
            "tilt": state.tilt,
            "lift": state.lift,
            "drag": state.drag,
        }
        if extra:
            record.update(extra)
        self.history.append(record)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert the recorded history into a dictionary of NumPy arrays for plotting."""
        if not self.history:
            raise ValueError("No data logged.")

        keys = self.history[0].keys()
        data = {}

        for k in keys:
            data[k] = np.array([rec[k] for rec in self.history])

        return data
    