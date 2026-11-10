import numpy as np
from src.dynamics.state import AircraftState

class StateLogger:
    def __init__(self):
        self.history = []

    def log(self, t: float, state: AircraftState, extra: dict = None):
        record = {
            't': t, 'pos': state.pos.copy(), 'vel': state.vel.copy(), 'att': state.att.copy(),
            'omega': state.omega.copy(), 'tilt': state.tilt, 'lift': state.lift, 'drag': state.drag
        }
        if extra: record.update(extra)
        self.history.append(record)

    def to_arrays(self):
        if not self.history: raise ValueError('No data')
        keys = self.history[0].keys()
        return {k: np.array([rec[k] for rec in self.history]) for k in keys}
