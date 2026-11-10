import numpy as np

class TransitionReference:
    def __init__(self, start_altitude: float, target_speed: float, transition_time: float):
        self.start_altitude = start_altitude
        self.target_speed = target_speed
        self.transition_time = transition_time

    def at(self, t: float) -> dict:
        progress = min(t / self.transition_time, 1.0)
        tilt = progress * (np.pi / 2)
        speed = progress * self.target_speed
        altitude = self.start_altitude
        return {'altitude': altitude, 'speed': speed, 'tilt': tilt, 'progress': progress}
