import numpy as np
import yaml

class GainScheduler:
    def __init__(self, config_path='configs/aircraft.yaml'):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Read the gains directly from the form!
        self.hover_gains = {'kp': cfg['hover_kp'], 'ki': cfg['hover_ki'], 'kd': cfg['hover_kd']}
        self.cruise_gains = {'kp': cfg['cruise_kp'], 'ki': cfg['cruise_ki'], 'kd': cfg['cruise_kd']}
        self.cruise_speed = 20.0

    def get_gains(self, current_speed: float) -> dict:
        blend = np.clip(abs(current_speed) / self.cruise_speed, 0.0, 1.0)
        kp = (1 - blend) * self.hover_gains['kp'] + blend * self.cruise_gains['kp']
        ki = (1 - blend) * self.hover_gains['ki'] + blend * self.cruise_gains['ki']
        kd = (1 - blend) * self.hover_gains['kd'] + blend * self.cruise_gains['kd']
        return {'kp': kp, 'ki': ki, 'kd': kd}
