"""Full atmospheric wind model with direction, shear, and turbulence."""
import numpy as np

class WindField:
    def __init__(self, mean_speed=5.0, mean_direction_deg=0.0, 
                 turbulence_intensity=0.2, ref_altitude=10.0, surface_roughness=0.03):
        """
        Args:
            mean_speed: Average wind speed at ref_altitude (m/s)
            mean_direction_deg: Direction wind is blowing FROM, degrees from North
            turbulence_intensity: 0 to 1 (fraction of mean speed that fluctuates)
            ref_altitude: Reference height for wind speed measurement (m)
            surface_roughness: z0 parameter for log wind profile (m)
        """
        self.mean_speed = mean_speed
        self.mean_direction = np.deg2rad(mean_direction_deg)
        self.turbulence = turbulence_intensity
        self.ref_alt = ref_altitude
        self.z0 = surface_roughness
        
        # Internal state for turbulence (Dryden-like first-order filter)
        self.turb_state = np.zeros(3)
        self.time = 0.0
        self.dt = 0.01
        self.tau = 1.0  # Turbulence time constant (s)
        
    def get_wind(self, t: float, altitude: float) -> np.ndarray:
        """Returns the 3D wind velocity vector at given time and altitude (NED frame)."""
        dt = t - self.time
        self.time = t
        
        altitude = max(altitude, 0.1)  # Prevent log(0)
        
        # 1. Log wind profile (wind speed increases with altitude)
        if altitude < 1.0:
            # Below 1m, use linear interpolation to avoid log singularities
            speed_ratio = altitude
        else:
            speed_ratio = np.log(altitude / self.z0) / np.log(self.ref_alt / self.z0)
        speed_ratio = max(speed_ratio, 0.1)  # Prevent zero wind
        
        steady_speed = self.mean_speed * speed_ratio
        
        # 2. Direction (rotate around vertical axis)
        # Wind blowing FROM direction means it pushes TOWARD the opposite
        # NED: X=North, Y=East
        wind_x = -steady_speed * np.cos(self.mean_direction)  # Blowing toward opposite of "from"
        wind_y = -steady_speed * np.sin(self.mean_direction)
        
        # 3. Turbulence (first-order Dryden filter)
        if dt > 0 and self.turbulence > 0:
            # Generate Gaussian noise and filter it
            noise = np.random.randn(3) * self.turbulence * steady_speed * np.sqrt(2 * dt / self.tau)
            self.turb_state = self.turb_state * (1 - dt / self.tau) + noise
        
        # 4. Combine
        wind = np.array([
            wind_x + self.turb_state[0],
            wind_y + self.turb_state[1],
            self.turb_state[2]  # Vertical turbulence only (no mean vertical wind)
        ])
        
        return wind

class ConstantGust:
    """Simple constant gust (for backwards compatibility)."""
    def __init__(self, u: float = 0.0, v: float = 0.0, w: float = 0.0):
        self.vector = np.array([u, v, w], dtype=float)
    
    def get_wind(self, t: float, altitude: float = 100.0) -> np.ndarray:
        return self.vector.copy()
