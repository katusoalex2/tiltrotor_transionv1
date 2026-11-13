class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, limit: float = 1.0):
        self.limit = limit
        self.set_gains(kp, ki, kd)
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None

    def set_gains(self, kp: float, ki: float, kd: float):
        """Allows us to change gains on the fly (Gain Scheduling)."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        
        error = setpoint - measurement
        p_term = self.kp * error
        
        if self.prev_measurement is None:
            self.prev_measurement = measurement
        d_measurement = (measurement - self.prev_measurement) / dt
        d_term = -self.kd * d_measurement
        self.prev_measurement = measurement

        candidate_integral = self.integral + error * dt
        if self.ki != 0.0:
            integral_bounds = sorted((
                (-self.limit - p_term - d_term) / self.ki,
                (self.limit - p_term - d_term) / self.ki,
            ))
            candidate_integral = max(
                integral_bounds[0],
                min(integral_bounds[1], candidate_integral),
            )
        self.integral = candidate_integral
        output = p_term + self.ki * self.integral + d_term

        return float(max(-self.limit, min(self.limit, output)))