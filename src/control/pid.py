class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, limit: float = 1.0):
        self.limit = limit
        self.set_gains(kp, ki, kd)
        self.integral = 0.0
        self.prev_measurement = None

    def set_gains(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0: return 0.0
        error = setpoint - measurement
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        if i_term > self.limit: i_term = self.limit
        if i_term < -self.limit: i_term = -self.limit
        if self.prev_measurement is None: self.prev_measurement = measurement
        d_measurement = (measurement - self.prev_measurement) / dt
        d_term = -self.kd * d_measurement
        self.prev_measurement = measurement
        output = p_term + i_term + d_term
        return max(-self.limit, min(self.limit, output))
