import numpy as np
from src.control.pid import PIDController

def test_pid_converges():
    pid = PIDController(kp=2.0, ki=1.0, kd=0.05)
    setpoint = 10.0
    measurement = 0.0
    dt = 0.01
    
    for _ in range(1000):
        output = pid.compute(setpoint, measurement, dt)
        measurement += output * dt 
        
    assert np.isclose(measurement, setpoint, atol=1.0)
