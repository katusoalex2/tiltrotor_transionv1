"""NACA Airfoil Generator and File Parser."""
import numpy as np

def generate_naca4(m: float, p: float, t: float, num_points: int = 30):
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if m > 0 and p > 0:
        for i in range(len(x)):
            if x[i] < p:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / p**2) * (p - x[i])
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl

def parse_airfoil_file(file_bytes):
    """Parses a standard .dat/.txt airfoil file from AirfoilTools."""
    try:
        content = file_bytes.decode("utf-8").splitlines()
        coords = []
        for line in content:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    coords.append([x, y])
                except ValueError:
                    continue
        coords = np.array(coords)
        # Split into upper and lower surfaces based on max X (leading edge)
        le_idx = np.argmin(coords[:, 0])
        upper = coords[:le_idx+1]
        lower = coords[le_idx:]
        return upper[:,0], upper[:,1], lower[:,0], lower[:,1]
    except Exception as e:
        return None
