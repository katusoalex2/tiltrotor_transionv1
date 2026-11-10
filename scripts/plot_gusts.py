same import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib
matplotlib.use('TkAgg') # Force a visible window
import matplotlib.pyplot as plt
import numpy as np
from src.wind.gust import ConstantGust, DrydenGust

def main():
    dt = 0.01
    n_steps = 2000
    constant = ConstantGust(u=0.0, v=5.0, w=2.0)
    const_log = np.array([constant.step(dt) for _ in range(n_steps)])
    dryden = DrydenGust(sigma=(2.5, 2.5, 1.5), airspeed=15.0, seed=42)
    dryden_log = np.array([dryden.step(dt) for _ in range(n_steps)])
    t = np.arange(n_steps) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes[0, 0].plot(t, const_log[:, 0], label='u'); axes[0, 0].plot(t, const_log[:, 1], label='v'); axes[0, 0].plot(t, const_log[:, 2], label='w')
    axes[0, 0].set_title('Constant Gust'); axes[0, 0].grid(True); axes[0, 0].legend()

    axes[0, 1].plot(t, dryden_log[:, 0], label='u'); axes[0, 1].plot(t, dryden_log[:, 1], label='v'); axes[0, 1].plot(t, dryden_log[:, 2], label='w')
    axes[0, 1].set_title('Dryden Turbulence'); axes[0, 1].grid(True); axes[0, 1].legend()

    for i, name in enumerate(['u', 'v', 'w']): axes[1, 0].hist(dryden_log[:, i], bins=40, alpha=0.5, label=name)
    axes[1, 0].set_title('Gaussian Distribution'); axes[1, 0].legend()

    for i, name in enumerate(['u', 'v', 'w']):
        signal = dryden_log[:, i] - dryden_log[:, i].mean()
        freqs = np.fft.rfftfreq(len(signal), dt)
        psd = np.abs(np.fft.rfft(signal)) ** 2
        psd[0] = psd[1]
        axes[1, 1].loglog(freqs[1:], psd[1:], label=name, alpha=0.7)
    axes[1, 1].set_title('PSD (1/f^2)'); axes[1, 1].grid(True, which='both'); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('gust_models.png', dpi=100)
    print('Saved plot to gust_models.png')
    plt.show(block=True)

if __name__ == '__main__': main()
