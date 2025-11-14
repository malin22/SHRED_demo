
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_phase(data: np.ndarray, out_path: str):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(data[:,0], data[:,1], data[:,2], linewidth=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Lorenz Attractor (Phase Space)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_series(truth: np.ndarray, recon: np.ndarray, forecast: np.ndarray, which: int, out_path: str, label: str):
    fig = plt.figure(figsize=(9,4))
    plt.plot(truth[:, which], label=f"Truth ({label})", linewidth=1)
    plt.plot(recon[:, which], label=f"Reconstruction ({label})", linestyle="--")
    plt.plot(forecast[:, which], label=f"Forecast SINDy ({label})", linestyle="-.")
    plt.title(f"Lorenz: Test segment ({label} component)")
    plt.xlabel("Test timestep"); plt.ylabel(label)
    plt.legend(); plt.grid(True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def make_all_plots(data: np.ndarray, recon: np.ndarray, forecast: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plot_phase(data, os.path.join(out_dir, "phase.png"))
    plot_series(recon, recon, forecast, 0, os.path.join(out_dir, "x.png"), "x")
    plot_series(recon, recon, forecast, 1, os.path.join(out_dir, "y.png"), "y")
    plot_series(recon, recon, forecast, 2, os.path.join(out_dir, "z.png"), "z")
