
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_phase(data, out_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(data[:,0], data[:,1], data[:,2], linewidth=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Lorenz Attractor (Phase Space)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_series(truth, recon, forecast, which, out_path, label):
    fig = plt.figure(figsize=(9,4))
    plt.plot(truth[:, which], label=f"Truth ({label})", linewidth=1)
    plt.plot(recon[:, which], label=f"Reconstruction ({label})", linestyle="--")
    plt.plot(forecast[:, which], label=f"Forecast SINDy ({label})", linestyle="-.")
    plt.title(f"Lorenz: Test segment ({label} component)")
    plt.xlabel("Test timestep"); plt.ylabel(label)
    plt.legend(); plt.grid(True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_total_error(truth, recon, out_path):
    # L2 norm error per timestep across all 3 dims
    error = np.linalg.norm(truth - recon, axis=1)   # shape (T,)

    fig = plt.figure(figsize=(10,4))
    plt.plot(error, label="Total Reconstruction Error (L2 norm)")
    plt.xlabel("Test timestep")
    plt.ylabel("Error")
    plt.title("Total Reconstruction Error Over Time")
    plt.grid(True)
    plt.legend()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_series_stacked(truth, recon, forecast, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    labels = ["x", "y", "z"]

    for i, ax in enumerate(axes):
        ax.plot(truth[:, i], label=f"Truth ({labels[i]})", linewidth=1)
        ax.plot(recon[:, i], label=f"Reconstruction ({labels[i]})", linestyle="--")
        ax.plot(forecast[:, i], label=f"Forecast SINDy ({labels[i]})", linestyle="-.")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Test timestep")
    fig.suptitle("Lorenz Test Segment", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_all_plots(data, truth, recon, forecast, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Phase plot
    plot_phase(data, os.path.join(out_dir, "phase.png"))

    # Stacked xyz time series
    plot_series_stacked(truth, recon, forecast,
                        os.path.join(out_dir, "xyz_stacked.png"))

    # Total reconstruction error
    plot_total_error(truth, recon,
                     os.path.join(out_dir, "recon_total_error.png"))

    # (optional) individual component plots
    plot_series(truth, recon, forecast, 0, os.path.join(out_dir, "x.png"), "x")
    plot_series(truth, recon, forecast, 1, os.path.join(out_dir, "y.png"), "y")
    plot_series(truth, recon, forecast, 2, os.path.join(out_dir, "z.png"), "z")




