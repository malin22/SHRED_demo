import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import os


#create lorenz system data
def lorenz(t, s, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = s
    return [sigma*(y-x), x*(rho-z) - y, x*y - beta*z]

def generate(T=50, dt_sample=0.01):

    # Simulation
    T = T
    dt_sample = dt_sample
    t = np.arange(0, T, dt_sample)
    sol = solve_ivp(lorenz, [t[0], t[-1]], y0=[1.0, 1.0, 1.0], t_eval=t, rtol=1e-9, atol=1e-9)
    data = sol.y.T
    print(data.shape)  # (time, 3)

    os.makedirs("data", exist_ok=True)

    # Speichere die Lorenz-Daten
    np.save("data/lorenz.npy", data)


def generate_noisy(
    T=50,
    dt_sample=0.01,
    noise_std=0.1,
    relative=False,
    seed=42,
    save_path="data/lorenz_noisy.npy",
):
    """
    Generates Lorenz system data with additive Gaussian noise.

    Parameters:
        T (float): total simulation time
        dt_sample (float): sampling time step
        noise_std (float): noise standard deviation
        relative (bool): if True → noise_std is relative (percentage of signal)
        seed (int): random seed for reproducibility
        save_path (str): where to save the noisy data

    Returns:
        t: time array
        noisy_data: array with shape (time, 3)
    """
    # Generate clean data
    t, clean_data = generate(T=T, dt_sample=dt_sample, save_path="data/lorenz_clean_tmp.npy")

    # Remove temporary file
    if os.path.exists("data/lorenz_clean_tmp.npy"):
        os.remove("data/lorenz_clean_tmp.npy")

    # Set RNG
    rng = np.random.default_rng(seed)

    if relative:
        # noise_std is percentage of the absolute value
        noise = rng.normal(0, noise_std * np.abs(clean_data), clean_data.shape)
    else:
        # absolute Gaussian noise
        noise = rng.normal(0, noise_std, clean_data.shape)

    noisy_data = clean_data + noise

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, noisy_data)

    print("Generated NOISY Lorenz data:", noisy_data.shape)
    print(f"Saved to {save_path}")

    return t, noisy_data
