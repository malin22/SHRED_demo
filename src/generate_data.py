import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import os


#create lorenz system data
def lorenz(t, s, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = s
    return [sigma*(y-x), x*(rho-z) - y, x*y - beta*z]

# Simulation
T = 50.0
dt_sample = 0.01
t = np.arange(0, T, dt_sample)
sol = solve_ivp(lorenz, [t[0], t[-1]], y0=[1.0, 1.0, 1.0], t_eval=t, rtol=1e-9, atol=1e-9)
data = sol.y.T
print(data.shape)  # (time, 3)

os.makedirs("data", exist_ok=True)

# Speichere die Lorenz-Daten
np.save("data/lorenz.npy", data)