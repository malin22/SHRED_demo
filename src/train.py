
import os
import numpy as np
import pysindy as ps
from pyshred import DataManager, SHRED, SHREDEngine, SINDy_Forecaster
from .model import ProjectedLSTM
from sklearn.metrics import mean_squared_error

def prepare_manager(data: np.ndarray, lags=50, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    manager = DataManager(lags=lags, train_size=train_size, val_size=val_size, test_size=test_size)
    manager.add_data(data=data, id="Lorenz", stationary=[(0,)], compress=False, seed=seed)
    return manager

def train(data, dt, epochs=50, poly_order=3, proj_dim=4,
          threshold= 0.01, alpha=0.0001, lags=50, out_dir="runs/lorenz"):
    

    os.makedirs(out_dir, exist_ok=True)

    manager = prepare_manager(data, lags=lags)
    train_dataset, val_dataset, test_dataset = manager.prepare()

    latent_forecaster = SINDy_Forecaster(
        dt=dt, 
        poly_order=poly_order, 
        include_sine=False, 
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha)
    )
    
    sequence_model = ProjectedLSTM(hidden_size_upstream=64, proj_dim=proj_dim)

    shred = SHRED(sequence_model=sequence_model, decoder_model="MLP", latent_forecaster=latent_forecaster)
    val_errors = shred.fit(train_dataset=train_dataset, val_dataset=val_dataset, num_epochs=epochs, verbose=True)

    test_mse_proc = shred.evaluate(dataset=test_dataset)

    engine = SHREDEngine(manager, shred)
    h = len(manager.test_sensor_measurements)
    val_latents = engine.sensor_to_latent(manager.val_sensor_measurements)
    init_latents = val_latents[-shred.latent_forecaster.seed_length:]
    lat_fore = engine.forecast_latent(h=h, init_latents=init_latents)
    forecast = engine.decode(lat_fore)
    truth_test = {"Lorenz": manager.test_sensor_measurements["Lorenz"]}

    # Save simple metrics
    mse_fore = mean_squared_error(truth_test["Lorenz"].reshape(-1,3), forecast["Lorenz"].reshape(-1,3))

    np.save(os.path.join(out_dir, "forecast.npy"), forecast["Lorenz"])
    np.save(os.path.join(out_dir, "test_truth.npy"), truth_test["Lorenz"])
    np.save(os.path.join(out_dir, "val_errors.npy"), np.array(val_errors))
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Processed-space Test MSE: {test_mse_proc:.6f}\n")
        f.write(f"Forecast Test MSE (raw space): {mse_fore:.6f}\n")
        f.write(str(shred.latent_forecaster) + "\n")

    return out_dir, shred, engine
