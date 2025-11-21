
import os
import numpy as np
import pysindy as ps
from pyshred import DataManager, SHRED, SHREDEngine, SINDy_Forecaster
from model import ProjectedLSTM
import pdb

def prepare_manager(data: np.ndarray, lags=50, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    manager = DataManager(lags=lags, train_size=train_size, val_size=val_size, test_size=test_size)
    manager.add_data(data=data, id="Lorenz", stationary=[(0,)], compress=False, seed=seed)
    return manager

def train(data, dt, epochs=50, poly_order=3, proj_dim=4,
          threshold= 0.01, alpha=0.0001, lags=50, out_dir="runs/lorenz"):
    
    out_dir = os.path.join(out_dir, "train")
    os.makedirs(out_dir, exist_ok=True)

    manager = prepare_manager(data, lags=lags)

    #|-------- TRAIN --------|----- VAL -----|----- TEST -----|
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

    #number of timesteps in the test split/how many should be forecasted
    h = len(manager.test_sensor_measurements)

    #encodes the sensor measurements of the validation set in to the model's latent space
    val_latents = engine.sensor_to_latent(manager.val_sensor_measurements)

    #takes the last timesteps (as many as lags) of the validation set and puts it inot latent space, to use it as a starting point of forecasting the test set
    init_latents = val_latents[-shred.latent_forecaster.seed_length:]

    #forecast h timesteps, starting form init_latents
    lat_fore = engine.forecast_latent(h=h, init_latents=init_latents)

    #decode the forecasted latents
    forecast = engine.decode(lat_fore)

    #collects the ground-truth data of the test segment
    #truth_test = {"Lorenz": manager.test_sensor_measurements}
    truth_test = {"Lorenz": data[-h:]}

    recon = engine.decode(engine.sensor_to_latent(manager.test_sensor_measurements))["Lorenz"][:h]



    np.save(os.path.join(out_dir, "forecast.npy"), forecast["Lorenz"])
    np.save(os.path.join(out_dir, "test_truth.npy"), truth_test["Lorenz"])
    np.save(os.path.join(out_dir, "val_errors.npy"), np.array(val_errors))
    np.save(os.path.join(out_dir, "reconstruction.npy"), recon)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"Processed-space Test MSE: {test_mse_proc:.6f}\n")
        f.write(str(shred.latent_forecaster) + "\n")

    print("done!!!")

    return out_dir, shred, engine


def main():
    data = np.load("data/lorenz.npy")
    train(data=data, dt=0.01, epochs=50, poly_order=3, proj_dim=4,
          threshold= 0.01, alpha=0.0001, lags=50, out_dir="runs/lorenz")
    
if __name__ == "__main__":
    main()


