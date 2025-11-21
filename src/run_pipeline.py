import numpy as np
import os
from train_shred import train
import generate_data
import plotting


experiment_name = "lorenz_noisy_0.7"
data_path = "data/lorenz_noisy_0.7.npy"
epochs = 50


output_dir = os.path.join('runs', experiment_name)
train_data_dir = os.path.join(output_dir, "train")


def create_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(data_path):
        #os.makedirs(data_path, exist_ok=True)
        print("generating data...")
        data = generate_data.generate_noisy(noise_std=0.7, relative=False, save_path=data_path)
    else:
        print("loading Lorenz data.")

    data = np.load(data_path)
    return data


def main():

    print("here")
    data = create_data()

    if not os.path.exists(train_data_dir):
        train(data=data, dt=0.01, epochs=epochs, poly_order=3, proj_dim=4,
            threshold= 0.01, alpha=0.0001, lags=50, out_dir=output_dir)
        

    #plotting.plot_phase(data=data, out_path=output_dir)

    forecast = np.load(os.path.join(train_data_dir, "forecast.npy"))
    recon = np.load(os.path.join(train_data_dir, "reconstruction.npy"))
    truth = np.load(os.path.join(train_data_dir, "test_truth.npy"))
    plotting.make_all_plots(data=data, truth = truth, recon=recon, forecast=forecast, out_dir=output_dir)



if __name__ == "__main__":
    main()

    