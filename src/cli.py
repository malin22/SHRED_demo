
import argparse, os, numpy as np
from .generate_data import generate_lorenz, save_numpy, load_numpy
from .train import train as train_fn
from .plotting import make_all_plots
from pyshred import SHREDEngine

def main():
    parser = argparse.ArgumentParser(prog="lorenz_shred")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("gen-data")
    p_gen.add_argument("--T", type=float, default=50.0)
    p_gen.add_argument("--dt", type=float, default=0.01)
    p_gen.add_argument("--out", type=str, default="data/lorenz.npy")

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--poly-order", type=int, default=3)
    p_train.add_argument("--proj-dim", type=int, default=4)
    p_train.add_argument("--lags", type=int, default=50)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--data", type=str, required=True)

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--data", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "gen-data":
        t, data = generate_lorenz(T=args.T, dt=args.dt)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        save_numpy(args.out, data)
        print(f"Saved: {args.out} shape={data.shape}")
    elif args.cmd == "train":
        data = load_numpy(args.data)
        out_dir, shred, engine = train_fn(
            data=data, dt=0.01, epochs=args.epochs, poly_order=args.poly_order, proj_dim=args.proj_dim, lags=args.lags
        )
        # Save a lightweight state dict for reuse (optional)
        import pickle
        with open(os.path.join(out_dir, "engine.pkl"), "wb") as f:
            pickle.dump({"shred": shred}, f)
        print(f"Training complete. Artifacts in: {out_dir}")
    elif args.cmd == "eval":
        data = load_numpy(args.data)
        # For simplicity, re-run training quickly to get an engine (could also load from pickle).
        out_dir, shred, engine = train_fn(data=data, dt=0.01, epochs=1)
        # Reconstruction over test split
        recon = engine.decode(engine.sensor_to_latent(engine.manager.test_sensor_measurements))["Lorenz"]
        forecast = np.load(os.path.join(out_dir, "forecast.npy"))
        truth = np.load(os.path.join(out_dir, "test_truth.npy"))
        print("Shapes -> truth:", truth.shape, "recon:", recon.shape, "forecast:", forecast.shape)
    elif args.cmd == "plot":
        data = load_numpy(args.data)
        out_dir, shred, engine = train_fn(data=data, dt=0.01, epochs=1)
        recon = engine.decode(engine.sensor_to_latent(engine.manager.test_sensor_measurements))["Lorenz"]
        forecast = np.load(os.path.join(out_dir, "forecast.npy"))
        plots_dir = os.path.join(out_dir, "plots")
        make_all_plots(data, recon, forecast, plots_dir)
        print(f"Saved plots to: {plots_dir}")

if __name__ == "__main__":
    main()
