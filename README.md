# Lorenz + PySHRED (VS Code Project)

This is a minimal, reproducible project structure that mirrors your notebook:
- Generates Lorenz data
- Trains SHRED with a custom **ProjectedLSTM** + **SINDy_Forecaster**
- Evaluates and plots results

## Quickstart

```bash
# 1) Create & activate a venv (recommended)
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Run: generate data, train, evaluate, plot
python -m lorenz_shred.cli gen-data --T 50.0 --dt 0.01 --out data/lorenz.npy
python -m lorenz_shred.cli train --data data/lorenz.npy --epochs 50 --poly-order 3 --proj-dim 4
python -m lorenz_shred.cli eval --data data/lorenz.npy
python -m lorenz_shred.cli plot --data data/lorenz.npy
```

Artifacts (metrics/models/plots) are written to the `runs/` directory.

## VS Code
This project includes a `.vscode/launch.json` for one-click runs:
- **Train Lorenz (PySHRED)**: runs `python -m lorenz_shred.cli train ...`
- **Eval & Plot**: runs evaluation and plotting.

You can tweak arguments in `lorenz_shred/cli.py` or directly in VS Code launch configs.
