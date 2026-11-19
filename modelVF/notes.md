# modelVF Technical Notes & Usage Guide

`modelVF` contains the complete machine learning, differential privacy, and federated learning
subsystems for kinematic mouse telemetry analysis. All constants, file paths, thresholds, and
hyperparameters are centralized in `modelVF/configs.py`.

## Directory Layout

- `configs.py`: Single source of truth for file paths, dataset filenames, viewport sizes,
  window/stride lengths, perturbation ranges, 1D-CNN layer dimensions, training defaults,
  DP-SGD clipping/noise settings, FedAvg round size, and service ports/endpoints.
- `pre_training/`:
  - `features.py`: Computes 5-channel kinematic derivatives (`velocity`, `acceleration`,
    `jerk`, `angle`, `log_dt`) from consecutive `(x, y, timestamp)` events with optional
    viewport normalization to `1920x1080` and slices streams into overlapping `(100, 5)` windows.
  - `preprocess_tacd.py`: Parses The Attentive Cursor Dataset (`participants.tsv`,
    `logs/<log_id>.csv`, and `logs/<log_id>.xml`)[cite: 11], extracts `<window>` dimensions,
    filters non-zero mouse events, and attaches demographic age bins (flagging bins `>= 8`
    as seniors)[cite: 11].
  - `augment_minors.py`: Applies developmental motor perturbations (velocity scaling,
    angular jitter, corrective overshoot oscillations, and irregular pause noise) to real
    coordinate trajectories to synthesize minor sessions and handoff pairs, plus creates
    synthetic minor anchor tensors during training/updates.
  - `dataset.py`: Splits sessions `70/15/15` by participant before constructing window pairs
    (with a synthetic fallback if real logs are unavailable), computes channel-wise `mean` and
    `std` strictly on the training split, and builds PyTorch `DataLoader` objects.
  - `model.py`: Defines `KinematicDualModel`, a 1D-CNN with `GroupNorm` that encodes a
    `(100, 5)` window into a 32-D embedding and feeds two heads:
    1. `minor_head`: Predicts whether the active window exhibits minor motor kinematics.
    2. `handoff_head`: Compares `[z_base, z_curr, |z_curr - z_base|]` to detect mid-session shifts.
  - `train.py`: Trains `KinematicDualModel` jointly on both heads using `BCEWithLogitsLoss`
    and `Adam`, saving the best checkpoint to `modelVF/network_model.pt`.
  - `evaluate.py`: Evaluates a saved checkpoint across `normal_adult`, `senior_adult`,
    `child_starts`, `adult_to_minor_handoff`, and `adult_to_adult_handoff` scenarios.
- `inference/`:
  - `inference_engine.py`: Loads checkpoints, synchronizes global model weights/scaling from
    the federated learning server, evaluates `100`-step sliding windows with strike counting
    and EMA baseline drift adaptation, and accumulates local DP-SGD weight deltas in memory.
- `federated_learning/`:
  - `federated_dp.py`: Implements per-update gradient norm clipping and Gaussian noise
    injection (`compute_dp_weight_delta`), within-session delta averaging, tensor
    serialization/deserialization, and sample-weighted `FederatedAggregator` (`FedAvg`).
  - `fl_server.py`: Standalone Flask server exposing `GET /api/fl/model` to serve global
    weights and `POST /api/fl/submit` to buffer DP deltas and run `FedAvg` updates.

## How to Use

### 1. Pre-train the Model
From the `modelVF` directory (with TACD `participants.tsv` and `logs/` in the parent directory,
or passed via `--tacd-dir`):

```bash
python3 pre_training/train.py --tacd-dir .. --epochs 10 --batch-size 64