# Detailed Reasoning: Code Changes & Why This Version Is Significantly Better

## 1. Shift from Tabular Bet Averages to Raw Kinematic Time-Series
- **Original Code (`features.py`, `generate_data.py`, `personas.py`):** Computed six
  session/bet-level averages (`log(gap_s)`, `bet / balance_before`, `mouse_speed`,
  `straightness`, `click_ms`, `misclick`) over 16 baseline actions vs. 8 recent actions
 [cite: 3, 8, 9].
- **Why It Was Limited:** Averaging over 8–16 actions smooths out high-frequency motor jitter,
  micro-corrections, and acceleration/jerk bursts. Furthermore, those features depended on
  gambling bet/balance fields rather than pure cursor kinematics.
- **New Code (`features.py`, `preprocess_tacd.py`, `model.py`):** Extracts 5 kinematic
  features per step (`velocity`, `acceleration`, `jerk`, `angle`, `log_dt`) with optional
  viewport normalization (`width x height -> 1920x1080`) over `100`-step sequences and
  processes them with a 1D-CNN (`Conv1d` + `GroupNorm` + `ReLU` + `AdaptiveAvgPool1d`).
  1D convolutions preserve local temporal order and fine-grained trajectory shape.

## 2. Solving the "Child Starts" & Drift Vulnerabilities via Dual-Head Architecture
- **Original Code (`dataset.py`, `model.py`, `inference.py`):** Only predicted whether a
  switch occurred relative to the first 16 actions (`LogisticRegression` or a 1-hidden-layer
  32-unit MLP over 18 summary numbers)[cite: 1, 4, 5]. If a minor started the session and
  played consistently, no switch occurred and the model returned low risk. Conversely, long
  sessions with natural adult fatigue could drift from the static initial baseline.
- **New Code (`model.py`, `inference_engine.py`):** Uses a shared encoder producing a 32-D
  embedding `z_t` with two heads:
  - `minor_head` evaluates `z_curr` directly on every window (including the very first
    window) to catch a child starting the session.
  - `handoff_head` evaluates `[z_base, z_curr, |z_curr - z_base|]` to catch mid-session
    handoffs, while `baseline_embedding` slowly adapts via Exponential Moving Average
    (`BASELINE_EMA_ALPHA = 0.05`) on verified normal windows to tolerate gradual fatigue.
  - Strike counting (`MAX_ALERT_STRIKES = 2`) reduces single-window false positives.

## 3. Grounding Adult Data in TACD & Biomechanical Minor Augmentation
- **Original Code (`generate_data.py`, `personas.py`):** Invented all adult and minor
  parameters from hardcoded assumptions (`BASE`, `MOTOR_MOD`, `NEW_MOD`)[cite: 8, 9].
- **New Code (`preprocess_tacd.py`, `augment_minors.py`, `dataset.py`):** Uses real human
  crowdsourced mouse logs and XML browser metadata from The Attentive Cursor Dataset
  across age bins 1–9 (including older adults in bins 8–9 so senior motor patterns are
  represented)[cite: 11]. Because crowdsourced datasets lack underage subjects[cite: 11],
  `perturb_trajectory_to_minor` synthesizes minor-like trajectories by applying
  biomechanical perturbations (pacing shifts, lateral jitter, and sinusoidal overshoot
  oscillations) to real coordinate streams, and `make_synthetic_minor_window_tensor`
  provides negative anchors during local updates to prevent catastrophic forgetting.
- **Participant-Level Splitting (`dataset.py`):** Splits sessions by participant before
  windowing so windows from the same user never leak across train, validation, and test sets.

## 4. Eliminating the 100-Event Blind Spot in Sliding Windows
- **Original Code (`app.py`):** Called `session["feature_window"].clear()` after every 100
  events[cite: 10], forcing the system to wait another full 100 events before making the
  next check.
- **New Code (`app.py`, `configs.py`):** Pops only `STRIDE` (`25`) items from the left of
  the `deque` (`popleft()`), evaluating every 25 new events while still evicting old
  entries from memory.

## 5. Adding Local Differential Privacy (DP-SGD) & Microservice Federated Learning
- **Original Federated Code (`clients.py`, `partition.py`, `server.py`, `train_fl.py`):**
  Performed offline shard training and returned raw `model.state_dict()` weights without
  gradient clipping or differential privacy noise[cite: 13, 14, 15, 16].
- **New Code (`federated_dp.py`, `fl_server.py`, `inference_engine.py`, `app.py`):**
  - Isolates federated coordination in a separate Flask app (`fl_server.py`) queried at
    session start (`GET /api/fl/model`) and submitted to during/at the end of sessions
    (`POST /api/fl/submit`).
  - Computes local updates on a cloned model in RAM, clips parameter gradients to
    `DP_MAX_GRAD_NORM` (`1.0`), and injects Gaussian noise scaled by
    `DP_NOISE_MULTIPLIER * DP_MAX_GRAD_NORM` before sending only the weight delta.
  - Aggregates deltas using sample-weighted `FedAvg` (weighting by `windows_contributed`,
    retaining the sample-weighting strength from the original `server.py`[cite: 15]).

## 6. Centralizing Constants in `configs.py` & Removing `__init__.py`
- **Why:** Moving paths, thresholds, layer sizes, DP/FL parameters, and ports into
  `modelVF/configs.py` eliminates scattered magic values and makes tuning straightforward.
  Omitting `__init__.py` keeps the folder hierarchy minimal for a compact codebase while
  explicit `sys.path` setup in each entry module ensures clean imports without package boilerplate.