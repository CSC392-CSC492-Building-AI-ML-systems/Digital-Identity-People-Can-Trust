<!-- I had an LLM write me this documentation and then I went over it for some small fixes. 
Generally, AI is pretty decent at documentation from what I've seen so I made small modifications.
-->

# Behavioural Handoff Detection Model

This component explores whether changes in gameplay behaviour can help flag a
possible user handoff during a session.

It contains two PyTorch models:

- A small neural network.
- A logistic regression baseline.

Both models return a score for a **possible handoff**. The score does not prove
that a handoff occurred or that the current user is underage.

## Files

| File | Purpose |
| --- | --- |
| `features.py` | Converts session actions into model features. |
| `model.py` | Defines the neural network and logistic regression models. |
| `dataset.py` | Loads data, splits sessions, scales features, and creates batches. |
| `train.py` | Trains a model and saves its best validation-loss checkpoint. |
| `evaluate.py` | Evaluates alerts on validation or test sessions. |
| `inference.py` | Loads a trained model and scores the latest session window. |
| `requirements.txt` | Lists the Python dependencies. |

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r machine_learning/requirements.txt
```

Training and evaluation expect these files:

```text
machine_learning/data/sessions.csv
machine_learning/data/handoff_sessions.csv
```

The CSVs and trained checkpoints may need to be obtained separately if they
aren't included in your checkout.

## Model input

Inference expects a list of actions from **one session**, ordered from oldest to
newest. Each action needs these fields:

| Field | Meaning |
| --- | --- |
| `gap_s` | Seconds since the previous bet; must be positive. |
| `bet` | Amount wagered. |
| `balance_before` | Account balance immediately before the bet; must be positive. |
| `mouse_speed` | Mouse speed in pixels per second. |
| `straightness` | Cursor-path straightness, from 0 to 1. |
| `click_ms` | Click duration in milliseconds. |
| `misclick` | `1` if the click missed, otherwise `0`. |

The telemetry collection code must define and calculate these fields
consistently with the data used for training. In particular, the team needs
to agree on how mouse paths and misclicks are measured.

## Feature extraction

The model uses:

1. The first **16 actions** as the baseline.
2. The latest **8 actions** as the recent window.

For each group, the code calculates six features:

- Average log time between bets.
- Average bet as a fraction of balance.
- Average mouse speed.
- Average cursor-path straightness.
- Average click duration.
- Misclick rate.

The model input contains the six baseline values, six recent values, and six
differences between them: **18 values in total**.

Before inference, each feature is scaled using the training-set mean and
standard deviation saved in the checkpoint.

## Running inference

```python
from machine_learning.inference import load_model, score_session

# Load once and reuse for subsequent scores.
model, mean, std = load_model("machine_learning/network_model.pt")

# session_rows is a list of action dictionaries in chronological order.
score = score_session(model, mean, std, session_rows)

if score is None:
    print("Not enough actions to score yet.")
else:
    print("Possible handoff score:", score)
```

`score_session()` returns `None` until the session has at least **24 actions**.
After that, it returns a score between 0 and 1 for the latest window.

The function does not apply an alert threshold. The assurance system should
decide what action to take based on the score and its intervention policy.

## Training

From the repository root, train the neural network:

```bash
python -m machine_learning.train --model network
```

Or train logistic regression:

```bash
python -m machine_learning.train --model logistic
```

The script reports training and validation loss after each epoch. It saves a
checkpoint whenever validation loss improves. The checkpoint contains the
model type, learned weights, training scaling values, epoch, validation loss,
and data-split seed.

By default, checkpoints are saved as:

```text
machine_learning/network_model.pt
machine_learning/logistic_model.pt
```

Running training again for the same model overwrites its checkpoint.

## Evaluation

Compare thresholds on the validation set:

```bash
python -m machine_learning.evaluate machine_learning/network_model.pt --sweep
```

After choosing a threshold using validation results, evaluate it on the test
set:

```bash
python -m machine_learning.evaluate machine_learning/network_model.pt \
  --split test --threshold 0.7
```

The evaluator reports:

- Normal sessions with at least one alert.
- Handoff sessions with an alert before the switch.
- Handoff sessions with an alert after the switch.
- Mean delay among detected handoffs.

Pre-switch and post-switch alerts are reported separately: the same session can
count in both. Mean delay excludes handoffs that were not detected.

For older checkpoints without a saved split seed, provide the seed used for
training with `--seed`.

## Current limitations

- The data is synthetic. Performance reflects the generator's assumptions and
  does not establish real-world handoff or age-detection accuracy.
- Current handoff examples simulate an adult followed by a minor. More varied
  handoffs and same-person behaviour changes would make evaluation stronger.
- The model produces a possible-handoff score; it does not identify the current
  user or establish their age.
- Current inference runs in Python. Browser-compatible feature extraction,
  scaling, and inference still need to be integrated.
- The model needs 24 actions before its first score.
- Federated learning and the assurance system are separate components.