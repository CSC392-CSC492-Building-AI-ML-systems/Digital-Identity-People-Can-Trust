# ChatGPT was used as a co-developer to write much of this code and brainstorm ideas.
"""Load a trained model and score recent behaviour in one session."""

import torch

from machine_learning.features import summarize
from machine_learning.model import HandoffModel, LogisticRegression


def load_model(path):
    """Load the model and training scaling values once for reuse."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    models = {"network": HandoffModel, "logistic": LogisticRegression}
    model = models[checkpoint["model_type"]]()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["std"]


def score_session(model, mean, std, rows):
    """Score the latest window; rows must be one session in action order.

    Return None until there are 16 baseline actions and 8 recent actions.
    Only behaviour fields are read; labels and switch steps are not used.
    """
    if len(rows) < 24:
        return None

    baseline = summarize(rows[:16])
    recent = summarize(rows[-8:])
    differences = [current - original for original, current in zip(baseline, recent)]
    features = torch.tensor(baseline + recent + differences, dtype=torch.float32)
    scaled = (features - mean) / std
    model.eval()
    with torch.no_grad():
        logit = model(scaled)
        return torch.sigmoid(logit).item()
