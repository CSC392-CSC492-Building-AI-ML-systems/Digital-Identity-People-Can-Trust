# ChatGPT was used as a co-developer to write much of this code and brainstorm ideas.
"""Prepare session data for training the handoff model."""

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from machine_learning.features import make_features


def load_sessions(path):
    """Group CSV rows by session and put each session in step order."""
    data = pd.read_csv(path).sort_values(["session_id", "step"])
    return [group.to_dict("records") for _, group in data.groupby("session_id")]


def split_sessions(sessions, rng):
    """Split sessions 70/15/15 before making overlapping windows."""
    sessions = list(sessions)
    rng.shuffle(sessions)
    train_end = int(len(sessions) * 0.7)
    validation_end = int(len(sessions) * 0.85)
    return sessions[:train_end], sessions[train_end:validation_end], sessions[validation_end:]


def make_examples(sessions):
    """Label each window by whether a handoff has happened at its final step."""
    features, labels = [], []
    for rows in sessions:
        switch = rows[0].get("switch_step")
        switch_step = int(switch) if switch else None
        for step, values in make_features(rows):
            features.append(values)
            labels.append(float(switch_step is not None and step >= switch_step))
    if not features:
        raise ValueError("No complete windows found in this split")
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def get_session_splits(data_dir=None, seed=0):
    """Return the same session splits for training and evaluation."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    data_dir = Path(data_dir)
    rng = random.Random(seed)
    normal = split_sessions(load_sessions(data_dir / "sessions.csv"), rng)
    handoff = split_sessions(load_sessions(data_dir / "handoff_sessions.csv"), rng)
    return [normal[i] + handoff[i] for i in range(3)]


def make_loaders(data_dir=None, batch_size=32, seed=0):
    """Return train, validation and test loaders with training-only scaling."""
    splits = [make_examples(sessions) for sessions in get_session_splits(data_dir, seed)]
    train_features = splits[0][0]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)

    loaders = []
    for i, (features, labels) in enumerate(splits):
        dataset = TensorDataset((features - mean) / std, labels)
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=(i == 0)))
    return (*loaders, mean, std)
