"""Local training on one client's private session shard."""

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from machine_learning.dataset import make_examples
from machine_learning.model import HandoffModel, LogisticRegression


def build_model(model_type):
    if model_type == "logistic":
        return LogisticRegression()
    return HandoffModel()


def train_local(model_type, global_weights, sessions, mean, std,
                epochs, batch_size, lr):
    """Start from the server weights, train on local data, send weights back.

    Returns (state_dict, num_examples). The server only needs the weights and
    how many examples this client used, never the raw rows.
    """
    features, labels = make_examples(sessions)
    scaled = (features - mean) / std
    loader = DataLoader(
        TensorDataset(scaled, labels),
        batch_size=batch_size,
        shuffle=True,
    )

    model = build_model(model_type)
    model.load_state_dict(copy.deepcopy(global_weights))

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model.state_dict(), len(labels)
