"""Server-side FedAvg: mix client weights into one global model."""

import copy

import torch


def fedavg(client_weights, sample_counts):
    """Average state dicts. Bigger clients count more.

    client_weights: list of state_dicts
    sample_counts: how many training rows each client used
    """
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("no training examples from any client")

    averaged = copy.deepcopy(client_weights[0])
    for key in averaged:
        averaged[key] = torch.zeros_like(averaged[key], dtype=torch.float32)

    for weights, n in zip(client_weights, sample_counts):
        scale = n / total
        for key in averaged:
            averaged[key] += weights[key].float() * scale

    return averaged
