# ChatGPT was used as a co-developer to write much of this code and brainstorm ideas.
"""A small model for scoring possible changes in session behaviour."""

import torch
from torch import nn


class LogisticRegression(nn.Module):
    """A linear baseline that returns one logit per feature vector."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(18, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class HandoffModel(nn.Module):
    """Take baseline, recent, and difference features and return a logit."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
