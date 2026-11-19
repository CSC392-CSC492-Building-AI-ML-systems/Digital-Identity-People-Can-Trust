"""Dual-Head 1D Convolutional Neural Network for kinematic age and handoff detection.

``KinematicDualModel`` encodes a sequence of kinematic steps into a fixed-size
behavioral embedding using 1D convolutions, ``GroupNorm``, combined average+max
pooling, and ``Dropout`` regularization. Two classification heads operate on the
embeddings:

- ``minor_head`` detects absolute minor-like motor kinematics from a single window.
- ``handoff_head`` compares a baseline embedding against a current embedding to
  detect mid-session behavioral shifts.
"""

from pathlib import Path
import sys
import torch
from torch import nn

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
if _MODEL_VF_DIR not in sys.path:
    sys.path.insert(0, _MODEL_VF_DIR)

import configs as cfg


class KinematicDualModel(nn.Module):
    """1D-CNN encoder with an Absolute Minor Head and a Siamese Handoff Head.

    ``GroupNorm`` is used instead of ``BatchNorm1d`` so single-sample or small-batch
    local updates in differential privacy and federated learning remain stable.

    :param in_channels: Number of input kinematic channels per time step.
    :type in_channels: int
    :param embedding_dim: Dimensionality of the latent behavioral embedding.
    :type embedding_dim: int
    :param dropout_rate: Dropout probability applied during training.
    :type dropout_rate: float
    """

    def __init__(
        self,
        in_channels=cfg.IN_CHANNELS,
        embedding_dim=cfg.EMBEDDING_DIM,
        dropout_rate=cfg.DROPOUT_RATE,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(
                in_channels,
                cfg.CONV1_OUT_CHANNELS,
                kernel_size=cfg.CONV1_KERNEL_SIZE,
                padding=cfg.CONV1_PADDING,
            ),
            nn.GroupNorm(cfg.CONV1_GROUPS, cfg.CONV1_OUT_CHANNELS),
            nn.ReLU(),
            nn.Conv1d(
                cfg.CONV1_OUT_CHANNELS,
                cfg.CONV2_OUT_CHANNELS,
                kernel_size=cfg.CONV2_KERNEL_SIZE,
                padding=cfg.CONV2_PADDING,
            ),
            nn.GroupNorm(cfg.CONV2_GROUPS, cfg.CONV2_OUT_CHANNELS),
            nn.ReLU(),
        )

        # Combine average pooling (overall pacing) and max pooling (peak jerk/velocity spikes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(cfg.CONV2_OUT_CHANNELS * 2, embedding_dim),
            nn.ReLU(),
        )

        self.minor_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, cfg.MINOR_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.MINOR_HEAD_HIDDEN_DIM, 1),
        )

        # Input concatenates [z_base, z_curr, |z_curr - z_base|], giving 3 * embedding_dim features
        self.handoff_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 3, cfg.HANDOFF_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(cfg.HANDOFF_HEAD_HIDDEN_DIM, 1),
        )

    def encode_window(self, window: torch.Tensor) -> torch.Tensor:
        """Encode a kinematic window tensor into a latent embedding.

        :param window: Tensor of shape ``(T, C)`` or ``(B, T, C)``.
        :type window: torch.Tensor
        :return: Embedding tensor of shape ``(B, embedding_dim)``.
        :rtype: torch.Tensor
        """
        if window.dim() == 2:
            window = window.unsqueeze(0)
        # Conv1d expects (batch, channels, sequence_length)
        features = self.conv_blocks(window.transpose(1, 2))
        pooled = torch.cat(
            [self.avg_pool(features).squeeze(-1), self.max_pool(features).squeeze(-1)],
            dim=-1,
        )
        return self.projection(pooled)

    def classify_minor_from_embedding(self, z_curr: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for the minor classification head.

        :param z_curr: Current window embedding tensor of shape ``(B, embedding_dim)``.
        :type z_curr: torch.Tensor
        :return: Raw logit tensor of shape ``(B,)``.
        :rtype: torch.Tensor
        """
        return self.minor_head(z_curr).squeeze(-1)

    def classify_handoff_from_embeddings(
        self, z_base: torch.Tensor, z_curr: torch.Tensor
    ) -> torch.Tensor:
        """Compute raw logits for the handoff head from baseline and current embeddings.

        :param z_base: Baseline embedding tensor of shape ``(B, embedding_dim)``.
        :type z_base: torch.Tensor
        :param z_curr: Current embedding tensor of shape ``(B, embedding_dim)``.
        :type z_curr: torch.Tensor
        :return: Raw logit tensor of shape ``(B,)``.
        :rtype: torch.Tensor
        """
        diff = torch.abs(z_curr - z_base)
        combined = torch.cat([z_base, z_curr, diff], dim=-1)
        return self.handoff_head(combined).squeeze(-1)

    def forward(self, base_window: torch.Tensor, curr_window: torch.Tensor):
        """Run both encoder and classification heads on paired windows.

        :param base_window: Baseline window tensor of shape ``(B, T, C)``.
        :type base_window: torch.Tensor
        :param curr_window: Current window tensor of shape ``(B, T, C)``.
        :type curr_window: torch.Tensor
        :return: Tuple of ``(logit_minor, logit_handoff)``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        z_base = self.encode_window(base_window)
        z_curr = self.encode_window(curr_window)
        logit_minor = self.classify_minor_from_embedding(z_curr)
        logit_handoff = self.classify_handoff_from_embeddings(z_base, z_curr)
        return logit_minor, logit_handoff