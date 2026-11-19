"""Differential Privacy (DP-SGD) local update engine and Federated Averaging (FedAvg).

Provides gradient clipping and Gaussian noise injection for privacy-preserving
local weight updates, plus ``FederatedAggregator`` for aggregating session deltas
using sample-weighted ``FedAvg`` across federated rounds.
"""

from pathlib import Path
import copy
import sys
import torch
from torch import nn

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent.parent / "pre_training")
for _p in (_MODEL_VF_DIR, _PRE_TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from model import KinematicDualModel
from augment_minors import make_synthetic_minor_window_tensor


def compute_dp_weight_delta(
    global_model: KinematicDualModel,
    base_window_norm: torch.Tensor,
    curr_window_norm: torch.Tensor,
    is_minor_label: float = cfg.DP_VERIFIED_MINOR_LABEL,
    is_handoff_label: float = cfg.DP_VERIFIED_HANDOFF_LABEL,
    lr: float = cfg.DP_LOCAL_LR,
    max_grad_norm: float = cfg.DP_MAX_GRAD_NORM,
    noise_multiplier: float = cfg.DP_NOISE_MULTIPLIER,
) -> dict:
    """Compute a single L2-clipped, Gaussian-noised DP-SGD parameter delta.

    :param global_model: Reference model used to initialize a local clone.
    :type global_model: KinematicDualModel
    :param base_window_norm: Normalized baseline window tensor of shape ``(T, C)``.
    :type base_window_norm: torch.Tensor
    :param curr_window_norm: Normalized current window tensor of shape ``(T, C)``.
    :type curr_window_norm: torch.Tensor
    :param is_minor_label: Target label for the minor head (``0.0`` or ``1.0``).
    :type is_minor_label: float
    :param is_handoff_label: Target label for the handoff head (``0.0`` or ``1.0``).
    :type is_handoff_label: float
    :param lr: Local learning rate applied to sanitized gradients.
    :type lr: float
    :param max_grad_norm: Maximum L2 norm bound for gradient clipping.
    :type max_grad_norm: float
    :param noise_multiplier: Gaussian noise scale factor relative to ``max_grad_norm``.
    :type noise_multiplier: float
    :return: Dictionary mapping parameter names to CPU weight delta tensors.
    :rtype: dict[str, torch.Tensor]
    """
    local_model = copy.deepcopy(global_model)
    local_model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    b_win = base_window_norm.unsqueeze(0)
    c_win = curr_window_norm.unsqueeze(0)

    # Pair verified adult windows with a synthetic minor anchor to avoid forgetting
    if is_minor_label == cfg.DP_VERIFIED_MINOR_LABEL:
        syn_minor = make_synthetic_minor_window_tensor(curr_window_norm).unsqueeze(0)
        batch_base = torch.cat([b_win, b_win], dim=0)
        batch_curr = torch.cat([c_win, syn_minor], dim=0)
        y_m = torch.tensor(
            [cfg.DP_VERIFIED_MINOR_LABEL, cfg.DP_ANCHOR_MINOR_LABEL],
            dtype=torch.float32,
        )
        y_h = torch.tensor(
            [is_handoff_label, cfg.DP_ANCHOR_HANDOFF_LABEL],
            dtype=torch.float32,
        )
    else:
        batch_base = b_win
        batch_curr = c_win
        y_m = torch.tensor([is_minor_label], dtype=torch.float32)
        y_h = torch.tensor([is_handoff_label], dtype=torch.float32)

    local_model.zero_grad()
    logit_m, logit_h = local_model(batch_base, batch_curr)
    loss = loss_fn(logit_m, y_m) + loss_fn(logit_h, y_h)
    loss.backward()

    # Step 1: Bound sensitivity by clipping global L2 gradient norm
    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=max_grad_norm)

    # Step 2: Add calibrated Gaussian noise before converting gradients into weight deltas
    delta_dict = {}
    noise_std = noise_multiplier * max_grad_norm
    with torch.no_grad():
        for name, param in local_model.named_parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_std
                sanitized_grad = param.grad + noise
                delta_dict[name] = (-lr * sanitized_grad).cpu()

    return delta_dict


def accumulate_session_deltas(accumulated: dict, new_delta: dict, count: int) -> dict:
    """Update a running average of parameter deltas within a single session.

    :param accumulated: Existing running average delta dictionary, or ``None``.
    :type accumulated: dict[str, torch.Tensor] or None
    :param new_delta: Newly computed weight delta dictionary.
    :type new_delta: dict[str, torch.Tensor]
    :param count: Number of prior deltas already averaged into ``accumulated``.
    :type count: int
    :return: Updated running average delta dictionary.
    :rtype: dict[str, torch.Tensor]
    """
    if not accumulated:
        return {k: v.clone() for k, v in new_delta.items()}
    updated = {}
    for key in accumulated:
        updated[key] = (accumulated[key] * count + new_delta[key]) / (count + 1)
    return updated


def serialize_tensor_dict(tensor_dict: dict) -> dict:
    """Convert a dictionary of tensors into JSON-serializable nested lists.

    :param tensor_dict: Mapping of parameter names to tensors.
    :type tensor_dict: dict[str, torch.Tensor]
    :return: JSON-compatible dictionary of lists.
    :rtype: dict[str, list]
    """
    return {k: v.detach().cpu().tolist() for k, v in tensor_dict.items()}


def deserialize_tensor_dict(json_dict: dict) -> dict:
    """Convert a JSON list dictionary back into float32 PyTorch tensors.

    :param json_dict: Mapping of parameter names to nested lists.
    :type json_dict: dict[str, list]
    :return: Mapping of parameter names to float32 tensors.
    :rtype: dict[str, torch.Tensor]
    """
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in json_dict.items()}


class FederatedAggregator:
    """Server-side coordinator that buffers DP deltas and applies FedAvg.

    :param checkpoint_path: Path to the global model checkpoint file.
    :type checkpoint_path: Path or str
    :param aggregation_round_size: Number of submissions required per FedAvg round.
    :type aggregation_round_size: int
    """

    def __init__(
        self,
        checkpoint_path: Path = cfg.DEFAULT_CHECKPOINT_PATH,
        aggregation_round_size: int = cfg.FL_AGGREGATION_ROUND_SIZE,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.k_sessions = aggregation_round_size
        self.pending_submissions = []
        self.round_number = 0
        self.model = KinematicDualModel()
        self.mean = torch.zeros(cfg.IN_CHANNELS, dtype=torch.float32)
        self.std = torch.ones(cfg.IN_CHANNELS, dtype=torch.float32)
        self._load_or_init_checkpoint()

    def _load_or_init_checkpoint(self):
        """Load weights and normalization statistics from checkpoint if present."""
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt["model_state"])
            self.mean = ckpt["mean"]
            self.std = ckpt["std"]
            self.round_number = ckpt.get("fl_round", 0)
        self.model.eval()

    def get_global_payload(self) -> dict:
        """Serialize current global model weights, scaling vectors, and round count.

        :return: JSON-serializable global model payload.
        :rtype: dict
        """
        return {
            "fl_round": self.round_number,
            "model_state": serialize_tensor_dict(self.model.state_dict()),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def submit_client_delta(self, delta_tensors: dict, windows_contributed: int = 1) -> dict:
        """Record a DP delta submission and run FedAvg once the round threshold is met.

        :param delta_tensors: Parameter delta dictionary for a session.
        :type delta_tensors: dict[str, torch.Tensor]
        :param windows_contributed: Weight proportional to evaluated windows.
        :type windows_contributed: int
        :return: Status dictionary describing buffering or aggregation outcome.
        :rtype: dict
        """
        weight_count = max(1, int(windows_contributed))
        self.pending_submissions.append((delta_tensors, weight_count))
        buffered = len(self.pending_submissions)

        if buffered >= self.k_sessions:
            self._apply_fedavg()
            return {
                "status": "aggregated_new_global_model",
                "fl_round": self.round_number,
                "pending_buffer": 0,
            }

        return {
            "status": "buffered_for_fedavg",
            "fl_round": self.round_number,
            "pending_buffer": buffered,
            "required_for_round": self.k_sessions,
        }

    def _apply_fedavg(self):
        """Apply sample-weighted averaging of buffered deltas and save checkpoint."""
        total_weight = float(sum(w for _, w in self.pending_submissions))
        if total_weight <= 0:
            total_weight = float(len(self.pending_submissions))

        state = self.model.state_dict()
        with torch.no_grad():
            first_delta = self.pending_submissions[0][0]
            for name in first_delta:
                if name in state:
                    weighted_sum = torch.zeros_like(state[name], dtype=torch.float32)
                    for delta_d, w_cnt in self.pending_submissions:
                        weighted_sum += delta_d[name].float() * (float(w_cnt) / total_weight)
                    state[name] = state[name] + weighted_sum
            self.model.load_state_dict(state)

        self.pending_submissions.clear()
        self.round_number += 1
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": cfg.MODEL_TYPE_NAME,
                "model_state": self.model.state_dict(),
                "mean": self.mean,
                "std": self.std,
                "fl_round": self.round_number,
                "training": "federated_dp",
            },
            self.checkpoint_path,
        )