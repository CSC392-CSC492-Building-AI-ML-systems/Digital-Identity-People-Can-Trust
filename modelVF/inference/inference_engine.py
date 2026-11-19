"""Live session inference, FL weight synchronization, and local DP-SGD hooks.

Provides checkpoint loading, synchronization with the federated learning server,
sliding-window evaluation across both heads of ``KinematicDualModel``, baseline
drift adaptation, and in-memory DP-SGD weight delta accumulation.
"""

from pathlib import Path
import sys
import requests
import torch

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent.parent / "pre_training")
_FED_DIR = str(Path(__file__).resolve().parent.parent / "federated_learning")
for _p in (_MODEL_VF_DIR, _PRE_TRAIN_DIR, _FED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from model import KinematicDualModel
from federated_dp import (
    compute_dp_weight_delta,
    accumulate_session_deltas,
    serialize_tensor_dict,
    deserialize_tensor_dict,
)


def load_model(path=cfg.DEFAULT_CHECKPOINT_PATH):
    """Load model weights and feature scaling vectors from a local checkpoint.

    :param path: Path to the ``.pt`` checkpoint file.
    :type path: Path or str
    :return: Tuple of ``(model, mean, std)``.
    :rtype: tuple[KinematicDualModel, torch.Tensor, torch.Tensor]
    """
    ckpt_path = Path(path)
    model = KinematicDualModel()
    if not ckpt_path.exists():
        model.eval()
        return model, torch.zeros(cfg.IN_CHANNELS), torch.ones(cfg.IN_CHANNELS)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["std"]


def sync_from_fl_server(model, mean, std, fl_url=cfg.FL_SERVER_URL):
    """Fetch latest global weights and scaling tensors from the FL server.

    :param model: Existing model instance to update in-place.
    :type model: KinematicDualModel
    :param mean: Fallback channel mean tensor.
    :type mean: torch.Tensor
    :param std: Fallback channel standard deviation tensor.
    :type std: torch.Tensor
    :param fl_url: Base URL of the federated learning server.
    :type fl_url: str
    :return: Tuple of ``(model, mean, std, fl_round)`` where ``fl_round`` is
        ``-1`` if the server could not be reached.
    :rtype: tuple[KinematicDualModel, torch.Tensor, torch.Tensor, int]
    """
    try:
        resp = requests.get(
            f"{fl_url}{cfg.FL_MODEL_ENDPOINT}",
            timeout=cfg.FL_REQUEST_TIMEOUT_GET,
        )
        if resp.status_code == 200:
            payload = resp.json()
            state_dict = deserialize_tensor_dict(payload["model_state"])
            model.load_state_dict(state_dict)
            model.eval()
            new_mean = torch.tensor(payload["mean"], dtype=torch.float32)
            new_std = torch.tensor(payload["std"], dtype=torch.float32)
            return model, new_mean, new_std, payload.get("fl_round", 0)
    except Exception:
        pass
    return model, mean, std, -1


def push_dp_delta_to_fl_server(session, fl_url=cfg.FL_SERVER_URL):
    """Send accumulated DP-sanitized weight deltas to the FL server and reset buffer.

    :param session: Active session dictionary holding DP delta state.
    :type session: dict
    :param fl_url: Base URL of the federated learning server.
    :type fl_url: str
    :return: Response dictionary from the FL server or local skip status.
    :rtype: dict
    """
    dp_delta = session.get("dp_weight_delta")
    if not dp_delta or session.get("alert_strikes", 0) > 0:
        return {"status": "skipped_unverified_or_empty"}

    win_count = session.get("dp_window_count", 1)
    try:
        resp = requests.post(
            f"{fl_url}{cfg.FL_SUBMIT_ENDPOINT}",
            json={
                "dp_weight_delta": serialize_tensor_dict(dp_delta),
                "windows_contributed": win_count,
            },
            timeout=cfg.FL_REQUEST_TIMEOUT_POST,
        )
        session["dp_weight_delta"] = None
        session["dp_window_count"] = 0
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "fl_server_unreachable"}


def evaluate_session_window(
    model,
    mean,
    std,
    session,
    minor_threshold=cfg.MINOR_ALERT_THRESHOLD,
    handoff_threshold=cfg.HANDOFF_ALERT_THRESHOLD,
    max_strikes=cfg.MAX_ALERT_STRIKES,
):
    """Score a full kinematic window and update baseline/DP state for a session.

    :param model: Trained ``KinematicDualModel`` instance.
    :type model: KinematicDualModel
    :param mean: Channel-wise normalization mean tensor.
    :type mean: torch.Tensor
    :param std: Channel-wise normalization standard deviation tensor.
    :type std: torch.Tensor
    :param session: Active session state dictionary.
    :type session: dict
    :param minor_threshold: Probability cutoff for the minor classification head.
    :type minor_threshold: float
    :param handoff_threshold: Probability cutoff for the handoff head.
    :type handoff_threshold: float
    :param max_strikes: Consecutive anomaly windows required before stopping.
    :type max_strikes: int
    :return: Dictionary with ``action``, ``reason``, ``prob_minor``, and ``prob_handoff``.
    :rtype: dict
    """
    window_list = list(session["feature_window"])
    if len(window_list) < cfg.WINDOW_SIZE:
        return {"action": "pass", "reason": "warming_up", "prob_minor": 0.0, "prob_handoff": 0.0}

    raw_tensor = torch.tensor(window_list, dtype=torch.float32)
    scaled = (raw_tensor - mean) / std

    with torch.no_grad():
        z_curr = model.encode_window(scaled)
        prob_minor = torch.sigmoid(model.classify_minor_from_embedding(z_curr)).item()

        # First full window establishes the session baseline after checking for minor start
        if session.get("baseline_embedding") is None:
            if prob_minor >= minor_threshold:
                return {
                    "action": "stop",
                    "reason": "Minor motor kinematics detected at session start.",
                    "prob_minor": prob_minor,
                    "prob_handoff": 0.0,
                }
            session["baseline_embedding"] = z_curr.detach()
            session["baseline_window_norm"] = scaled.detach()
            return {
                "action": "pass",
                "reason": "baseline_established",
                "prob_minor": prob_minor,
                "prob_handoff": 0.0,
            }

        z_base = session["baseline_embedding"]
        prob_handoff = torch.sigmoid(
            model.classify_handoff_from_embeddings(z_base, z_curr)
        ).item()

    is_anomaly = (prob_minor >= minor_threshold) or (prob_handoff >= handoff_threshold)
    if is_anomaly:
        session["alert_strikes"] = session.get("alert_strikes", 0) + 1
        # Discard pending adult deltas when anomalous behavior is observed
        session["dp_weight_delta"] = None
        session["dp_window_count"] = 0
    else:
        session["alert_strikes"] = max(0, session.get("alert_strikes", 0) - 1)
        decay = 1.0 - cfg.BASELINE_EMA_ALPHA
        session["baseline_embedding"] = (decay * z_base + cfg.BASELINE_EMA_ALPHA * z_curr).detach()

        if session.get("baseline_window_norm") is not None:
            step_delta = compute_dp_weight_delta(
                model,
                session["baseline_window_norm"],
                scaled.detach(),
                is_minor_label=cfg.DP_VERIFIED_MINOR_LABEL,
                is_handoff_label=cfg.DP_VERIFIED_HANDOFF_LABEL,
            )
            count = session.get("dp_window_count", 0)
            session["dp_weight_delta"] = accumulate_session_deltas(
                session.get("dp_weight_delta"), step_delta, count
            )
            session["dp_window_count"] = count + 1

    if session["alert_strikes"] >= max_strikes:
        reason = (
            "Minor motor kinematics detected."
            if prob_minor >= minor_threshold
            else "Mid-session behavioral handoff detected."
        )
        return {
            "action": "stop",
            "reason": reason,
            "prob_minor": prob_minor,
            "prob_handoff": prob_handoff,
        }

    return {
        "action": "pass",
        "reason": "ok",
        "prob_minor": prob_minor,
        "prob_handoff": prob_handoff,
    }