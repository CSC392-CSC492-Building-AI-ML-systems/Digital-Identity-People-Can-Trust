"""Standalone Federated Learning (FedAvg) and Differential Privacy Flask Server.

Exposes HTTP endpoints for fetching the global model at session start and
submitting DP-sanitized weight deltas during or at the end of a session.
"""

from pathlib import Path
import sys
from flask import Flask, request, jsonify

_MODEL_VF_DIR = str(Path(__file__).resolve().parents[1])
_FED_DIR = str(Path(__file__).resolve().parent)
for _p in (_MODEL_VF_DIR, _FED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import configs as cfg
from federated_dp import FederatedAggregator, deserialize_tensor_dict

fl_app = Flask(__name__)
AGGREGATOR = FederatedAggregator(
    checkpoint_path=cfg.DEFAULT_CHECKPOINT_PATH,
    aggregation_round_size=cfg.FL_AGGREGATION_ROUND_SIZE,
)


@fl_app.route(cfg.FL_MODEL_ENDPOINT, methods=["GET"])
def get_global_model():
    """Return current global model weights, normalization vectors, and round number.

    :return: JSON response with global model payload and HTTP 200 status.
    :rtype: tuple[flask.Response, int]
    """
    return jsonify(AGGREGATOR.get_global_payload()), 200


@fl_app.route(cfg.FL_SUBMIT_ENDPOINT, methods=["POST"])
def submit_dp_update():
    """Accept a DP-sanitized weight delta and buffer or aggregate it via FedAvg.

    :return: JSON status response and HTTP status code.
    :rtype: tuple[flask.Response, int]
    """
    data = request.json or {}
    raw_delta = data.get("dp_weight_delta")
    windows_contributed = data.get("windows_contributed", 1)

    if not raw_delta or not isinstance(raw_delta, dict):
        return jsonify({"error": "Missing or invalid dp_weight_delta payload."}), 400

    try:
        delta_tensors = deserialize_tensor_dict(raw_delta)
    except Exception as exc:
        return jsonify({"error": f"Failed to deserialize tensor delta: {exc}"}), 400

    result = AGGREGATOR.submit_client_delta(
        delta_tensors, windows_contributed=windows_contributed
    )
    return jsonify(result), 200


if __name__ == "__main__":
    fl_app.run(host=cfg.FL_SERVER_HOST, port=cfg.FL_SERVER_PORT, debug=True)