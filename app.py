"""Flask application server for identity verification, ZKP issuance, and telemetry monitoring.

Coordinates document verification sessions, session lifecycle endpoints, HMAC-SHA256
telemetry signature checks, sliding-window kinematic inference, and federated
learning synchronization.
"""

from pathlib import Path
import sys
from flask import Flask, request, jsonify
from collections import deque
import hmac
import hashlib
import secrets
import time

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
_MODEL_VF_DIR = str(Path(__file__).resolve().parent / "modelVF")
_INFERENCE_DIR = str(Path(__file__).resolve().parent / "modelVF" / "inference")
_PRE_TRAIN_DIR = str(Path(__file__).resolve().parent / "modelVF" / "pre_training")
for _p in (_PROJECT_ROOT, _MODEL_VF_DIR, _INFERENCE_DIR, _PRE_TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from zkp import user_identaty_util, zkp_util
import configs as cfg
from features import compute_kinematic_step
from inference_engine import (
    load_model,
    evaluate_session_window,
    sync_from_fl_server,
    push_dp_delta_to_fl_server,
)

app = Flask(__name__)

ML_MODEL, ML_MEAN, ML_STD = load_model()

active_sessions = {}
verification_sessions = {}


def _verification_session(session_id):
    """Retrieve an active verification session if it has not expired.

    :param session_id: Verification session identifier.
    :type session_id: str
    :return: Verification session dictionary or ``None``.
    :rtype: dict or None
    """
    session = verification_sessions.get(session_id)
    if not session or session['expires_at'] < time.time():
        verification_sessions.pop(session_id, None)
        return None
    return session


@app.route('/api/verification/sessions', methods=['POST'])
def create_verification_session():
    """Create a new document verification session with a finite TTL.

    :return: JSON containing ``session_id`` and ``expires_in`` with HTTP 201.
    :rtype: tuple[flask.Response, int]
    """
    session_id = secrets.token_urlsafe(24)
    verification_sessions[session_id] = {
        'status': 'waiting_for_photo',
        'expires_at': time.time() + cfg.VERIFICATION_TTL_SECONDS,
        'proof': None,
        'error': None,
    }
    return jsonify({'session_id': session_id, 'expires_in': cfg.VERIFICATION_TTL_SECONDS}), 201


@app.route('/api/verification/sessions/<session_id>', methods=['GET'])
def get_verification_session(session_id):
    """Return current status, proof, and error state for a verification session.

    :param session_id: Verification session identifier.
    :type session_id: str
    :return: JSON session state or 404 error.
    :rtype: tuple[flask.Response, int]
    """
    session = _verification_session(session_id)
    if not session:
        return jsonify({'error': 'Verification session expired or was not found.'}), 404
    return jsonify({'status': session['status'], 'proof': session['proof'], 'error': session['error']})


@app.route('/api/verification/sessions/<session_id>/document', methods=['POST'])
def upload_verification_document(session_id):
    """Validate an uploaded ID image, generate a ZKP for legal adults, and wipe image bytes.

    :param session_id: Verification session identifier.
    :type session_id: str
    :return: JSON verification result and HTTP status code.
    :rtype: tuple[flask.Response, int]
    """
    session = _verification_session(session_id)
    if not session:
        return jsonify({'error': 'Verification session expired or was not found.'}), 404
    if session['status'] in cfg.FINISHED_VERIFICATION_STATUSES:
        return jsonify({'error': 'This verification session is already processing or finished.'}), 409

    document = request.files.get('document')
    if not document or not document.filename:
        session.update(status='scan_failed', error='Take or choose a photo of the back of your driver’s licence.')
        return jsonify({'status': session['status'], 'error': session['error']}), 400
    if document.mimetype not in cfg.ALLOWED_DOC_MIMETYPES:
        session.update(status='scan_failed', error='Use a JPEG, PNG, or WebP image.')
        return jsonify({'status': session['status'], 'error': session['error']}), 400

    image_bytes = document.read(cfg.MAX_DOCUMENT_BYTES + 1)
    if len(image_bytes) > cfg.MAX_DOCUMENT_BYTES:
        session.update(status='scan_failed', error='The image must be 10 MB or smaller.')
        return jsonify({'status': session['status'], 'error': session['error']}), 413

    session.update(status='processing', error=None)
    try:
        scan = user_identaty_util.extract_identity_from_image(image_bytes)
    except ValueError as exc:
        session.update(status='scan_failed', error=str(exc))
        return jsonify({'status': session['status'], 'error': session['error']}), 400
    except RuntimeError as exc:
        session.update(status='scan_failed', error=str(exc))
        return jsonify({'status': session['status'], 'error': session['error']}), 503
    except Exception:
        app.logger.exception('Document scan failed')
        session.update(status='scan_failed', error='The server could not process this photo. Please try again.')
        return jsonify({'status': session['status'], 'error': session['error']}), 500
    finally:
        # Discard raw document bytes immediately after extraction
        image_bytes = b''

    if scan['status'] == 'not_found':
        session.update(status='scan_failed', error='We couldn’t read the photo. Please try again.')
        return jsonify({'status': session['status'], 'error': session['error']}), 422
    if scan['status'] == 'wrong_barcode':
        session.update(status='wrong_barcode', error='We couldn’t read your driver’s licence. Please try again.')
        return jsonify({'status': session['status'], 'error': session['error']}), 422
    if scan['status'] == 'unsupported':
        session.update(status='unsupported', error='A PDF417 barcode was read, but it does not contain a supported AAMVA date of birth.')
        return jsonify({'status': session['status'], 'error': session['error']}), 422

    user_id = secrets.token_hex(8)
    private_key, _ = zkp_util.generate_keypair()
    proof = zkp_util.create_zkp(user_id, scan['identity']['birth_date'], zkp_util.generate_secure_nonce(), private_key)
    if proof is None:
        session.update(status='underage', error=None)
        return jsonify({'status': 'underage'}), 403

    session.update(status='verified', proof=proof, error=None)
    return jsonify({'status': 'verified', 'proof': proof})


@app.route('/api/session/start', methods=['POST'])
def start_session():
    """Verify identity, sync global model weights from FL server, and open a session.

    :return: JSON session credentials, ZKP proof, and active FL round.
    :rtype: tuple[flask.Response, int]
    """
    global ML_MODEL, ML_MEAN, ML_STD
    _ = request.json or {}

    try:
        identity_data = user_identaty_util.get_identaty(cfg.IDENTITY_DOC_TYPE)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not identity_data:
        return jsonify({"error": "Failed to read identity document."}), 400

    user_id = secrets.token_hex(8)
    hmac_secret = secrets.token_hex(16)
    dob = identity_data.get('birth_date')

    private_pem, _ = zkp_util.generate_keypair()
    nonce = zkp_util.generate_secure_nonce()
    zkp_proof = zkp_util.create_zkp(user_id, dob, nonce, private_pem)

    if zkp_proof is None:
        return jsonify({"message": "Please stop session. User is a minor."}), 403

    ML_MODEL, ML_MEAN, ML_STD, fl_round = sync_from_fl_server(ML_MODEL, ML_MEAN, ML_STD)

    active_sessions[user_id] = {
        "status": "verified",
        "hmac_secret": hmac_secret,
        "feature_window": deque(maxlen=cfg.WINDOW_SIZE),
        "baseline_embedding": None,
        "baseline_window_norm": None,
        "dp_weight_delta": None,
        "dp_window_count": 0,
        "alert_strikes": 0,
        "fl_round": fl_round,
        "last_x": None,
        "last_y": None,
        "last_ts": None,
        "last_v": 0.0,
        "last_a": 0.0,
    }

    return jsonify({
        "message": "Session started successfully.",
        "user_id": user_id,
        "hmac_secret": hmac_secret,
        "zkp_proof": zkp_proof,
        "fl_round": fl_round,
    }), 200


@app.route('/api/telemetry/submit', methods=['POST'])
def submit_telemetry():
    """Validate HMAC signature, compute kinematics, run model inference, and slide window.

    :return: JSON pass/stop decision and HTTP status code.
    :rtype: tuple[flask.Response, int]
    """
    data = request.json or {}
    user_id = data.get('user_id')
    x = data.get('x')
    y = data.get('y')
    timestamp = data.get('timestamp')
    signature = data.get('signature')
    width = data.get('width')
    height = data.get('height')

    if user_id not in active_sessions:
        return jsonify({"message": "stop", "reason": "Invalid session"}), 401

    session = active_sessions[user_id]

    payload_str = f"x:{x}|y:{y}|ts:{timestamp}"
    expected_sig = hmac.new(
        session["hmac_secret"].encode('utf-8'),
        payload_str.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, signature):
        return jsonify({"message": "stop", "reason": "Signature mismatch"}), 403

    feat_vec, new_state, status = compute_kinematic_step(
        x, y, timestamp,
        session["last_x"], session["last_y"], session["last_ts"],
        session["last_v"], session["last_a"],
        width=width, height=height,
    )

    if status == "non_positive_dt":
        return jsonify({"message": "stop", "reason": "Replay attack or negative time delta."}), 403
    if status == "superhuman_velocity":
        return jsonify({"message": "stop", "reason": "Superhuman velocity detected."}), 403

    session.update(new_state)
    if status == "ok":
        session["feature_window"].append(feat_vec)

    if len(session["feature_window"]) == cfg.WINDOW_SIZE:
        eval_res = evaluate_session_window(ML_MODEL, ML_MEAN, ML_STD, session)

        # Advance by STRIDE so consecutive evaluations overlap without a full-window blind spot
        for _ in range(cfg.STRIDE):
            if session["feature_window"]:
                session["feature_window"].popleft()

        if eval_res["action"] != "pass":
            return jsonify({
                "message": "stop",
                "reason": eval_res["reason"],
                "prob_minor": round(eval_res["prob_minor"], 4),
                "prob_handoff": round(eval_res["prob_handoff"], 4),
            }), 403

        if session.get("dp_window_count", 0) >= cfg.DP_FLUSH_WINDOW_INTERVAL:
            push_dp_delta_to_fl_server(session)

    return jsonify({"message": "pass"}), 200


@app.route('/api/session/end', methods=['POST'])
def end_session():
    """Flush remaining DP weight deltas to the FL server and wipe session memory.

    :return: JSON termination confirmation and HTTP status code.
    :rtype: tuple[flask.Response, int]
    """
    data = request.json or {}
    user_id = data.get('user_id')

    if user_id in active_sessions:
        session = active_sessions[user_id]
        fl_status = push_dp_delta_to_fl_server(session)

        session["feature_window"].clear()
        session.clear()
        del active_sessions[user_id]
        return jsonify({
            "message": "Session successfully wiped and terminated.",
            "federated_update": fl_status,
        }), 200

    return jsonify({"error": "Session not found."}), 404


if __name__ == '__main__':
    app.run(port=cfg.APP_SERVER_PORT, debug=True)