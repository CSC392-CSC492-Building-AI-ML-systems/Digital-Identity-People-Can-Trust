from flask import Flask, request, jsonify
from collections import deque
import hmac
import hashlib
import secrets
import math
import user_identaty_util
import zkp_util

app = Flask(__name__)

# --- Configuration & Thresholds ---
WINDOW_SIZE = 100
MAX_VELOCITY_PX_MS = 50.0  # Heuristic limit for bot detection

active_sessions = {}

@app.route('/api/session/start', methods=['POST'])
def start_session():
    data = request.json or {}
    
    try:
        # Currently only supports "drivers_licance"[cite: 3]
        identity_data = user_identaty_util.get_identaty("drivers_licance")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not identity_data:
        return jsonify({"error": "Failed to read identity document."}), 400

    user_id = secrets.token_hex(8)
    hmac_secret = secrets.token_hex(16)
    dob = identity_data.get('birth_date')

    # Generate ephemeral keypair and secure nonce statelessly[cite: 4]
    private_pem, public_pem = zkp_util.generate_keypair()
    nonce = zkp_util.generate_secure_nonce()
    
    # Automatically returns None if the user's age is below LEGAL_AGE[cite: 4]
    zkp_proof = zkp_util.create_zkp(user_id, dob, nonce, private_pem)

    if zkp_proof is None:
        return jsonify({"message": "Please stop session. User is a minor."}), 403

    # Initialize secure tracking session
    active_sessions[user_id] = {
        "status": "verified",
        "hmac_secret": hmac_secret,
        "feature_window": deque(maxlen=WINDOW_SIZE),
        "last_x": None, 
        "last_y": None, 
        "last_ts": None,
        "last_v": 0.0, 
        "last_a": 0.0
    }

    return jsonify({
        "message": "Session started successfully.",
        "user_id": user_id,
        "hmac_secret": hmac_secret,
        "zkp_proof": zkp_proof
    }), 200


@app.route('/api/telemetry/submit', methods=['POST'])
def submit_telemetry():
    data = request.json or {}
    user_id = data.get('user_id')
    x = data.get('x')
    y = data.get('y')
    timestamp = data.get('timestamp')
    signature = data.get('signature')

    if user_id not in active_sessions:
        return jsonify({"message": "stop", "reason": "Invalid session"}), 401
    
    session = active_sessions[user_id]
    
    # 1. Cryptographic Validation
    payload_str = f"x:{x}|y:{y}|ts:{timestamp}"
    expected_sig = hmac.new(
        session["hmac_secret"].encode('utf-8'), 
        payload_str.encode('utf-8'), 
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected_sig, signature):
        return jsonify({"message": "stop", "reason": "Signature mismatch"}), 403

    # 2. Extract Mathematical Derivatives (Inline processing)
    last_ts = session["last_ts"]
    last_x = session["last_x"]
    last_y = session["last_y"]
    
    if last_ts is not None:
        delta_t = timestamp - last_ts
        if delta_t <= 0:
            return jsonify({"message": "stop", "reason": "Replay attack or negative time delta."}), 403
            
        delta_x = x - last_x
        delta_y = y - last_y
        
        velocity = math.sqrt(delta_x**2 + delta_y**2) / delta_t
        acceleration = (velocity - session["last_v"]) / delta_t
        jerk = (acceleration - session["last_a"]) / delta_t
        angle = math.atan2(delta_y, delta_x)
        
        if velocity > MAX_VELOCITY_PX_MS:
            return jsonify({"message": "stop", "reason": "Superhuman velocity detected."}), 403
            
        feature_vector = (velocity, acceleration, jerk, angle, delta_t)
        session["feature_window"].append(feature_vector)
        
        # Update rolling state
        session.update({
            "last_x": x, "last_y": y, "last_ts": timestamp,
            "last_v": velocity, "last_a": acceleration
        })
    else:
        # First data point in session
        session.update({"last_x": x, "last_y": y, "last_ts": timestamp})

    # 3. Ephemeral Batch Execution (Inline Model Hook)
    if len(session["feature_window"]) == WINDOW_SIZE:
        # Placeholder for SGDClassifier inference or FedAvg weight updates
        # e.g., prediction = sgd_model.predict(list(session["feature_window"]))
        model_result = "pass" 
        
        # Immediately overwrite the data to maintain data minimization
        session["feature_window"].clear()
        
        if model_result != "pass":
            return jsonify({"message": "stop", "reason": "Model rejected behavior"}), 403

    return jsonify({"message": "pass"}), 200


@app.route('/api/session/end', methods=['POST'])
def end_session():
    data = request.json or {}
    user_id = data.get('user_id')

    if user_id in active_sessions:
        # Securely wipe the arrays and destroy the state
        active_sessions[user_id]["feature_window"].clear()
        active_sessions[user_id].clear()
        del active_sessions[user_id]
        return jsonify({"message": "Session successfully wiped and terminated."}), 200
    
    return jsonify({"error": "Session not found."}), 404

if __name__ == '__main__':
    app.run(port=5432, debug=True)