# Privacy-Preserving Age Verification API Documentation

This module (`app.py`) provides the core Flask REST API routing for initiating secure sessions, capturing physical identity, generating Zero-Knowledge Proofs (ZKPs), and ingesting telemetry data for continuous behavioral verification.

### Architecture Core Concepts

1. Privacy-First Data Handling: Raw `x` and `y` coordinates are not stored long-term. They are immediately converted into mathematical derivatives (velocity, acceleration, jerk) to protect user privacy.

2. Cryptographic Binding: Telemetry payloads must be signed using an ephemeral HMAC secret issued during session creation, preventing hijacking or data spoofing.

3. Delegated Session Control: If an anomaly is detected, the API returns a `"stop"` flag to the 3rd-party platform. The session remains active in memory until the 3rd-party explicitly calls the termination endpoint, leaving the final enforcement decision to them.

### Configuration

- `WINDOW_SIZE`: Defines the maximum number of telemetry feature vectors held in the in-memory queue before the model inference hook is triggered (and the queue is subsequently wiped).

- `MAX_VELOCITY_PX_MS`: A strict physical limit heuristic (pixels per millisecond). If telemetry exceeds this, it is flagged as a bot or superhuman anomaly.

### Endpoints

- `POST /api/session/start`

  - Purpose: Initializes a session and generates the ZKP.

  - Action: Triggers the local Tkinter GUI (`user_identaty_util`) to scan a physical Canadian Driver's License. 

  - Output: Returns a unique `user_id`, an `hmac_secret` for signing future requests, and the `zkp_proof` (returns `403` if the user is a minor).

- `POST /api/telemetry/submit`

  - Purpose: Ingests real-time mouse/pointer telemetry from the external application.

  - Inputs: Requires `user_id`, `x`, `y`, `timestamp`, and an HMAC `signature`.

  - Action: 

    - Validates the HMAC signature.

    - Enforces monotonic timestamp progression (prevents replay attacks).

    - Extracts behavioral features (velocity, acceleration, jerk, angle).

    - Evaluates physical heuristics (max velocity).

    - Batches data into the sliding window and triggers the machine learning model hook once full.

  - Output: Returns `"pass"` if behavior is normal, or `"stop"` with a reason if anomalous behavior is detected.

- `POST /api/session/end`

  - Purpose: Safely concludes the session.

  - Action: Cryptographically wipes the user's telemetry feature arrays and completely destroys the active session state from server memory, returning the environment to a state as if the user was never there.

### Execution

Ensure your virtual environment is active and all dependencies (`flask`, `cryptography`, `opencv-python`, etc.) are installed. 

Run the application directly from the terminal:

`python app.py`

The API will bind to `localhost` on port `5432` by default.