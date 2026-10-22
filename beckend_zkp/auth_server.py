"""
Authentication Server Module (Milestone 1 - D1.2 & D1.3).

This module implements a Flask-based API backend that handles user registration,
challenge generation, and Zero-Knowledge Proof verification. It grants access
only to mathematically verified credentials.
"""

from flask import Flask, request, jsonify
import secrets
from zkp_core import verify_proof

app = Flask(__name__)

# In-memory mock databases for simulation purposes
db_users = {}
db_sessions = {}

@app.route('/api/get_challenge', methods=['POST'])
def get_challenge():
    """
    API Endpoint to generate a secure challenge nonce for a user.

    This nonce must be used by the client to generate their proof, ensuring
    protection against replay attacks.

    :return: JSON response containing the generated 'nonce' or an error message.
    :rtype: flask.Response
    """
    data = request.json or {}
    user_id = data.get('user_id')
    
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
        
    nonce = secrets.token_hex(16)
    db_sessions[user_id] = nonce
    return jsonify({"nonce": nonce}), 200

@app.route('/api/register', methods=['POST'])
def register():
    """
    API Endpoint to register a new user using their ZKP credential.

    Replaces legacy ID uploads. It requires a valid user_id and a mathematical
    proof based on an active challenge nonce.

    :return: JSON response indicating success (201) or failure (400/401).
    :rtype: flask.Response
    """
    data = request.json or {}
    user_id = data.get('user_id')
    proof = data.get('proof')
    
    if user_id in db_users:
        return jsonify({"error": "User already registered"}), 400
        
    nonce = db_sessions.get(user_id)
    if not nonce:
        return jsonify({"error": "No active challenge nonce found"}), 400
        
    if verify_proof(user_id, nonce, proof):
        db_users[user_id] = {"status": "active"}
        del db_sessions[user_id] # Consume the nonce for security
        return jsonify({"message": "Registration successful, age verified."}), 201
    else:
        return jsonify({"error": "Invalid zero-knowledge proof"}), 401

@app.route('/api/login', methods=['POST'])
def login():
    """
    API Endpoint to authenticate a returning user seamlessly.

    Fulfills User Story 3 (Frictionless Re-Login). Verifies a new ZKP to 
    grant a session.

    :return: JSON response indicating successful login (200) or failure (401/404).
    :rtype: flask.Response
    """
    data = request.json or {}
    user_id = data.get('user_id')
    proof = data.get('proof')
    
    if user_id not in db_users:
        return jsonify({"error": "User not found"}), 404
        
    nonce = db_sessions.get(user_id)
    if not nonce:
        return jsonify({"error": "No active challenge nonce found"}), 400

    if verify_proof(user_id, nonce, proof):
        del db_sessions[user_id] # Consume the nonce
        return jsonify({"message": "Login successful, session granted."}), 200
    else:
        return jsonify({"error": "Invalid zero-knowledge proof"}), 401

if __name__ == '__main__':
    app.run(port=5000, debug=True)