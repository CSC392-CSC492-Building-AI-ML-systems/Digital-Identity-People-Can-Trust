"""
Client Wallet Simulator Module (Milestone 1 - D1.1).

Simulates the user's digital device and wallet. It securely stores the user's
sensitive data (like Date of Birth) locally and mathematically generates proofs
to interact with the authentication server without exposing raw data.
"""

import requests
from zkp_core import generate_proof

SERVER_URL = "http://localhost:5000/api"

class DigitalWallet:
    """
    A simulated digital wallet that handles zero-knowledge proof generation
    and server communication.
    """

    def __init__(self, user_id, dob_string):
        """
        Initializes the Digital Wallet.

        :param user_id: The unique username/identifier for the account.
        :type user_id: str
        :param dob_string: The user's Date of Birth ('YYYY-MM-DD'). This is kept strictly local.
        :type dob_string: str
        """
        self.user_id = user_id
        self.dob_string = dob_string 

    def get_challenge_nonce(self):
        """
        Requests a cryptographic challenge nonce from the backend server.

        :return: A secure hex string nonce.
        :rtype: str
        :raises Exception: If the server fails to provide a nonce.
        """
        response = requests.post(f"{SERVER_URL}/get_challenge", json={"user_id": self.user_id})
        if response.status_code == 200:
            return response.json().get("nonce")
        raise Exception("Failed to get server nonce")

    def register_account(self):
        """
        Handles the Milestone 1 Registration Flow.

        Retrieves a nonce, generates the ZKP locally, and submits the proof
        to the server to register the account.

        :return: True if registration is successful, False otherwise.
        :rtype: bool
        """
        try:
            nonce = self.get_challenge_nonce()
        except Exception as e:
            print(f"Error: {e}")
            return False
            
        proof_data = generate_proof(self.user_id, self.dob_string, nonce)
        
        if proof_data["status"] == "rejected":
            print(f"Registration Blocked: {proof_data['reason']}")
            return False
            
        payload = {
            "user_id": self.user_id,
            "proof": proof_data["proof"]
        }
        
        response = requests.post(f"{SERVER_URL}/register", json=payload)
        return response.status_code == 201
        
    def login_account(self):
        """
        Handles the Milestone 1 Login Flow (Frictionless Re-Login).

        :return: True if login is successful, False otherwise.
        :rtype: bool
        """
        try:
            nonce = self.get_challenge_nonce()
        except Exception as e:
            return False
            
        proof_data = generate_proof(self.user_id, self.dob_string, nonce)
        if proof_data["status"] == "rejected":
            return False
            
        payload = {
            "user_id": self.user_id,
            "proof": proof_data["proof"]
        }
        
        response = requests.post(f"{SERVER_URL}/login", json=payload)
        return response.status_code == 200