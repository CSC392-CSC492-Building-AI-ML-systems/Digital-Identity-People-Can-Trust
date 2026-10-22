"""
Zero-Knowledge Proof Unit Testing Suite.

This module contains exhaustive tests for the ZKP Core, ensuring that all 
mathematical generation, age validations, and server verifications work 
flawlessly and securely according to Milestone 1 constraints.
"""

import unittest
from datetime import datetime
from zkp_core import calculate_age, generate_proof, verify_proof

class TestZKPCoreLogic(unittest.TestCase):
    """
    Test cases for the underlying Zero-Knowledge Proof math and logic.
    """

    def test_calculate_age_adult(self):
        """
        Test that calculate_age accurately computes the age of an adult.
        """
        # Assuming current year is at least 2026 as per documentation
        age = calculate_age("1990-01-01")
        self.assertGreaterEqual(age, 36)

    def test_calculate_age_child(self):
        """
        Test that calculate_age accurately computes the age of a minor.
        """
        current_year = datetime.today().year
        recent_year = current_year - 10
        age = calculate_age(f"{recent_year}-01-01")
        self.assertEqual(age, 10)

    def test_generate_proof_success(self):
        """
        Test that an adult user successfully generates a proof hash.
        """
        result = generate_proof("user_1", "1995-05-05", "mock_nonce_123")
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["proof"])

    def test_generate_proof_underage_rejection(self):
        """
        Test that a child user is immediately rejected by the client generator.
        (Validates User Story 1: The Child Registration Attempt)
        """
        current_year = datetime.today().year
        child_year = current_year - 15
        result = generate_proof("child_1", f"{child_year}-05-05", "mock_nonce_123")
        self.assertEqual(result["status"], "rejected")
        self.assertIsNone(result["proof"])

    def test_verify_proof_valid(self):
        """
        Test that the server successfully verifies a legitimate proof.
        """
        user_id = "user_valid"
        nonce = "secure_nonce_999"
        proof_data = generate_proof(user_id, "1980-10-10", nonce)
        
        is_valid = verify_proof(user_id, nonce, proof_data["proof"])
        self.assertTrue(is_valid)

    def test_verify_proof_tampered_nonce(self):
        """
        Test that modifying the nonce invalidates the proof (replay attack prevention).
        """
        user_id = "user_tamper"
        original_nonce = "secure_nonce_111"
        proof_data = generate_proof(user_id, "1980-10-10", original_nonce)
        
        # Server verifies with a DIFFERENT nonce
        is_valid = verify_proof(user_id, "different_nonce_222", proof_data["proof"])
        self.assertFalse(is_valid)

    def test_verify_proof_wrong_user(self):
        """
        Test that one user's proof cannot be used for another user's session.
        """
        nonce = "nonce_shared"
        proof_data = generate_proof("alice", "1990-01-01", nonce)
        
        # Bob tries to use Alice's proof
        is_valid = verify_proof("bob", nonce, proof_data["proof"])
        self.assertFalse(is_valid)

if __name__ == '__main__':
    unittest.main()