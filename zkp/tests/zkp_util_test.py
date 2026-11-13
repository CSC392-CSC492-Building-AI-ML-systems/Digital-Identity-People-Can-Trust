# zkp_util_test.py
"""
Rigorous test suite for asymmetric zkp_util.py functions and W3C VC toggling.
"""

import unittest
from datetime import datetime, timedelta
import config
from zkp_util import (
    generate_keypair,
    generate_secure_nonce, 
    validate_dob_format, 
    compute_age, 
    create_zkp, 
    check_zkp
)

class TestAsymmetricZKPUtils(unittest.TestCase):
    
    def setUp(self):
        self.private_pem, self.public_pem = generate_keypair()
        self.today = datetime.today()
        # Default config state for testing
        config.LEGAL_AGE = 19
        config.USE_W3C_VC = False

    def test_compute_age(self):
        # Subtract exactly 25 calendar years, handling the Feb 29 leap year edge case safely
        try:
            target_date = self.today.replace(year=self.today.year - 25)
        except ValueError:
            target_date = self.today.replace(year=self.today.year - 25, month=2, day=28)
            
        dob_25_years_ago = target_date.strftime("%Y-%m-%d")
        self.assertEqual(compute_age(dob_25_years_ago), 25)
    
    def test_create_and_check_zkp_raw_signature(self):
        config.USE_W3C_VC = False
        nonce = generate_secure_nonce()
        user = "test_user_adult"
        dob = "1990-01-01"
        
        proof = create_zkp(user, dob, nonce, self.private_pem)
        self.assertIsInstance(proof, str)
        
        is_valid = check_zkp(user, nonce, proof, self.public_pem)
        self.assertTrue(is_valid)

    def test_create_and_check_zkp_w3c_vc(self):
        config.USE_W3C_VC = True
        nonce = generate_secure_nonce()
        user = "test_user_vc"
        dob = "1990-01-01"
        
        proof = create_zkp(user, dob, nonce, self.private_pem)
        self.assertIsInstance(proof, dict)
        self.assertIn("@context", proof)
        self.assertEqual(proof["credentialSubject"]["id"], f"urn:did:user:{user}")
        
        is_valid = check_zkp(user, nonce, proof, self.public_pem)
        self.assertTrue(is_valid)

    def test_create_zkp_underage(self):
        nonce = generate_secure_nonce()
        child_dob = self.today.replace(year=self.today.year - 15).strftime("%Y-%m-%d")
        
        proof = create_zkp("child_user", child_dob, nonce, self.private_pem)
        self.assertIsNone(proof)

    def test_check_zkp_tampered_nonce(self):
        nonce = generate_secure_nonce()
        proof = create_zkp("hacker", "1985-05-05", nonce, self.private_pem)
        
        is_valid = check_zkp("hacker", "different_nonce", proof, self.public_pem)
        self.assertFalse(is_valid)

    def test_check_zkp_wrong_public_key(self):
        nonce = generate_secure_nonce()
        proof = create_zkp("alice", "1985-05-05", nonce, self.private_pem)
        
        _, wrong_public_pem = generate_keypair()
        is_valid = check_zkp("alice", nonce, proof, wrong_public_pem)
        self.assertFalse(is_valid)

if __name__ == '__main__':
    unittest.main()