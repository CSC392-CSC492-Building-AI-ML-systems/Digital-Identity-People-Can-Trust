"""
Utility module for handling Asymmetric Zero-Knowledge Proof (ZKP) operations.
Requires: pip install cryptography

Operates statelessly without a wallet abstraction; credentials must be provided
at the time of proof generation.
"""

import secrets
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
import config

def generate_keypair():
    """
    Generates an ephemeral Elliptic Curve (SECP256R1) private/public key pair
    for the current verification session.
    
    :return: A tuple of (private_key_pem, public_key_pem) as bytes.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem

def generate_secure_nonce(length_bytes=16):
    """
    Generates a cryptographically secure random nonce for the session.
    
    :param length_bytes: Number of bytes for the nonce.
    :return: Hexadecimal string representation of the nonce.
    """
    return secrets.token_hex(length_bytes)

def validate_dob_format(dob_string):
    """
    Validates that a date of birth string matches the expected 'YYYY-MM-DD' format.
    
    :param dob_string: Date string to validate.
    :return: True if valid, False otherwise.
    """
    try:
        datetime.strptime(dob_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def compute_age(dob_string):
    """
    Computes the user's exact age based on their Date of Birth.
    
    :param dob_string: Date of birth in 'YYYY-MM-DD' format.
    :return: Integer representing age in years.
    """
    dob = datetime.strptime(dob_string, "%Y-%m-%d")
    today = datetime.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def format_w3c_vc(user_id, signature_hex, public_key_pem):
    """
    Wraps the cryptographic proof in a W3C Verifiable Credential JSON format.
    
    :param user_id: Identifier for the user.
    :param signature_hex: The ECDSA signature.
    :param public_key_pem: The public key used for verification.
    :return: A dictionary representing the Verifiable Credential.
    """
    return {
        "@context": [
            "https://www.w3.org/2018/credentials/v1",
            "https://www.w3.org/2018/credentials/examples/v1"
        ],
        "type": ["VerifiableCredential", "AgeVerificationCredential"],
        "issuer": "urn:did:zkp_system",
        "issuanceDate": datetime.now(timezone.utc).isoformat(),
        "credentialSubject": {
            "id": f"urn:did:user:{user_id}",
            f"age_verified_{config.LEGAL_AGE}_plus": True
        },
        "proof": {
            "type": "EcdsaSecp256r1Signature2019",
            "verificationMethod": public_key_pem.decode('utf-8'),
            "proofValue": signature_hex
        }
    }

def create_zkp(user_id, dob_string, nonce, private_key_pem):
    """
    Generates an asymmetric cryptographic proof (ECDSA signature) that a user meets the age requirement.
    
    :param user_id: Identifier for the user.
    :param dob_string: Date of birth in 'YYYY-MM-DD' format.
    :param nonce: Secure challenge string provided for the session.
    :param private_key_pem: The private key (bytes) used to sign the proof.
    :return: Hex string signature or W3C VC dict (based on config), or None if underage.
    """
    if not validate_dob_format(dob_string):
        return None

    age = compute_age(dob_string)
    if age < config.LEGAL_AGE:
        return None
    
    message = f"{user_id}:age_verified_{config.LEGAL_AGE}_plus:{nonce}".encode('utf-8')
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    
    signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    signature_hex = signature.hex()

    if config.USE_W3C_VC:
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return format_w3c_vc(user_id, signature_hex, public_pem)
        
    return signature_hex

def check_zkp(user_id, nonce, proof_payload, public_key_pem):
    """
    Verifies the asymmetric mathematical proof provided by the client.
    Handles both raw hex signatures and W3C Verifiable Credential dictionaries.
    
    :param user_id: Identifier for the user.
    :param nonce: The one-time secure challenge string issued to the user.
    :param proof_payload: The cryptographic proof (hex string or W3C VC dict).
    :param public_key_pem: The user's public key (bytes) used for verification.
    :return: True if valid, False otherwise.
    """
    if isinstance(proof_payload, dict) and "proof" in proof_payload:
        signature_hex = proof_payload["proof"].get("proofValue")
    else:
        signature_hex = proof_payload

    if not signature_hex or not isinstance(signature_hex, str):
        return False

    expected_message = f"{user_id}:age_verified_{config.LEGAL_AGE}_plus:{nonce}".encode('utf-8')
    
    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
        signature = bytes.fromhex(signature_hex)
        public_key.verify(signature, expected_message, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError):
        return False