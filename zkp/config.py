"""
Configuration constants for the ZKP Age Verification system.
Extracts hardcoded variables into a centralized configuration file.
"""

# Minimum age required to generate a valid proof
LEGAL_AGE = 19

# Toggle for wrapping the cryptographic proof in a standard W3C Verifiable Credential JSON format
USE_W3C_VC = True