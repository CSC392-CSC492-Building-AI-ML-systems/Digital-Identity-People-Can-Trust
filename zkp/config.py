# config.py
"""
Configuration constants for the ZKP Age Verification system.
"""

# Minimum age required to generate a valid proof
LEGAL_AGE = 19

# Toggle for wrapping the cryptographic proof in a standard W3C Verifiable Credential JSON format
USE_W3C_VC = True

# --- New Identity UI Configurations ---
# 1/4 column background color (Hex code)
UI_SIDEBAR_COLOR = "#2C3E50" 

# Text to display on the 1/4 column
UI_SIDEBAR_TEXT = (
    "CANADIAN DRIVER'S LICENSE SCANNER\n\n"
    "1. Hold the FRONT of your license to the camera.\n"
    "2. Press 'SPACE' to capture.\n\n"
    "3. Flip to the BACK of your license.\n"
    "4. Press 'SPACE' to capture."
)