## ZKP Utilities Documentation

This module (`zkp_util.py`) provides stateless utility functions for generating and verifying Asymmetric Zero-Knowledge Proofs (ZKPs) using Elliptic Curve Cryptography (ECDSA). 

### Architecture Updates
1. Centralized Configuration: Variables like `LEGAL_AGE` and format toggles have been extracted to `config.py` to satisfy architectural upgrade requirements[cite: 3].
2. Stateless Proofs (No Wallet): The abstraction of a persistent client wallet has been removed[cite: 4]. Users must generate an ephemeral keypair and provide their Date of Birth (show ID) at the exact time of verification.
3. W3C Verifiable Credentials: The system now supports standard W3C VC JSON formatting. The 3rd party can easily parse standard identity tags without altering the underlying ECDSA verification math.

### Configuration (`config.py`)
* `LEGAL_AGE`: Defines the mathematical threshold for proof generation. 
* `USE_W3C_VC`: Boolean toggle. If `True`, `create_zkp()` returns a formatted JSON dictionary. If `False`, it returns a raw hex signature string.

### Core Functions
* `generate_keypair()`: Generates an ephemeral SECP256R1 keypair.
* `generate_secure_nonce()`: Generates a hex challenge string.
* `create_zkp(user_id, dob_string, nonce, private_key_pem)`: Signs the age-verification payload using the private key. Automatically blocks execution (returns `None`) if the `dob_string` calculates to an age below `LEGAL_AGE`.
* `check_zkp(user_id, nonce, proof_payload, public_key_pem)`: Designed to be run by the 3rd-party platform. It accepts either the raw signature string or the W3C VC dictionary, extracts the payload, and verifies the signature using the provided public key.

### Testing
Ensure `cryptography` is installed, then run the test suite to verify configuration toggles, keypair validity, and age restrictions:
`python -m unittest zkp_util_test.py -v`

## Environment Setup & Testing Execution

### Virtual Environment Initialization
To isolate the `cryptography` dependency from your global Python installation, run this module within a virtual environment.

1. Open your terminal and navigate to the `zkp` directory.
2. Create the virtual environment: `python -m venv venv`
3. Activate the environment:
   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`
4. Install required packages: `pip install cryptography`

### Handling the Directory Structure for Tests
This module is located in a `zkp` directory, with tests segregated into a `zkp/tests` subfolder. Because the `zkp` folder is not the project root, running the test script directly from inside the `tests` folder will cause Python to fail to find `zkp_util.py` and `config.py`.

**Correct Testing Method:**
To ensure Python correctly resolves the parent directory imports, you must remain in the `zkp` directory and run the test file as a module. 

```bash
# 1. Ensure you are in the 'zkp' folder
cd zkp

# 2. Run the test suite using the module flag
python -m unittest tests.zkp_util_test -v