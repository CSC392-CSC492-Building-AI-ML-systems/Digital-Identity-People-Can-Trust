# Utilities Documentation



## ZKP Utilities Documentation



This module (`zkp_util.py`) provides stateless utility functions for generating and verifying Asymmetric Zero-Knowledge Proofs (ZKPs) using Elliptic Curve Cryptography (ECDSA). 



### Architecture Updates

1. Centralized Configuration: Variables like `LEGAL_AGE` and format toggles have been extracted to `config.py` to satisfy architectural upgrade requirements.

2. Stateless Proofs (No Wallet): The abstraction of a persistent client wallet has been removed. Users must generate an ephemeral keypair and provide their Date of Birth (show ID) at the exact time of verification.

3. W3C Verifiable Credentials: The system now supports standard W3C VC JSON formatting. The 3rd party can easily parse standard identity tags without altering the underlying ECDSA verification math.



### Configuration (`config.py`)

- `LEGAL_AGE`: Defines the mathematical threshold for proof generation. 

- `USE_W3C_VC`: Boolean toggle. If `True`, `create_zkp()` returns a formatted JSON dictionary. If `False`, it returns a raw hex signature string.



### Core Functions

- `generate_keypair()`: Generates an ephemeral SECP256R1 keypair.

- `generate_secure_nonce()`: Generates a hex challenge string.

- `create_zkp(user_id, dob_string, nonce, private_key_pem)`: Signs the age-verification payload using the private key. Automatically blocks execution (returns `None`) if the `dob_string` calculates to an age below `LEGAL_AGE`.

- `check_zkp(user_id, nonce, proof_payload, public_key_pem)`: Designed to be run by the 3rd-party platform. It accepts either the raw signature string or the W3C VC dictionary, extracts the payload, and verifies the signature using the provided public key.



### Testing

Ensure `cryptography` is installed, then run the test suite to verify configuration toggles, keypair validity, and age restrictions:

`python -m unittest tests.zkp_util_test -v`





## Environment Setup & Testing Execution



### Virtual Environment Initialization

To isolate the `cryptography` dependency from your global Python installation, run this module within a virtual environment.



1. Open your terminal and navigate to the `zkp` directory.

2. Create the virtual environment: `python -m venv venv`

3. Activate the environment:

   - Windows: `venv\Scripts\activate`

   - macOS/Linux: `source venv/bin/activate`

4. Install required packages: `pip install cryptography`



### Handling the Directory Structure for Tests

This module is located in a `zkp` directory, with tests segregated into a `zkp/tests` subfolder. Because the `zkp` folder is not the project root, running the test script directly from inside the `tests` folder will cause Python to fail to find `zkp_util.py` and `config.py`.



Correct Testing Method:

To ensure Python correctly resolves the parent directory imports, you must remain in the `zkp` directory and run the test file as a module. 



# 1. Ensure you are in the 'zkp' folder

cd zkp



# 2. Run the test suite using the module flag

python -m unittest tests.zkp_util_test -v





## Identity Verification Utility Documentation



This module (`user_identaty_util.py`) provides a graphical interface to securely capture and extract user identity data (First Name, Last Name, and Date of Birth) directly from physical ID documents. It currently supports Canadian Driver's Licenses by scanning the AAMVA-standard PDF417 barcode on the back of the card.



### Dependencies

Ensure your virtual environment is active and install the required packages:

`pip install opencv-python Pillow zxing-cpp`



*(Note for macOS: If you encounter camera access issues, ensure your terminal has Camera permissions enabled in System Settings > Privacy & Security > Camera).*



### Core Functions



1. `get_identaty(id_type)`

   - The main entry point for the module.

   - Arguments: `id_type` (string). Currently only supports `"drivers_licance"`.

   - Returns: A dictionary containing `first_name`, `last_name`, and `birth_date` (formatted as `YYYY-MM-DD`). 

   - Raises: `ValueError` if an unsupported ID type is requested.



2. `_parse_aamva_barcode(barcode_text)`

   - A private helper function that uses Regular Expressions (Regex) to extract the AAMVA payload markers (`DAC`/`DCT` for First Name, `DCS` for Last Name, `DBB` for DOB) from the raw `zxing-cpp` output.



3. `_get_from_drivers_license()`

   - A private function that triggers the Tkinter GUI and OpenCV camera feed.

   - Hardware Preprocessing: It automatically converts the camera frame to grayscale and applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to mitigate glare from the card's glossy lamination before passing it to the barcode reader.



### How to Use the UI

1. Run the script. A window will pop up and force itself to the foreground.

2. Hold the FRONT of the license to the camera and press the Spacebar.

3. Flip the license and hold the BACK (where the dense 2D barcode is) to the camera. Keep it 4-6 inches away to allow the webcam to focus.

4. Press the Spacebar to scan. The window will close and return the data.



### Developer Override

Relying on hardware cameras during active development or live presentations can be frustrating. 

- If you press 'D' on your keyboard while the GUI is open, it will instantly close and inject a valid mock user payload (`TestUser Ontario, 1999-05-15`). This allows you to test the rest of your ZKP pipeline without needing to physically scan an ID every time.



### Testing Execution

The test suite (`tests/user_identaty_util_test.py`) isolates the Regex parser and routing logic. It mocks the GUI function to ensure your automated test pipeline does not hang waiting for a physical camera input.



Run the tests from the parent `zkp` directory:

`python -m unittest tests.user_identaty_util_test -v`