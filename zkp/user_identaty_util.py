# user_identaty_util.py
"""
Utility module for capturing and extracting user identity data from physical documents.
Requires: pip install opencv-python Pillow zxing-cpp
"""

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import zxingcpp
except ImportError:
    zxingcpp = None

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:
    Image = None
    ImageOps = None
    UnidentifiedImageError = OSError

try:
    import tkinter as tk
    from PIL import ImageTk
except ImportError:
    tk = None
    ImageTk = None
from . import config
from io import BytesIO
import os
import re

def get_identaty(id_type):
    """
    Main routing function to get user identity based on the provided document type.
    """
    if id_type == "drivers_licance":
        return _get_from_drivers_license()
    else:
        raise ValueError(f"Unsupported identity type: {id_type}")

def _parse_aamva_barcode(barcode_text):
    """
    Parses the AAMVA-standard PDF417 barcode text to extract name and DOB using Regex.
    This bypasses all line-ending and terminal formatting bugs.
    """
    data = {
        "first_name": "Unknown",
        "last_name": "Unknown",
        "birth_date": "Unknown"
    }
    
    # Hunt for Last Name (DCS followed by uppercase letters, hyphens, or spaces)
    last_name_match = re.search(r'DCS([A-Z\-\s]+)', barcode_text)
    if last_name_match:
        data["last_name"] = last_name_match.group(1).strip()
        
    # Hunt for First Name (DAC or DCT followed by uppercase letters, hyphens, or spaces)
    first_name_match = re.search(r'(?:DAC|DCT)([A-Z\-\s]+)', barcode_text)
    if first_name_match:
        data["first_name"] = first_name_match.group(1).strip()
        
    # Hunt for Date of Birth (DBB followed by exactly 8 digits)
    dob_match = re.search(r'DBB(\d{8})', barcode_text)
    if dob_match:
        raw_date = dob_match.group(1)
        # Format YYYYMMDD to YYYY-MM-DD
        data["birth_date"] = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                
    return data

def extract_identity_from_image(image_bytes):
    """Decode a machine-readable identity-card barcode from an uploaded image.

    The bytes remain in memory only; callers must not persist the original
    document image.  The currently supported barcode format is AAMVA PDF417.
    """
    if not Image or not zxingcpp:
        raise RuntimeError('Document barcode decoder dependencies are missing.')
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded:
            if uploaded.width * uploaded.height > 25_000_000:
                raise ValueError('Image dimensions are too large.')
            image = ImageOps.exif_transpose(uploaded).convert('RGB')
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('The uploaded file is not a readable image.') from exc

    # Look specifically for PDF417 before trying every barcode format. A phone
    # photo can contain both codes; a decoded 1-D strip says nothing about
    # whether the larger PDF417 in the same photo was readable.
    decoded_pdf417 = False
    detected_pdf417 = False

    def read_pdf417(candidate):
        nonlocal decoded_pdf417, detected_pdf417
        results = zxingcpp.read_barcodes(
            candidate, formats=zxingcpp.BarcodeFormat.PDF417,
            return_errors=True,
        )
        for result in results:
            detected_pdf417 = True
            if not result.valid:
                continue
            decoded_pdf417 = True
            # Do not infer a birth date from an unfamiliar PDF417 payload.
            if re.search(r'ANSI\s*\d{6,}', result.text):
                identity = _parse_aamva_barcode(result.text)
                if identity['birth_date'] != 'Unknown':
                    return {'status': 'supported', 'identity': identity}
        return None

    found = read_pdf417(image)
    if found:
        return found

    # The dense code often occupies a small part of a full-card photo. A
    # contrast pass and overlapping crops give the detector more pixels to
    # work with without asking users to crop a private document manually.
    working = image.copy()
    working.thumbnail((3200, 3200), Image.Resampling.LANCZOS)
    enhanced = ImageOps.autocontrast(working.convert('L'), cutoff=1)
    width, height = enhanced.size
    boxes = [
        (0, 0, width * 3 // 4, height * 3 // 4),
        (width // 4, 0, width, height * 3 // 4),
        (0, height // 4, width * 3 // 4, height),
        (width // 4, height // 4, width, height),
    ]
    for candidate in [enhanced, *(enhanced.crop(box) for box in boxes)]:
        found = read_pdf417(candidate)
        if found:
            return found

    # ZXing handles quarter-turns, but a tilted phone photo can still defeat
    # PDF417 detection. Straighten plausible camera angles as a final retry.
    for angle in (-12, 12, -20, 20, -28, 28):
        straightened = enhanced.rotate(
            angle, Image.Resampling.BICUBIC, expand=True, fillcolor=255
        )
        found = read_pdf417(straightened)
        if found:
            return found

    if decoded_pdf417:
        return {'status': 'unsupported'}
    if detected_pdf417:
        return {'status': 'not_found'}
    if zxingcpp.read_barcodes(image):
        return {'status': 'wrong_barcode'}
    return {'status': 'not_found'}

def _get_from_drivers_license():
    """
    Opens a centered GUI with a live camera feed to capture and decode the 
    PDF417 barcode on the back of a Canadian Driver's License.
    """
    if not all((cv2, tk, Image, ImageTk, zxingcpp)):
        raise RuntimeError('The desktop document scanner dependencies are not installed.')
    root = tk.Tk()
    root.title("Identity Verification - Driver's License")
    
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # --- FORCE WINDOW TO FOREGROUND ---
    root.lift()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))
    root.focus_force()
    
    if os.name == 'posix':
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
    
    left_frame = tk.Frame(root, bg=config.UI_SIDEBAR_COLOR, width=200, height=window_height)
    left_frame.pack(side=tk.LEFT, fill=tk.Y)
    left_frame.pack_propagate(False)
    
    instructions = config.UI_SIDEBAR_TEXT + "\n\n(Dev: Press 'D' to mock scan)"
    info_label = tk.Label(
        left_frame, 
        text=instructions, 
        bg=config.UI_SIDEBAR_COLOR, 
        fg="white", 
        font=("Helvetica", 12, "bold"),
        wraplength=180, 
        justify="left"
    )
    info_label.pack(pady=40, padx=10)

    right_frame = tk.Frame(root, width=600, height=window_height, bg="black")
    right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
    
    vid_label = tk.Label(right_frame, bg="black")
    vid_label.pack(expand=True, fill=tk.BOTH)
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    state = {
        "step": "front",
        "front_img": None,
        "back_img": None,
        "result_data": None
    }

    def update_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img.thumbnail((600, 600), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            vid_label.imgtk = imgtk
            vid_label.configure(image=imgtk)
        
        if state["step"] != "done":
            root.after(15, update_frame)
        else:
            root.destroy()

    def capture_image(event=None):
        ret, frame = cap.read()
        if not ret:
            return
            
        if state["step"] == "front":
            state["front_img"] = frame
            info_label.config(text="FRONT CAPTURED.\n\nNow show the BACK of the license.\n\nPress 'SPACE' to capture.")
            state["step"] = "back"
            
        elif state["step"] == "back":
            state["back_img"] = frame
            info_label.config(text="BACK CAPTURED.\n\nEnhancing and Scanning...")
            root.update()
            
            gray = cv2.cvtColor(state["back_img"], cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Read all barcodes
            results = zxingcpp.read_barcodes(enhanced)
            if not results:
                results = zxingcpp.read_barcodes(state["back_img"])
            
            valid_payload = None
            if results:
                for res in results:
                    # Look for the AAMVA payload markers
                    if 'DAC' in res.text or 'DCS' in res.text or 'DBB' in res.text:
                        valid_payload = res.text
                        break
            
            if valid_payload:
                state["result_data"] = _parse_aamva_barcode(valid_payload)
                state["step"] = "done"
            else:
                info_label.config(text="ERROR:\nNo PDF417 barcode detected.\n\nMove back slightly to allow autofocus.\nPress 'SPACE' to try again.")

    def dev_override(event=None):
        """Bypasses the camera and injects valid data for testing ZKP flows."""
        state["result_data"] = {
            "first_name": "TestUser",
            "last_name": "Ontario",
            "birth_date": "1999-05-15"
        }
        state["step"] = "done"

    root.bind('<space>', capture_image)
    root.bind('d', dev_override)
    root.bind('D', dev_override)
    
    update_frame()
    root.mainloop()
    
    if cap.isOpened():
        cap.release()
        
    return state.get("result_data")

if __name__ == "__main__":
    print(get_identaty("drivers_licance"))
