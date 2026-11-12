# user_identaty_util.py
"""
Utility module for capturing and extracting user identity data from physical documents.
Requires: pip install opencv-python Pillow zxing-cpp
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import zxingcpp
import config
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
    print("\n[DEBUG] Raw Barcode Payload:")
    print(repr(barcode_text))
    print("-" * 30)

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

def _get_from_drivers_license():
    """
    Opens a centered GUI with a live camera feed to capture and decode the 
    PDF417 barcode on the back of a Canadian Driver's License.
    """
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