import cv2
import pytesseract
import numpy as np
import os

# Set this if you're on Windows and Tesseract is not in PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
image = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\250.jpg"
original = cv2.imread(image)
original = cv2.resize(original, (800, 600))  # Resize for visibility

# Create output folder
os.makedirs("output_debug", exist_ok=True)

def save_and_show(name, img):
    path = f"output_debug/{name}.png"
    cv2.imwrite(path, img)
    cv2.imshow(name, img)

# ---------- STEP 1: Grayscale ----------
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
save_and_show("1_gray", gray)

# ---------- STEP 2: CLAHE (local contrast enhancement) ----------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
save_and_show("2_clahe", clahe_img)

# ---------- STEP 3: Adaptive Thresholding ----------
adaptive_thresh = cv2.adaptiveThreshold(clahe_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
save_and_show("3_adaptive_thresh", adaptive_thresh)

# ---------- STEP 4: Invert Threshold (black text on white) ----------
inverted = cv2.bitwise_not(adaptive_thresh)
save_and_show("4_inverted", inverted)

# ---------- STEP 5: Morphological Operations ----------
kernel = np.ones((2, 2), np.uint8)  # CHANGE: adjust kernel size
morphed = cv2.dilate(adaptive_thresh, kernel, iterations=1)
save_and_show("5_morphed_dilated", morphed)

# ---------- STEP 6: OCR with different strategies ----------
ocr_inputs = {
    "Original_Gray": gray,
    "CLAHE": clahe_img,
    "Adaptive_Threshold": adaptive_thresh,
    "Inverted": inverted,
    "Morphed_Dilated": morphed
}

psm_modes = [6, 7, 11, 13]  # CHANGE: You can try more like 3 or 12 if needed

for name, img in ocr_inputs.items():
    for psm in psm_modes:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(img, config=config)
        print(f"\n🧾 OCR on [{name}] using PSM {psm}:\n{text.strip() or '[No Text Found]'}")

# ---------- STEP 7: OCR with confidence inspection ----------
print("\n🔍 Checking OCR confidence on final processed image:")
data = pytesseract.image_to_data(morphed, config='--psm 6', output_type=pytesseract.Output.DICT)
for i, word in enumerate(data['text']):
    conf = int(data['conf'][i])
    if word.strip() != "" and conf > 50:
        print(f"Found '{word}' with confidence {conf}")

cv2.waitKey(0)
cv2.destroyAllWindows()