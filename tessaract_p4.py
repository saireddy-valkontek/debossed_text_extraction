import cv2
import pytesseract
import numpy as np
import os

# Path to input image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\240.jpg" # <== CHANGE THIS if needed
original = cv2.imread(image_path)
original = cv2.resize(original, (800, 600))  # Resize for visibility

# Output folder for debugging
os.makedirs("output_debug", exist_ok=True)

def save_and_show(name, img):
    path = f"output_debug/{name}.png"
    cv2.imwrite(path, img)
    cv2.imshow(name, img)

# ---------- STEP 1: Grayscale ----------
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
save_and_show("1_gray", gray)

# ---------- STEP 2: Denoise and Enhance Contrast ----------
denoised = cv2.bilateralFilter(gray, 11, 17, 17)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(denoised)
save_and_show("2_enhanced", enhanced)

# ---------- STEP 3: Thresholding with Otsu ----------
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
save_and_show("3_binary_otsu", binary)

# ---------- STEP 4: Invert (if needed) ----------
inverted = cv2.bitwise_not(binary)
save_and_show("4_inverted", inverted)

# ---------- STEP 5: Dilate to enhance contours ----------
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(inverted, kernel, iterations=1)
save_and_show("5_dilated", dilated)

# ---------- STEP 6: OCR with refined config ----------
config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

final_img = dilated  # you can change this to `binary` or `inverted` if better
text = pytesseract.image_to_string(final_img, config=config)

print("\n====================")
print("🧾 FINAL OCR OUTPUT:\n")
print(text.strip() or "[No Text Found]")
print("====================\n")

# ---------- STEP 7: Word-level confidence ----------
print("🔍 OCR confidence (words > 60%):")
data = pytesseract.image_to_data(final_img, config=config, output_type=pytesseract.Output.DICT)
for i, word in enumerate(data['text']):
    conf = int(data['conf'][i])
    if word.strip() != "" and conf > 60:
        print(f"'{word}' - {conf}%")

cv2.waitKey(0)
cv2.destroyAllWindows()