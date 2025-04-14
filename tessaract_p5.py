import cv2
import pytesseract
import numpy as np
import os

# Path to your image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\240.jpg"  # <== Update with your image path
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

# ---------- STEP 2: Adaptive Thresholding ----------
# Try varying parameters in the thresholding for better edges
adaptive_thresh = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
save_and_show("2_adaptive_thresh", adaptive_thresh)

# ---------- STEP 3: Inversion ----------
inverted = cv2.bitwise_not(adaptive_thresh)
save_and_show("3_inverted", inverted)

# ---------- STEP 4: Morphological Operations (Larger Kernel) ----------
kernel = np.ones((3, 3), np.uint8)  # Larger kernel size
morphed = cv2.dilate(inverted, kernel, iterations=2)
save_and_show("4_morphed_dilated", morphed)

# ---------- STEP 5: OCR with refined config ----------
# Update the config for a more aggressive OCR approach
config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# You can also experiment with psm 7 if needed, but 6 often works best for text lines
text = pytesseract.image_to_string(morphed, config=config)

print("\n====================")
print("🧾 FINAL OCR OUTPUT:\n")
print(text.strip() or "[No Text Found]")
print("====================\n")

# ---------- STEP 6: Word-level confidence ----------
print("🔍 OCR confidence (words > 60%):")
data = pytesseract.image_to_data(morphed, config=config, output_type=pytesseract.Output.DICT)
for i, word in enumerate(data['text']):
    conf = int(data['conf'][i])
    if word.strip() != "" and conf > 60:
        print(f"'{word}' - {conf}%")

cv2.waitKey(0)
cv2.destroyAllWindows()
