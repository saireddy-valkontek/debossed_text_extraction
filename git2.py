import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

# Set path to Tesseract if on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ----------- Load Image -----------
img_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\10.jpg"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Image not found at: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------- CLAHE (Contrast Limited Adaptive Histogram Equalization) -----------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)

# ----------- Adaptive Threshold -----------
thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 3)

# ----------- Morphology to clean small noise -----------
kernel = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# ----------- Edge Detection -----------
edges = cv2.Canny(cleaned, 100, 200)

# ----------- OCR -----------
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(cleaned, config=custom_config)

# ----------- Visualization -----------
plt.figure(figsize=(14, 6))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("CLAHE + Gray")
plt.imshow(equalized, cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Threshold + Cleaned")
plt.imshow(cleaned, cmap='gray')
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Edges")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

print("\n📝 OCR Extracted Text:")
print("-" * 40)
print(text.strip())
print("-" * 40)
