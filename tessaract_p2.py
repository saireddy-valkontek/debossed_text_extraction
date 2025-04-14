import cv2
import pytesseract
import numpy as np

# Optional (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
image = cv2.imread(r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\250.jpg")
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Adaptive threshold (better for uneven lighting & color)
thresh = cv2.adaptiveThreshold(
    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 15, 10)

# Morphological operations to enhance the debossed text
kernel = np.ones((2, 2), np.uint8)
morphed = cv2.dilate(thresh, kernel, iterations=1)

# OCR with config tweaks (psm 6 = assume single uniform block of text)
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Run OCR
text = pytesseract.image_to_string(morphed, config=custom_config)
print("Extracted Text:\n", text)

# Show the preprocessed image
cv2.imshow("Processed for OCR", morphed)
cv2.waitKey(0)
cv2.destroyAllWindows()