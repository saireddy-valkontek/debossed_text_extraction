import cv2
import numpy as np
from paddleocr import PaddleOCR

# Load the image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\Green.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Enhancement)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

# Apply Bilateral Filtering (Preserve edges)
filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

# Apply Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 15, 8)

# Apply Morphological Operations to clean small noise
kernel = np.ones((2, 2), np.uint8)
morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

# Resize Image for better OCR detection
resized = cv2.resize(morphed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

try:
    # Run OCR
    result = ocr.ocr(resized, cls=True)

    # Extract and print detected text
    print("\nExtracted Text from Image:")
    extracted_text = []
    for line in result:
        for word in line:
            extracted_text.append(word[1][0])  # Extracted text

    print("\n".join(extracted_text))

except Exception as e:
    print("Error in PaddleOCR:", str(e))
