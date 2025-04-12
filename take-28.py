import cv2
import numpy as np
from paddleocr import PaddleOCR

# Load the image
fn = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\Green.jpg"
img = cv2.imread(fn)

# Check if the image is loaded successfully
if img is None:
    print("Error: Could not load image.")
    exit()

# Median blur to reduce glare
blr = cv2.medianBlur(img, 15)

# Convert to HSV and extract brightness channel
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Corrected BGR to HSV
val = hsv[:, :, 2]

# Adaptive thresholding to detect bright spots
at = cv2.adaptiveThreshold(255 - val, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 17)
ia = 255 - at
iv = cv2.adaptiveThreshold(ia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 9)
ib = cv2.subtract(iv, ia)

# Convert mask to 3 channels
bz = cv2.merge([ib, ib, ib])

# Apply blur where mask is detected
result = np.where(bz == (0, 0, 0), blr, img)

# OCR Processing
try:
    ocr = PaddleOCR(use_angle_cls=True)  # Initialize OCR

    # Convert the result image to a format PaddleOCR can process
    _, encoded_image = cv2.imencode('.jpg', result)
    image_bytes = encoded_image.tobytes()

    # Run OCR directly on the processed image
    ocr_result = ocr.ocr(image_bytes, cls=True)

    print("\nExtracted Text from Image:")
    extracted_text = []
    for line in ocr_result:
        for word in line:
            extracted_text.append(word[1][0])  # Extract text content

    print("\n".join(extracted_text))

except Exception as e:
    print("Error in PaddleOCR:", str(e))