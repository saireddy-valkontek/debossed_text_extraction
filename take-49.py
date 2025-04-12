import cv2
import numpy as np
from paddleocr import PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

#load the image
image = cv2.imread(r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\red.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Enhancement)
clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
clahe_applied = clahe.apply(gray)

# Apply Bilateral Filtering (Keeps edges sharp)
bilateral = cv2.bilateralFilter(clahe_applied, 8, 82 , 70)

# Apply Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2 )

# Apply Morphological Operations (Closing) to remove noise
kernel = np.ones((1, 1), np.uint8)
# morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
sharpened = cv2.filter2D(adaptive_thresh, -1, sharpen_kernel)

# Apply Gamma Correction (Brightness Adjustment)
gamma = 6.5  # Adjust this value based on brightness
gamma_corrected = np.array(255 * (sharpened / 255) ** gamma, dtype=np.uint8)
# cv2.imshow("gamma",gamma_corrected )

# Save the processed image (Fixed for PaddleOCR)
processed_image_path = "processed.jpg"
cv2.imwrite(processed_image_path, gamma_corrected)

# Run OCR on the saved processed image (Fixed)
ocr = PaddleOCR(use_angle_cls=True, lang="en")
result = ocr.ocr(processed_image_path, cls=True)

# Extract and print detected text
print("\nExtracted Text from Image:")
for line in result:
    for word in line:
        print(word[1][0])  # Extracted text