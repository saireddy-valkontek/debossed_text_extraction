import cv2
import numpy as np
from paddleocr import PaddleOCR

# Load the image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\soft_light_images\black.jpg"
image = cv2.imread(image_path)  # Read the image

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert to HSV and extract the Value channel (brightness)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Reduce glare using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
v = clahe.apply(v)

# Merge back and convert to BGR
hsv = cv2.merge([h, s, v])
glare_reduced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

try:
    # Run OCR on the processed image
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(glare_reduced, cls=True)

    # Extract and print detected text
    print("\nExtracted Text from Image:")
    for line in result:
        if line:  # Ensure line is not empty
            for word in line:
                print(word[1][0])  # Extracted text

except Exception as e:
    print(f"Error: OCR processing failed. {e}")
