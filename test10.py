import cv2
import pytesseract
import numpy as np

# Set Tesseract-OCR path (Update this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the image
image = cv2.imread(r"C:\Users\sai\Downloads\Camera\2.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to enhance text visibility
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Use morphological operations to further enhance text
kernel = np.ones((3,3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Perform OCR using Tesseract
custom_config = r'--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6 (Assumes a single block of text)
extracted_text = pytesseract.image_to_string(morph, config=custom_config)

# Show processed images (Optional)
cv2.imshow("Original", image)
cv2.imshow("Thresholded", thresh)
cv2.imshow("Morphological Enhancement", morph)

print("Extracted Text:", extracted_text)

cv2.waitKey(0)
cv2.destroyAllWindows()
