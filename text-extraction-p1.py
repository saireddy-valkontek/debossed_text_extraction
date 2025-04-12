import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\soft_light_images\red.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# Apply Edge Detection
edges = cv2.Canny(enhanced, 50, 150)

# Use Morphological Transformations to enhance edges
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=2)

# Thresholding for better text extraction
_, binary = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY_INV)
adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,13, 8)


# Combine Edge Detection & Thresholding
processed = cv2.bitwise_or(adaptive_thresh, dilated)

# Show processed image
plt.figure(figsize=(10, 6))
plt.imshow(processed, cmap='gray')
plt.title("Preprocessed Image for OCR")
plt.axis("off")
plt.show()

# Run OCR using EasyOCR
reader = easyocr.Reader(['en'])
results = reader.readtext(processed)

# Print extracted text
extracted_text = " ".join([res[1] for res in results])
print("Extracted Text:", extracted_text)
