import cv2
import pytesseract
import numpy as np

# Optional: Set tesseract path (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image = cv2.imread(r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\240.jpg")  # replace with your image path

# Resize for better visibility
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Increase contrast
gray = cv2.equalizeHist(gray)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Edge detection to highlight debossed text
edges = cv2.Canny(blurred, 100, 200)

# Morphological operations to enhance the features
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

# Combine with original for better effect
combined = cv2.bitwise_or(gray, dilated)

# Thresholding to get binary image
_, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# OCR
text = pytesseract.image_to_string(thresh)

# Output result
print("Extracted Text:\n", text)

# Optional: Show intermediate results
cv2.imshow("Original", image)
cv2.imshow("Processed for OCR", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
