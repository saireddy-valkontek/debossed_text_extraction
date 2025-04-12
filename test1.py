import cv2
import numpy as np
import easyocr
from paddleocr import PaddleOCR

# Load the image
image = cv2.imread(r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\grey_2.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding (Otsu's Binarization)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply Canny Edge Detection
edges = cv2.Canny(gray, 50, 150)

# Use Morphological Operations (Dilation & Erosion) to enhance text
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Run OCR on the processed image
ocr = PaddleOCR(use_angle_cls=True, lang="en")
result = ocr.ocr(eroded, cls=True)

# Extract and print detected text
print("\nExtracted Text from Image:")
for line in result:
    if line:  # Ensure line is not empty
        for word in line:
            print(word[1][0])

        # Show results
cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Thresholding", thresh)
cv2.imshow("Edges", edges)
cv2.imshow("Enhanced Text", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()
