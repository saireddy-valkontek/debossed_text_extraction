import cv2
import numpy as np
import pytesseract
import imutils

# Optional: Set tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Step 1: Load input image
image = cv2.imread(r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\240.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Step 3: Skew Correction
def correct_skew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return corrected

# Apply thresholding before skew correction for better results
_, thresh_for_skew = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
deskewed = correct_skew(thresh_for_skew)

# Step 4: Gaussian Blur
blurred = cv2.GaussianBlur(deskewed, (5, 5), 0)

# Step 5: Thresholding + Canny Edge Detection
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 50, 150)

# Step 6: Tesseract OCR
# Use the original (preprocessed) image for OCR
custom_config = r'--oem 3 --psm 6'  # OEM 3: Default (LSTM), PSM 6: Assume a single uniform block of text
text = pytesseract.image_to_string(blurred, config=custom_config)

# Display output (optional)
print("=== Extracted Text ===")
print(text)

# Uncomment to see intermediate results
cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized)
cv2.imshow("Deskewed", deskewed)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
