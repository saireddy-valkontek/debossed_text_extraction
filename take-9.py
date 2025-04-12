import cv2
import numpy as np
import pytesseract
import easyocr
import re
import imutils

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Load, preprocess, and enhance image for OCR."""
    # Load the image
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=1000)  # Resize for better text clarity

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Non-Local Means Denoising (better than Bilateral Filter)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Remove Shadows using Morphological Transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    shadow_removed = cv2.add(gray, blackhat)

    # Apply Otsu's Thresholding (automatically determines best threshold)
    _, otsu = cv2.threshold(shadow_removed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Adaptive Thresholding for better edge contrast
    adaptive = cv2.adaptiveThreshold(otsu, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

    # Morphological Opening (Erosion + Dilation) to refine text regions
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=1)

    return processed_image

def extract_text_tesseract(image):
    """Extract text using Tesseract OCR with optimized config."""
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(image, config=custom_config).strip()
    return post_process_text(text)

def extract_text_easyocr(image_path):
    """Extract text using EasyOCR."""
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    text = " ".join([res[1] for res in results])
    return post_process_text(text)

def post_process_text(text):
    """Cleans up OCR output by removing unwanted characters."""
    return re.sub(r'[^A-Z0-9]', '', text)  # Keep only alphanumeric characters

def extract_chassis_number(image_path):
    """Complete process: preprocess image, apply OCR, and validate extracted text."""
    processed_image = preprocess_image(image_path)

    # Try both OCR methods
    text_tesseract = extract_text_tesseract(processed_image)
    text_easyocr = extract_text_easyocr(image_path)

    # Choose the best result based on length
    final_text = text_easyocr if len(text_easyocr) > len(text_tesseract) else text_tesseract

    return final_text

# Path to image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\Green.jpg"

# Extract chassis/engine number
extracted_text = extract_chassis_number(image_path)

print(f"Extracted Chassis/Engine Number: {extracted_text}")
