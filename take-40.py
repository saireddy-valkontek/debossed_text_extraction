import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)


# Folder containing images
folder_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers"

# Check if folder exists
if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' does not exist.")
    exit()

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Get all .jpg files in the folder
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

# Check if any images are found
if not image_files:
    print("No .jpg images found in the directory.")
    exit()

# Process each image
for filename in image_files:
    image_path = os.path.join(folder_path, filename)
    print(f"\nProcessing File: {filename}")

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Could not load image {filename}. Skipping...")
        continue

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
        result = ocr.ocr(glare_reduced, cls=True)

        # Extract and print detected text
        print("Extracted Text:")
        for line in result:
            if line:  # Ensure line is not empty
                for word in line:
                    print(word[1][0])  # Extracted text

    except Exception as e:
        print(f"Error: OCR processing failed for {filename}. {e}")

print("\nProcessing Completed!")
