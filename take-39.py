import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Apply median blur to reduce glare
    blr = cv2.medianBlur(img, 15)

    # Convert to HSV and extract brightness channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]

    # Adaptive thresholding
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
        ocr = PaddleOCR(use_angle_cls=True)
        _, encoded_image = cv2.imencode('.jpg', result)
        image_bytes = encoded_image.tobytes()
        ocr_result = ocr.ocr(image_bytes, cls=True)

        print(f"\nExtracted Text from {os.path.basename(image_path)}:")
        extracted_text = []
        for line in ocr_result:
            for word in line:
                extracted_text.append(word[1][0])

        print("\n".join(extracted_text))
    except Exception as e:
        print(f"Error in PaddleOCR for {image_path}: {str(e)}")


if __name__ == "__main__":
    image_dir = r"C:\\vltk sai reddy\\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\\Resources\\Images\\Engine_Chassis_Numbers"

    if not os.path.exists(image_dir):
        print("Error: Directory not found.")
        exit()

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(image_dir, filename)
            process_image(file_path)
