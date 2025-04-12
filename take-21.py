from paddleocr import PaddleOCR
import cv2

# Define the image path
image_path = rimage_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\soft_light_images\blue.jpg"

# Load the image
img = cv2.imread(image_path)

# Check if the image is loaded correctly
if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Resize the image
    resized = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    try:
        # Run OCR on the processed image
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        result = ocr.ocr(resized, cls=True)

        # Extract and print detected text
        print("\nExtracted Text from Image:")
        for line in result:
            if line:  # Ensure line is not empty
                for word in line:
                    print(word[1][0])  # Extracted text

    except Exception as e:
        print(f"Error: OCR processing failed. {e}")
