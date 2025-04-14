import cv2
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import numpy as np

# Path to your image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\240.jpg"  # replace with your actual image path

# Load image
img = cv2.imread(image_path)

# ---------- Preprocessing (optional but can help) ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Optional: Save or preview preprocessed image
# cv2.imwrite('preprocessed.jpg', thresh)

# ---------- PaddleOCR ----------
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # English model
result = ocr.ocr(thresh, cls=True)

# ---------- Print Results ----------
for line in result:
    for box in line:
        text = box[1][0]
        score = box[1][1]
        print(f'Text: {text}, Confidence: {score:.2f}')

# ---------- (Optional) Draw results ----------
# If you want to visualize detected text
boxes = [elements[0] for line in result for elements in line]
txts = [elements[1][0] for line in result for elements in line]
scores = [elements[1][1] for line in result for elements in line]

# Draw on original image
img_with_box = draw_ocr(img, boxes, txts, scores, font_path='arial.ttf')  # You may need to provide a font
plt.imshow(img_with_box)
plt.axis('off')
plt.show()
