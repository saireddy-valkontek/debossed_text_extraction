from ultralytics import YOLO
import cv2

# Global config
YOLO_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.15 # Lowered for cropped image sensitivity
MODEL_PATH = "./models/yolov8/best_crp_2.pt"
# IMAGE_PATH = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\158.jpg"
IMAGE_PATH = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\new_engine_chassis_numbers\cropped\7.jpeg"
# IMAGE_PATH = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\new_engine_chassis_numbers\3.jpeg"

# Load model
model = YOLO(MODEL_PATH)

# Read image
image = cv2.imread(IMAGE_PATH)

# Optional: Pad image to preserve aspect ratio (just like YOLO's letterbox during training)
def letterbox(img, new_size=YOLO_IMAGE_SIZE):
    height, width = img.shape[:2]
    scale = min(new_size / width, new_size / height)
    new_w, new_h = int(width * scale), int(height * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = 255 * np.ones((new_size, new_size, 3), dtype=np.uint8)
    top = (new_size - new_h) // 2
    left = (new_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas

# Resize with padding to match training letterbox format
import numpy as np
letterboxed = letterbox(image)

# Run inference
results = model(letterboxed, imgsz=YOLO_IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD)

# Draw results on image
annotated_image = results[0].plot()

# Display the image
cv2.imshow("YOLOv8 Detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
