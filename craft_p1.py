import cv2
import torch
import numpy as np
import os
from craft_text_detector import Craft

# Path to the image
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\2.jpg"

# Initialize CRAFT model
craft = Craft()

# === Read the image ===
image = cv2.imread(image_path)

# === Run CRAFT to detect text regions ===
text_result = craft.detect_text(image)

# === Extract and display the detected text regions ===
for box in text_result['boxes']:
    points = box.astype(int)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

# Resize and display the result image
resized_image = cv2.resize(image, (800, 600))
cv2.imshow("CRAFT Text Detection", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result if necessary
cv2.imwrite("detected_text_result.jpg", resized_image)

# Clean up the CRAFT resources
craft.shutdown()
