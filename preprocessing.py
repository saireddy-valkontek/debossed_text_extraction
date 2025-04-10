from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

# Load image
# img_path = "/Users/sai/Developer/Resources/images/Engine_Chassis_Numbers/photo_250.jpg"
img_path = "./images/1.jpeg"
img = Image.open(img_path).convert("L")  # Grayscale

# Enhance contrast
enhancer = ImageEnhance.Contrast(img)
img_enhanced = enhancer.enhance(3.0)  # Boost contrast

# Apply a slight edge enhancement
img_filtered = img_enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)

# Convert to OpenCV format
open_cv_image = np.array(img_filtered)

# Optionally apply threshold or Canny
# open_cv_image = cv2.Canny(open_cv_image, 50, 150)

# Save to check result
cv2.imwrite("processed.jpg", open_cv_image)
