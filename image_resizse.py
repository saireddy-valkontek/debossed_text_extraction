import cv2
import numpy as np

def letterbox_resize(image, target_size=768, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create new canvas
    result = np.full((target_size, target_size, 3), color, dtype=np.uint8)

    # Compute padding
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2

    # Paste the resized image onto the canvas
    result[top:top + new_h, left:left + new_w] = resized
    return result

# Usage
image = cv2.imread(r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\new_engine_chassis_numbers\cropped\7.jpeg")
resized_image = letterbox_resize(image, 768)
cv2.imwrite("resized.jpg", resized_image)
