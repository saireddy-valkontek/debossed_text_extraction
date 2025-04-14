import cv2
import numpy as np
import os

# === Path to EAST model ===
model_path = r"C:\Users\sai\Desktop\Dev\Projects\Emdedded_ai_apps\debossed_text_extraction\models\east\frozen_east_text_detection.pb"

print("OpenCV version:", cv2.__version__)
print("Model file exists:", os.path.isfile(model_path))

try:
    net = cv2.dnn.readNet(model_path)
    print("✅ Model loaded successfully!")
except cv2.error as e:
    print("❌ Error loading model:")
    print(e)
    exit(1)

# === Read the input image ===
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\2.jpg"
image = cv2.imread(image_path)
orig = image.copy()
(H, W) = image.shape[:2]

# === Resize to a multiple of 32 ===
newW, newH = (W // 32) * 32, (H // 32) * 32
rW = W / float(newW)
rH = H / float(newH)

resized = cv2.resize(image, (newW, newH))
blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                             (123.68, 116.78, 103.94), swapRB=True, crop=False)

net.setInput(blob)
(scores, geometry) = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

# === Decode the predictions ===
def decode_predictions(scores, geometry, score_thresh=0.5):
    boxes = []
    confidences = []
    (num_rows, num_cols) = scores.shape[2:4]
    for y in range(num_rows):
        for x in range(num_cols):
            score = scores[0, 0, y, x]
            if score < score_thresh:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = geometry[0, 0, y, x]
            w = geometry[0, 1, y, x]

            endX = int(offsetX + (cos * w + sin * h))
            endY = int(offsetY - (sin * w - cos * h))
            startX = int(endX - w)
            startY = int(endY - h)

            boxes.append((startX, startY, endX, endY))
            confidences.append(float(score))

    print(f"Detected {len(boxes)} boxes before NMS")
    return boxes, confidences

boxes, confidences = decode_predictions(scores, geometry, score_thresh=0.5)

# Visualize boxes before NMS (Optional)
for (startX, startY, endX, endY) in boxes:
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

cv2.imshow("Boxes before NMS", orig)
cv2.waitKey(0)

# Apply non-max suppression
indices = cv2.dnn.NMSBoxes([(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in boxes], confidences, 0.5, 0.3)
print(f"{len(indices)} boxes remaining after NMS")

# Ensure indices are in a proper format
if len(indices) > 0:
    # Draw final bounding boxes
    for i in indices.flatten():  # Flatten in case indices is 2D
        (startX, startY, endX, endY) = boxes[i]

        # Scale boxes back to original image size
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
else:
    print("No bounding boxes detected after NMS")

# Resize output image for display if too big
max_display_width = 1000
max_display_height = 1000

(h, w) = orig.shape[:2]
if w > max_display_width or h > max_display_height:
    scale = min(max_display_width / w, max_display_height / h)
    resized_display = cv2.resize(orig, (int(w * scale), int(h * scale)))
else:
    resized_display = orig

cv2.imshow("Text Detection", resized_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
