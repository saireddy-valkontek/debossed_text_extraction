import cv2
import numpy as np

# Load the pre-trained EAST model
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# Read the input image
image = cv2.imread('image.jpg')
height, width, _ = image.shape

# Prepare the image (scale and mean subtraction)
blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (123.68, 116.78, 103.94), True, crop=False)

# Set the input for the model
net.setInput(blob)

# Perform forward pass to get text regions
scores, geometry = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

# Process the results to get bounding boxes
boxes = []
confidences = []

for y in range(scores.shape[2]):
    for x in range(scores.shape[3]):
        score = scores[0, 0, y, x]
        if score > 0.5:  # Confidence threshold
            geometry_values = geometry[0, :, y, x]
            offset_x, offset_y, width, height, angle = geometry_values

            # Calculate the box
            end_x = int(x * 4 + offset_x)
            end_y = int(y * 4 + offset_y)
            boxes.append([end_x, end_y, width, height])

            confidences.append(score)

# Apply non-maximum suppression (NMS) to remove overlapping boxes
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)
for i in indices:
    box = boxes[i[0]]
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

cv2.imshow('Text Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
