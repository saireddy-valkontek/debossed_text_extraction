import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt

# Paths
yolo_model_path = "modals/best_4.pt"
cnn_model_path = "modals/char_cnn_kaggle.pth"
# test_image_path = "/Users/sai/Developer/Resources/images/Engine_Chassis_Numbers/photo_250.jpg"
# test_image_path = "./images/7.jpeg"
test_image_path = "processed.jpg"

# Load YOLOv8 model
yolo_model = YOLO(yolo_model_path)

# Run detection
results = yolo_model(test_image_path)[0]
boxes = results.boxes.xyxy.cpu().numpy()   # x1, y1, x2, y2
conf = results.boxes.conf.cpu().numpy()

# Load CNN model
class CharCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CharCNN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16 * 16, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 31)
        )

    def forward(self, x):
        return self.net(x)

# Load dataset classes to get class names
# dataset = datasets.ImageFolder(r"/Users/sai/Developer/Resources/datasets/classification_dataset/train")  # or update path
class_names = class_names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "I", "J",
    "K", "L", "M", "N", "O", "P", "R", "U", "V",
    "W", "X", "Z"
]


# Load trained model
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
cnn_model = CharCNN(num_classes=len(class_names)).to(device)
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
cnn_model.eval()

# Transform for CNN input
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and prepare image
img = Image.open(test_image_path).convert("RGB")
img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Prepare list to hold (x, predicted_char)
predictions = []

# Sort boxes left-to-right by x1
sorted_boxes = sorted(boxes, key=lambda x: x[0])

for box in sorted_boxes:
    x1, y1, x2, y2 = map(int, box)
    crop = img.crop((x1, y1, x2, y2))
    input_tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = cnn_model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        predictions.append(class_names[pred_class])

# Final text
final_text = "".join(predictions)
print("🧾 Extracted Text:", final_text)

# Optional: Show detections
for box in sorted_boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("Detected Characters", img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
