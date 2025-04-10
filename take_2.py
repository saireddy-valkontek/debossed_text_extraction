import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

# Paths
yolo_model_path = "modals/best_4.pt"
cnn_model_path = "modals/char_cnn_kaggle.pth"
test_image_path = "./images/5.jpeg"

# Load YOLOv8 model
yolo_model = YOLO(yolo_model_path)

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

# Load class names
class_names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "I", "J",
    "K", "L", "M", "N", "O", "P", "R", "U", "V",
    "W", "X", "Z"
]

# Load trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

# -------- 🔧 Preprocessing Step --------
original_img = cv2.imread(test_image_path)

# Convert to grayscale
gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Contrast enhancement
equalized = cv2.equalizeHist(gray)

# Sharpening kernel
sharpen_kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])
sharpened = cv2.filter2D(equalized, -1, sharpen_kernel)

# Convert back to BGR for YOLO
preprocessed_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# Save preprocessed image temporarily (YOLO expects a file path)
temp_path = "temp_preprocessed.jpg"
cv2.imwrite(temp_path, preprocessed_img)

# -------- 🔍 Detection Step --------
results = yolo_model(temp_path)[0]
boxes = results.boxes.xyxy.cpu().numpy()
conf = results.boxes.conf.cpu().numpy()

# -------- 🔠 Classification Step --------
predictions = []
sorted_boxes = sorted(boxes, key=lambda x: x[0])

for box in sorted_boxes:
    x1, y1, x2, y2 = map(int, box)
    crop = sharpened[y1:y2, x1:x2]  # use sharpened grayscale crop
    crop_pil = Image.fromarray(crop)
    input_tensor = transform(crop_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = cnn_model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        predictions.append(class_names[pred_class])

# -------- 🧾 Output --------
final_text = "".join(predictions)
print("🧾 Extracted Text:", final_text)

# -------- 🖼️ Optional: Show result --------
for box in sorted_boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(preprocessed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Characters", preprocessed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
