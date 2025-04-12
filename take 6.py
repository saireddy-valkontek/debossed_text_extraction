import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import numpy as np

# ====================== Constants ======================
MODEL_PATH = "crnn_model.pth"
IMAGE_PATH = "test_image.jpg"
ALPHANUMERIC = "0123456789ABCDEFGIJKLMNOPRUVWXZ"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for detected text

# ====================== CRNN Model Definition ======================
class CRNNModel(nn.Module):
    def __init__(self):
        super(CRNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.feature_reduction = nn.Linear(512 * 4, 64)
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2, len(ALPHANUMERIC))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        x = self.feature_reduction(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# ====================== Load CRNN Model ======================
def load_crnn_model():
    model = CRNNModel().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("[INFO] CRNN model loaded successfully.")
    return model

# ====================== Decode CRNN Output ======================
def decode_crnn_output(output):
    _, indices = torch.max(output, 2)
    indices = indices.squeeze(0).detach().cpu().numpy()

    text = ""
    prev = -1
    for idx in indices:
        if idx != prev and idx < len(ALPHANUMERIC):
            text += ALPHANUMERIC[idx]
        prev = idx
    return text

# ====================== Process Image & Run OCR ======================
def process_image(image_path, yolo_model, crnn_model):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("[ERROR] Image not found or corrupted.")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb)

        boxes = []
        scores = []
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                if float(conf) > CONFIDENCE_THRESHOLD:  # Filter low-confidence boxes
                    boxes.append(box.tolist())
                    scores.append(float(conf))

        if not boxes:
            print("[INFO] No text detected above confidence threshold.")
            return

        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold=0.3)

        transform = transforms.Compose([
            transforms.Resize((64, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        for idx in keep:
            x1, y1, x2, y2 = map(int, boxes[idx])
            confidence = scores[idx]

            if x2 - x1 == 0 or y2 - y1 == 0:
                print(f"[WARNING] Empty bounding box skipped: ({x1}, {y1}, {x2}, {y2})")
                continue

            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                print(f"[WARNING] Cropped image is empty at ({x1}, {y1}, {x2}, {y2}). Skipping...")
                continue

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(gray)
            input_tensor = transform(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = crnn_model(input_tensor)
                predicted_text = decode_crnn_output(output)

            label = f"{predicted_text} ({confidence:.2f})"
            print(f"[INFO] Detected text: {predicted_text}, Confidence: {confidence:.2f}")

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print("drawing box")
            # Draw text label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        print("here before save")
        # Save and show result
        cv2.imwrite("output.jpg", img)
        cv2.imshow("Text Detection & Recognition", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] While processing image: {e}")

# ====================== Capture Image ======================
def capture_image():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_GAIN, 7)

    print("Press 'c' to capture image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        cv2.imshow("Live OCR Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite(IMAGE_PATH, frame)
            break

    cap.release()
    cv2.destroyAllWindows()

# ====================== Main Function ======================
def main():
    try:
        capture_image()
        yolo_model = YOLO("yolo_model.pt")
        crnn_model = load_crnn_model()
        process_image(IMAGE_PATH, yolo_model, crnn_model)
    except Exception as e:
        print(f"[FATAL ERROR] {e}")

if __name__ == "__main__":
    main()
