from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

# ====================== Capture High-Resolution Image ======================
# Open the camera
cap = cv2.VideoCapture(0)

# Set the highest resolution (modify based on camera specs)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

# Set manual exposure and gain for better clarity
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto-exposure
cap.set(cv2.CAP_PROP_GAIN, 7)  # Increase brightness

# Capture a frame
while True:
    ret, frame = cap.read()
    # Show Frame
    cv2.imshow("Live OCR Detection", frame)

    # Press 'c' to quit
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite("test_image.jpg", frame)
        break

else:
    print("Failed to capture image.")
    cap.release()
    exit()

# Release the camera
cap.release()
cv2.destroyAllWindows()

# ====================== Load YOLO Model ======================
yolo_model = YOLO("yolo_model.pt")  # Load your trained YOLOv8 model

# ====================== Load CRNN Model ======================
class CRNN(torch.nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.rnn = torch.nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(256, 26)  # Assuming 26 characters (A-Z)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, 64)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# Load CRNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crnn_model = CRNN().to(device)
crnn_model.load_state_dict(torch.load("crnn_model.pth", map_location=device))
crnn_model.eval()

# ====================== Process Image ======================
def process_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO for text detection
    results = yolo_model(img_rgb)

    # Loop through detected objects
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes
        labels = result.boxes.cls  # Class labels
        confs = result.boxes.conf  # Confidence scores

        for box, label, conf in zip(boxes, labels, confs):
            x1, y1, x2, y2 = map(int, box)
            cropped_text_region = img[y1:y2, x1:x2]

            # Convert to grayscale for CRNN
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_text_region, cv2.COLOR_BGR2GRAY))

            # Transform for CRNN model
            transform = transforms.ToTensor()
            crnn_input = transform(cropped_pil).unsqueeze(0).to(device)

            # CRNN text recognition
            with torch.no_grad():
                output = crnn_model(crnn_input)

            # Convert output to text
            predicted_text = "".join([chr(65 + torch.argmax(output[0][i]).item()) for i in range(output.shape[1])])

            # Draw bounding box & recognized text
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, predicted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show final image
    cv2.imshow("Text Detection & Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ====================== Run Pipeline ======================
image_path = "test_image.jpg"  # Replace with your image
process_image(image_path)
