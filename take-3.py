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


# ====================== Define CRNN Model ======================
class CRNN(torch.nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.rnn = torch.nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(256, 26)  # Assuming 26 characters (A-Z)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Convolution
        x = x.squeeze(1)  # Ensure the correct shape
        x = x.permute(0, 2, 3, 1)  # (Batch, Width, Height, Channels)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten height & channels
        x = torch.nn.Linear(x.shape[2], 64).to(x.device)(x)
        x, _ = self.rnn(x)  # Pass through LSTM
        x = self.fc(x)  # Fully connected layer
        return x


# Load CRNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crnn_model = CRNN().to(device)
crnn_model.load_state_dict(torch.load("crnn_model.pth", map_location=device))
crnn_model.eval()

# ====================== Define Text Decoding Function ======================
alphnumeric = "0123456789ABCDEFGIJKLMNOPUVWXZ"


def decode_crnn_output(output):
    _, indices = torch.max(output, 2)  # Get max indices
    indices = indices.squeeze(0).cpu().numpy()  # Convert to numpy

    text = ""
    prev_char = None
    for index in indices:
        if index != prev_char:  # Avoid duplicate consecutive characters
            text += alphnumeric[index]
        prev_char = index
    return text


# ====================== Define Image Processing Function ======================
def process_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO for text detection
    results = yolo_model(img_rgb)

    # Convert bounding boxes to tensor for NMS
    all_boxes = []
    all_scores = []

    for result in results:
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            all_boxes.append(box.tolist())
            all_scores.append(float(conf))

    if len(all_boxes) == 0:
        print("No text detected.")
        return

    # Apply Non-Maximum Suppression (NMS) to filter redundant boxes
    boxes = torch.tensor(all_boxes)
    scores = torch.tensor(all_scores)
    keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold=0.3)

    # Process only the filtered bounding boxes
    for idx in keep_indices:
        x1, y1, x2, y2 = map(int, boxes[idx])
        confidence = all_scores[idx]  # Get confidence score
        cropped_text_region = img[y1:y2, x1:x2]

        # Convert to grayscale for CRNN
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_text_region, cv2.COLOR_BGR2GRAY))

        # Transform for CRNN model
        transform = transforms.ToTensor()
        crnn_input = transform(cropped_pil).unsqueeze(0).to(device)

        # CRNN text recognition
        with torch.no_grad():
            output = crnn_model(crnn_input)

        # Decode predicted text
        predicted_text = decode_crnn_output(output)

        # Prepare label with confidence score
        label = f"{predicted_text} ({confidence:.2f})"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add background rectangle for better readability
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)

        # Put text with confidence score
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Show final image
    cv2.namedWindow("Text Detection & Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Text Detection & Recognition", 1500, 800)
    cv2.imshow("Text Detection & Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ====================== Run Pipeline ======================
# image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\photo_269.jpg"  # Replace with your image
image_path = "test_image.jpg"  # Replace with your image
process_image(image_path)
