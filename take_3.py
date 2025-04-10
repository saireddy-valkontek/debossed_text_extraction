from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO  # if using YOLOv8
import torch

# Load models
yolo_model = YOLO("./modals/best_4.pt")  # Your custom YOLO model
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Load original image
image_path = "./images/7.jpeg"
results = yolo_model(image_path)  # Run object detection

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
        img = Image.open(image_path).convert("RGB")
        cropped_img = img.crop((x1, y1, x2, y2))  # Crop to detected region

        # Text extraction with TrOCR
        pixel_values = trocr_processor(images=cropped_img, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("Detected Text:", text)
