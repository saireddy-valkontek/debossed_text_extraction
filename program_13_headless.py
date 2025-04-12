import cv2
import os
import csv
import time
import json
import argparse
import random
import logging
from datetime import datetime
from ultralytics import YOLO
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default model path
DEFAULT_MODEL_PATH = "models/yolov8/best_5.pt"
DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

def get_color_for_label(label):
    random.seed(hash(label) % 1000)
    return tuple(random.randint(0, 255) for _ in range(3))

def get_next_image_number(output_dir, prefix, suffix):
    existing = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(suffix)]
    numbers = [int(f.replace(prefix, "").replace(suffix, "")) for f in existing if f.replace(prefix, "").replace(suffix, "").isdigit()]
    return max(numbers, default=0) + 1

def capture_image(width, height):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        logging.error("Could not access camera.")
        return None

    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def process_image(original_img, yolo_model, conf_threshold, output_dir, csv_path, source, start_time):
    if original_img is None:
        logging.error("No image to process.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_h, original_w = original_img.shape[:2]
    resized_input = cv2.resize(original_img, (768, 768))
    img_rgb = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)

    logging.info("Running inference...")
    results = yolo_model.predict(img_rgb, verbose=False)
    result = results[0]

    scale_x = original_w / 768
    scale_y = original_h / 768

    detection_count = 0
    all_detected_labels = []
    detection_json_data = []
    image_filtered_detections = original_img.copy()

    boxes = result.boxes
    if boxes and boxes.xyxy.any():
        for i in range(len(boxes)):
            conf_val = float(boxes.conf[i])
            if conf_val < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            cls = int(boxes.cls[i])
            label = result.names[cls]
            color = get_color_for_label(label)

            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)

            cv2.rectangle(image_filtered_detections, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
            cv2.putText(image_filtered_detections, f"{label} ({conf_val:.2f})", (x1_orig, y1_orig - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detection_count += 1
            all_detected_labels.append(label)
            detection_json_data.append({
                "label": label,
                "confidence": round(conf_val, 2),
                "bounding_box": [x1_orig, y1_orig, x2_orig, y2_orig]
            })
    else:
        logging.info("No objects detected.")

    images_dir = os.path.join(output_dir, "images")
    json_dir = os.path.join(output_dir, "json")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    img_number = get_next_image_number(images_dir, "image_", ".jpg")
    img_file = f"image_{img_number}.jpg"
    json_file = f"image_{img_number}.json"

    img_path = os.path.join(images_dir, img_file)
    json_path = os.path.join(json_dir, json_file)

    cv2.imwrite(img_path, image_filtered_detections)

    with open(json_path, 'w') as jf:
        json.dump({
            "timestamp": timestamp,
            "source": source,
            "detections": detection_json_data
        }, jf, indent=4)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "timestamp", "source", "image_name", "json_file",
                "detected_classes", "total_detected_classes",
                "confidence_threshold"
            ])
        writer.writerow([
            timestamp,
            source,
            img_file,
            json_file,
            ", ".join(sorted(set(all_detected_labels))),
            len(set(all_detected_labels)),
            conf_threshold
        ])

    logging.info(f"Source: {source}")
    logging.info(f"Image saved: {img_path}")
    logging.info(f"JSON saved: {json_path}")
    logging.info(f"CSV updated: {csv_path}")
    logging.info(f"Detection time: {time.time() - start_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Headless Inference")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--source", type=str, choices=["camera", "image"], required=True, help="Input source")
    parser.add_argument("--image", type=str, help="Path to image file (required if source is image)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    parser.add_argument("--cam-width", type=int, default=768)
    parser.add_argument("--cam-height", type=int, default=768)

    args = parser.parse_args()

    if args.source == "image":
        if not args.image or not os.path.exists(args.image):
            logging.error("Image path not provided or invalid.")
            return
        img = cv2.imread(args.image)
        source_desc = args.image
    else:
        img = capture_image(args.cam_width, args.cam_height)
        source_desc = "camera"

    yolo_model = YOLO(args.model)
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "summary.csv")

    start_time = time.time()
    process_image(img, yolo_model, args.conf, args.output, csv_path, source_desc, start_time)

if __name__ == "__main__":
    main()