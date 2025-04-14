# Import necessary libraries
import cv2
import os
import csv
import time
import json
import argparse
import random
import webbrowser
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import logging

# Default model path and device selection (CUDA if available)
DEFAULT_MODEL_PATH = "models/yolov8/best_4.pt"
DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Functions
def get_next_image_number(output_dir, prefix, suffix):
    existing = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(suffix)]
    numbers = [int(f.replace(prefix, "").replace(suffix, "")) for f in existing if f.replace(prefix, "").replace(suffix, "").isdigit()]
    return max(numbers, default=0) + 1

def get_color_for_label(label):
    random.seed(hash(label) % 1000)
    return tuple(random.randint(0, 255) for _ in range(3))

def capture_image(width=768, height=768):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        logging.error("[ERROR] Could not open webcam.")
        return None

    logging.info("[INFO] Press 'c' to capture or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("[ERROR] Failed to grab frame.")
            break
        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def process_image(original_img, yolo_model, confidence_threshold, output_dir, csv_path, show_images, start_time):
    if original_img is None:
        logging.error("[ERROR] No image to process.")
        return

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        original_h, original_w = original_img.shape[:2]
        resized_input = cv2.resize(original_img, (768, 768))
        img_rgb = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)

        logging.info("[INFO] Running YOLO model inference...")
        results = yolo_model.predict(img_rgb, verbose=False)
        result = results[0]

        scale_x = original_w / 768
        scale_y = original_h / 768

        detection_count = 0
        used_label_positions = []
        all_detected_labels = []
        detection_json_data = []
        image_filtered_detections = original_img.copy()

        boxes = result.boxes
        if not boxes or not boxes.xyxy.any():
            logging.info("[INFO] No objects detected.")
        else:
            for i in range(len(boxes)):
                conf_val = float(boxes.conf[i])
                if conf_val < confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cls = int(boxes.cls[i])
                label = result.names[cls]
                conf_text = f"{label} ({conf_val:.2f})"
                color = get_color_for_label(label)

                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                cv2.rectangle(image_filtered_detections, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                label_y = y1_orig - 10 if y1_orig - 10 > th else y2_orig + 10
                while any(abs(label_y - used) < th + 5 for used in used_label_positions):
                    label_y += th + 5
                    if label_y + th > image_filtered_detections.shape[0]:
                        label_y = y1_orig + th + 10
                        break

                used_label_positions.append(label_y)
                cv2.rectangle(image_filtered_detections, (x1_orig, label_y - th), (x1_orig + tw + 4, label_y + 4), color, -1)
                cv2.putText(image_filtered_detections, conf_text, (x1_orig + 2, label_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                detection_count += 1
                all_detected_labels.append(label)
                detection_json_data.append({
                    "label": label,
                    "confidence": round(conf_val, 2),
                    "bounding_box": [x1_orig, y1_orig, x2_orig, y2_orig]
                })

        images_dir = os.path.join(output_dir, "images")
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        img_number = get_next_image_number(images_dir, "image_", ".jpg")
        conf_output_path = os.path.join(images_dir, f"image_{img_number}.jpg")
        json_output_path = os.path.join(json_dir, f"image_{img_number}.json")

        cv2.imwrite(conf_output_path, image_filtered_detections)

        if show_images:
            webbrowser.open('file://' + os.path.realpath(conf_output_path))

        label_counts = Counter(all_detected_labels)
        formatted_labels = sorted(label_counts.keys())
        total_classes = len(yolo_model.names)
        percentage_detected = (len(formatted_labels) / total_classes) * 100

        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "image_name", "json_file", "detected_classes", "total_detected_classes", "percentage_detected", "conf_threshold"])
            writer.writerow([
                timestamp,
                os.path.basename(conf_output_path),
                os.path.basename(json_output_path),
                ", ".join(formatted_labels),
                len(formatted_labels),
                f"{percentage_detected:.2f}%",
                f"{confidence_threshold:.2f}"
            ])

        with open(json_output_path, 'w') as json_file:
            json.dump({
                "timestamp": timestamp,
                "detections": detection_json_data
            }, json_file, indent=4)

        logging.info(f"[INFO] Saved image: {conf_output_path}")
        logging.info(f"[INFO] Saved JSON: {json_output_path}")
        logging.info(f"[INFO] CSV updated: {csv_path}")
        logging.info(f"[INFO] Processing time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"[ERROR] Exception in processing image: {e}")

# Main function
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam or Image Inference Tool")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--image", type=str, help="Path to an image file instead of using webcam")
    parser.add_argument("--output", type=str, default="output", help="Directory to save results")
    parser.add_argument("--no-show", action="store_true", help="Disable image preview")
    parser.add_argument("--cam-width", type=int, default=768, help="Camera width")
    parser.add_argument("--cam-height", type=int, default=768, help="Camera height")

    args = parser.parse_args()

    # Load YOLO model
    logging.info("[INFO] Loading YOLO model...")
    model = YOLO(args.model)

    # Get image
    image = None
    if args.image:
        if not os.path.exists(args.image):
            logging.error(f"[ERROR] Image file not found: {args.image}")
            return
        image = cv2.imread(args.image)
        if image is None:
            logging.error("[ERROR] Failed to load image from file.")
            return
    else:
        image = capture_image(args.cam_width, args.cam_height)

    # Start timer and process image
    start_time = time.time()
    process_image(
        image,
        yolo_model=model,
        confidence_threshold=args.conf,
        output_dir=args.output,
        csv_path=os.path.join(args.output, "results.csv"),
        show_images=not args.no_show,
        start_time=start_time
    )

if __name__ == "__main__":
    main()
