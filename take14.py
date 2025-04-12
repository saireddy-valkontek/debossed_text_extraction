# Import necessary libraries
import cv2  # For image processing
import os  # For file and directory operations
import csv  # For writing CSV files
import time  # For time-related operations
import json  # For handling JSON data
import argparse  # For command-line arguments
import random  # For random operations like generating colors
import webbrowser  # To open the output images in the browser
from datetime import datetime  # For getting current timestamp
from ultralytics import YOLO  # For loading YOLO model
from collections import Counter  # For counting object detection labels
import logging  # For logging info and errors
from pathlib import Path  # For working with paths
import shutil  # For file operations like copying or removing files

# Import OCR libraries
import pytesseract  # For using pytesseract
from paddleocr import PaddleOCR  # For using PaddleOCR

# Default model path and device selection (CUDA if available)
DEFAULT_MODEL_PATH = "models/best_2.pt"
DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

# Set up logging configuration to track the application's progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to get the next available image number for naming the image file
def get_next_image_number(output_dir, prefix, suffix):
    existing = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(suffix)]
    numbers = [int(f.replace(prefix, "").replace(suffix, "")) for f in existing if
               f.replace(prefix, "").replace(suffix, "").isdigit()]
    return max(numbers, default=0) + 1


# Function to generate a random color for object labels
def get_color_for_label(label):
    random.seed(hash(label) % 1000)  # Create a seed from label to ensure consistency
    return tuple(random.randint(0, 255) for _ in range(3))  # Return a random color tuple (BGR)


# Function to capture an image from the webcam
def capture_image(width=1024, height=1024):
    cap = cv2.VideoCapture(0)  # Initialize the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # Set frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # Set frame height

    if not cap.isOpened():  # Check if the webcam is accessible
        logging.error("[ERROR] Could not open webcam.")
        return None

    logging.info("[INFO] Press 'c' to capture an image or 'q' to quit.")  # Instructions
    while True:
        ret, frame = cap.read()  # Read the frame from webcam
        if not ret:
            logging.error("[ERROR] Failed to grab frame.")
            break

        cv2.imshow("Live Feed", frame)  # Display live feed
        key = cv2.waitKey(1) & 0xFF  # Capture keypress events

        if key == ord('c'):  # If 'c' is pressed, capture the image
            cap.release()
            cv2.destroyAllWindows()
            return frame  # Return the captured frame
        elif key == ord('q'):  # If 'q' is pressed, quit
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close the window
    return None


# Function to perform OCR using pytesseract
def perform_ocr_pytesseract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    text = pytesseract.image_to_string(gray)  # Perform OCR
    return text.strip()


# Function to perform OCR using PaddleOCR
def perform_ocr_paddleocr(image):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize PaddleOCR with English language
    result = ocr.ocr(image, cls=True)
    text = "\n".join([line[1][0] for line in result[0]])  # Extract OCR results
    return text.strip()


# Function to process the captured image (perform object detection and save results)
def process_image(original_img, yolo_model, confidence_threshold, output_dir, csv_path, show_images, start_time):
    if original_img is None:
        logging.error("[ERROR] No image to process.")  # Check if image is captured
        return

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp

        original_h, original_w = original_img.shape[:2]  # Get image dimensions
        resized_input = cv2.resize(original_img, (1024, 1024))  # Resize image for YOLO model
        img_rgb = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)  # Convert to RGB

        logging.info("[INFO] Running YOLO model inference...")  # Info about YOLO inference
        results = yolo_model.predict(img_rgb, verbose=False)  # Run the YOLO model on the image
        result = results[0]  # We assume one image at a time

        # Calculate scaling factors for resizing bounding boxes
        scale_x = original_w / 1024
        scale_y = original_h / 1024

        detection_count = 0  # Counter for detected objects
        used_label_positions = []  # Keep track of label positions to avoid overlaps
        all_detected_labels = []  # List to store all detected labels
        detection_json_data = []  # List to store detection details for JSON output
        image_filtered_detections = original_img.copy()  # Copy the original image for displaying detections

        boxes = result.boxes  # Get bounding boxes from detection results
        if not boxes or not boxes.xyxy.any():  # Check if no objects are detected
            logging.info("[INFO] No objects detected.")
        else:
            for i in range(len(boxes)):  # Loop over each detection box
                conf_val = float(boxes.conf[i])  # Confidence score
                if conf_val < confidence_threshold:  # Skip detections below confidence threshold
                    continue

                # Get coordinates and class of the detected object
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cls = int(boxes.cls[i])
                label = result.names[cls]
                conf_text = f"{label} ({conf_val:.2f})"  # Text for label and confidence
                color = get_color_for_label(label)  # Random color for the label

                # Scale bounding box coordinates back to original image size
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                # Draw the bounding box on the image
                cv2.rectangle(image_filtered_detections, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)

                # Crop the detected area for OCR processing
                cropped_image = original_img[y1_orig:y2_orig, x1_orig:x2_orig]

                # Perform OCR using pytesseract and PaddleOCR
                pytesseract_text = perform_ocr_pytesseract(cropped_image)
                paddleocr_text = perform_ocr_paddleocr(cropped_image)

                logging.info(f"[INFO] Detected Text (Pytesseract): {pytesseract_text}")
                logging.info(f"[INFO] Detected Text (PaddleOCR): {paddleocr_text}")

                detection_count += 1
                all_detected_labels.append(label)  # Add detected label to the list
                detection_json_data.append({
                    "label": label,
                    "confidence": round(conf_val, 2),
                    "bounding_box": [x1_orig, y1_orig, x2_orig, y2_orig],
                    "pytesseract_text": pytesseract_text,
                    "paddleocr_text": paddleocr_text
                })

        # Prepare directories for saving images and JSON files
        images_dir = os.path.join(output_dir, "images")
        json_dir = os.path.join(output_dir, "json")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        # Get the next image number for naming
        img_number = get_next_image_number(images_dir, "image_", ".jpg")
        conf_output_path = os.path.join(images_dir, f"image_{img_number}.jpg")
        json_output_path = os.path.join(json_dir, f"image_{img_number}.json")

        # Save the filtered image with detections
        cv2.imwrite(conf_output_path, image_filtered_detections)

        # If show_images is True, open the image in the web browser
        if show_images:
            webbrowser.open('file://' + os.path.realpath(conf_output_path))

        # Count detected labels and calculate percentage detection
        label_counts = Counter(all_detected_labels)
        formatted_labels = sorted(label_counts.keys())
        total_classes = len(yolo_model.names)
        percentage_detected = (len(formatted_labels) / total_classes) * 100

        # Write detection results to CSV
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:  # If the file doesn't exist, write the header
                writer.writerow([
                    "timestamp", "image_name", "json_file",
                    "detected_classes", "total_detected_classes",
                    "percentage_detected", "conf_threshold"
                ])
            writer.writerow([
                timestamp,
                os.path.basename(conf_output_path),
                os.path.basename(json_output_path),
                ", ".join(formatted_labels),
                len(formatted_labels),
                f"{percentage_detected:.2f}%",
                f"{confidence_threshold:.2f}"
            ])

        # Save detection results as JSON
        with open(json_output_path, 'w') as json_file:
            json.dump({
                "timestamp": timestamp,
                "detections": detection_json_data
            }, json_file, indent=4)

        # Log the saved paths and time taken
        logging.info(f"[INFO] Saved filtered detections image to: {conf_output_path}")
        logging.info(f"[INFO] Saved JSON to: {json_output_path}")
        logging.info(f"[INFO] CSV appended to: {csv_path}")
        logging.info(f"[INFO] Total processing time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logging.error(f"[ERROR] While processing image: {e}")


# Main function to parse arguments and invoke other functions
def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam or Image Inference Tool")  # Create argument parser
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLO model")  # Model path
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detection")  # Confidence threshold
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory for saving results")  # Output directory
    parser.add_argument("--csv", type=str, default="./output/detection_results.csv",
                        help="Path to CSV file for storing detection results")  # CSV output path
    parser.add_argument("--web", action="store_true",
                        help="Show images in web browser after processing")  # Show images flag
    args = parser.parse_args()  # Parse command line arguments

    # Load YOLO model
    yolo_model = YOLO(args.model)  # Load model
    logging.info("[INFO] Model loaded successfully.")  # Log info

    # Start time for performance tracking
    start_time = time.time()

    # Capture image from webcam or use a pre-captured image
    img = capture_image()  # Capture image (can be replaced with image loading)
    if img is not None:
        process_image(img, yolo_model, args.confidence, args.output, args.csv, args.web,
                      start_time)  # Process the image


# Run the script
if __name__ == "__main__":
    main()
