import cv2
import os
import time
import json
import argparse
import random
import webbrowser
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import logging
import numpy as np
import csv
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path
import platform

# Constants
DEFAULT_MODEL_PATH = "models/yolov8/best_s.pt"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_CAMERA_RESOLUTION = (640, 640)
TARGET_IMAGE_SIZE = (640, 640)
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
MAX_IMAGE_SIZE_MB = 10  # Maximum allowed image size in MB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('object_detection.log')
    ]
)
logger = logging.getLogger(__name__)


class SystemUtils:
    """Utility class for system-related operations"""

    @staticmethod
    def get_available_device() -> str:
        """Determine the best available device for inference."""
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return "cuda"
        if platform.system() == 'Darwin':  # macOS
            return "mps"
        return "cpu"

    @staticmethod
    def check_disk_space(path: str, min_space_gb: float = 0.1) -> bool:
        """Check if there's sufficient disk space."""
        try:
            stat = os.statvfs(path)
            free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            return free_space_gb >= min_space_gb
        except Exception:
            logger.warning("Could not check disk space")
            return True  # Assume enough space if check fails


class FileHandler:
    """Handles all file operations with proper validation"""

    @staticmethod
    def validate_image_path(image_path: str) -> bool:
        """Validate an image path exists and is supported format."""
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return False

        if not image_path.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
            logger.error(f"Unsupported image format. Supported formats: {SUPPORTED_IMAGE_EXTENSIONS}")
            return False

        # Check file size
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > MAX_IMAGE_SIZE_MB:
            logger.error(f"Image too large ({file_size_mb:.2f}MB > {MAX_IMAGE_SIZE_MB}MB limit)")
            return False

        return True

    @staticmethod
    def create_directory(path: str) -> bool:
        """Safely create directory if it doesn't exist."""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False

    @staticmethod
    def safe_image_write(path: str, image: np.ndarray) -> bool:
        """Safely write image to disk with validation."""
        try:
            if not isinstance(image, np.ndarray):
                raise ValueError("Invalid image format")

            if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
                raise ValueError("Unsupported output image format")

            cv2.imwrite(path, image)

            # Verify the image was written correctly
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                raise IOError("Failed to write image file")

            return True
        except Exception as e:
            logger.error(f"Failed to save image {path}: {e}")
            return False


class ObjectDetector:
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the object detector with a YOLO model."""
        self.model_path = model_path
        self.device = self._initialize_device(device)
        self.model = self._load_model()
        self.label_colors = {}  # Cache for label colors
        self._validate_model()

    def _initialize_device(self, device: str) -> str:
        """Initialize the processing device with validation."""
        if device == "auto":
            return SystemUtils.get_available_device()

        if device not in ["cpu", "cuda", "mps"]:
            logger.warning(f"Unsupported device {device}, defaulting to auto")
            return SystemUtils.get_available_device()

        return device

    def _load_model(self) -> YOLO:
        """Load the YOLO model with comprehensive error handling."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        try:
            logger.info(f"Loading YOLO model from {self.model_path} on {self.device}...")
            start_time = time.time()
            model = YOLO(self.model_path).to(self.device)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _validate_model(self):
        """Validate the loaded model has required attributes."""
        if not hasattr(self.model, 'names'):
            raise AttributeError("Loaded model missing 'names' attribute")
        if not hasattr(self.model, 'predict'):
            raise AttributeError("Loaded model missing 'predict' method")

    def get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        """Get a consistent color for a given label."""
        if label not in self.label_colors:
            random.seed(hash(label) % 1000)
            self.label_colors[label] = tuple(random.randint(0, 255) for _ in range(3))
        return self.label_colors[label]

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Preprocess image with padding and return scaling factors."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")

        original_h, original_w = image.shape[:2]

        # Calculate scaling factors before resizing
        scale_x = original_w / TARGET_IMAGE_SIZE[0]
        scale_y = original_h / TARGET_IMAGE_SIZE[1]

        # Resize with padding
        scale = min(TARGET_IMAGE_SIZE[0] / original_w, TARGET_IMAGE_SIZE[1] / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized_img = cv2.resize(image, (new_w, new_h))
        padded_img = np.zeros((TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0], 3), dtype=np.uint8)

        pad_x = (TARGET_IMAGE_SIZE[0] - new_w) // 2
        pad_y = (TARGET_IMAGE_SIZE[1] - new_h) // 2

        padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_img
        return padded_img, scale_x, scale_y

    def detect_objects(
            self,
            image: np.ndarray,
            confidence_threshold: float = DEFAULT_CONFIDENCE
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str]]:
        """Perform object detection on an image with comprehensive validation."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be in BGR format with 3 channels")

        try:
            # Preprocess image
            start_time = time.time()
            padded_img, scale_x, scale_y = self._preprocess_image(image)
            img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

            # Run inference
            inference_start = time.time()
            results = self.model.predict(img_rgb, verbose=False)
            inference_time = time.time() - inference_start

            if not results or len(results) == 0:
                logger.info("No detection results returned")
                return [], image.copy(), []

            result = results[0]
            boxes = result.boxes
            if not boxes or not boxes.xyxy.any():
                logger.info("No objects detected.")
                return [], image.copy(), []

            # Process detections
            detections = []
            all_labels = []
            filtered_image = image.copy()
            used_label_positions = []

            for i in range(len(boxes)):
                conf_val = float(boxes.conf[i])
                if conf_val < confidence_threshold:
                    continue

                # Extract detection info
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cls = int(boxes.cls[i])
                label = result.names[cls]
                all_labels.append(label)
                color = self.get_color_for_label(label)

                # Convert coordinates to original image space
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                # Draw bounding box
                cv2.rectangle(filtered_image, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)

                # Prepare label text
                conf_text = f"{label} ({conf_val:.2f})"
                (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Calculate non-overlapping label position
                label_y = self._calculate_label_position(
                    y1_orig, y2_orig, th, filtered_image.shape[0], used_label_positions
                )
                used_label_positions.append(label_y)

                # Draw label
                self._draw_label(filtered_image, x1_orig, label_y, tw, th, color, conf_text)

                # Store detection
                detections.append({
                    "label": label,
                    "confidence": round(conf_val, 2),
                    "bounding_box": [x1_orig, y1_orig, x2_orig, y2_orig],
                    "inference_time_ms": round(inference_time * 1000, 2)
                })

            total_time = time.time() - start_time
            logger.info(f"Detection completed in {total_time:.2f} seconds (inference: {inference_time:.2f}s)")
            return detections, filtered_image, all_labels

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

    def _calculate_label_position(
            self,
            y1: int,
            y2: int,
            text_height: int,
            image_height: int,
            used_positions: List[int]
    ) -> int:
        """Calculate optimal label position avoiding overlaps."""
        # Try above the box first
        label_y = y1 - 10 if y1 - 10 > text_height else y2 + 10

        # Adjust if overlaps with existing labels
        while any(abs(label_y - used) < text_height + 5 for used in used_positions):
            label_y += text_height + 5
            if label_y + text_height > image_height:
                label_y = y1 + text_height + 10
                break

        return label_y

    def _draw_label(
            self,
            image: np.ndarray,
            x: int,
            y: int,
            text_width: int,
            text_height: int,
            color: Tuple[int, int, int],
            text: str
    ):
        """Draw label with background on the image."""
        # Draw background rectangle
        cv2.rectangle(
            image,
            (x, y - text_height),
            (x + text_width + 4, y + 4),
            color,
            -1
        )
        # Draw text
        cv2.putText(
            image,
            text,
            (x + 2, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

class CameraHandler:
    """Handles all camera operations with proper resource management"""

    def __init__(self, width: int = DEFAULT_CAMERA_RESOLUTION[0], height: int = DEFAULT_CAMERA_RESOLUTION[1]):
        self.width = width
        self.height = height
        self.cap = None

    def __enter__(self):
        """Context manager entry for resource handling"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def capture_image(self) -> Optional[np.ndarray]:
        """Capture an image from the webcam with user interaction."""
        logger.info("Press 'c' to capture an image or 'q' to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                return None

            cv2.imshow("Live Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                return frame
            elif key == ord('q'):
                return None


class ResultSaver:
    """Handles saving detection results with proper validation"""

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.json_dir = self.output_dir / "json"
        self.csv_path = self.output_dir / "detections.csv"

        # Initialize directories
        self._initialize_directories()

        # Cache for performance
        self._last_image_number = self._get_highest_image_number()

    def _initialize_directories(self):
        """Create required directories with validation."""
        if not FileHandler.create_directory(self.images_dir):
            raise RuntimeError(f"Failed to create images directory: {self.images_dir}")
        if not FileHandler.create_directory(self.json_dir):
            raise RuntimeError(f"Failed to create JSON directory: {self.json_dir}")

    def _get_highest_image_number(self) -> int:
        """Get the highest existing image number for sequencing."""
        try:
            existing = [
                f.stem for f in self.images_dir.glob("image_*.*")
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
            ]
            numbers = [
                int(stem.replace("image_", ""))
                for stem in existing
                if stem.replace("image_", "").isdigit()
            ]
            return max(numbers, default=0)
        except Exception as e:
            logger.warning(f"Could not determine highest image number: {e}")
            return 0

    def save_results(
            self,
            image: np.ndarray,
            detections: List[Dict[str, Any]],
            all_labels: List[str],
            confidence_threshold: float,
            show_images: bool = True
    ) -> Tuple[Path, Path, Path]:
        """Save all detection results with comprehensive validation."""
        try:
            if not SystemUtils.check_disk_space(str(self.output_dir)):
                raise RuntimeError("Insufficient disk space")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._last_image_number += 1
            img_number = self._last_image_number

            # Save paths
            image_path = self.images_dir / f"image_{img_number}.jpg"
            json_path = self.json_dir / f"image_{img_number}.json"

            # Save image
            if not FileHandler.safe_image_write(str(image_path), image):
                raise RuntimeError("Failed to save image")

            # Save JSON
            self._save_json(json_path, timestamp, detections)

            # Update CSV
            self._update_csv(timestamp, image_path.name, json_path.name, all_labels, confidence_threshold)

            # Optionally show image
            if show_images:
                self._show_image(image_path)

            logger.info(f"Results saved successfully: {image_path}")
            return image_path, json_path, self.csv_path

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

    def _save_json(self, path: Path, timestamp: str, detections: List[Dict[str, Any]]):
        """Save detection results to JSON file."""
        try:
            with open(path, 'w') as json_file:
                json.dump({
                    "timestamp": timestamp,
                    "detections": detections,
                    "metadata": {
                        "version": "1.0",
                        "detection_count": len(detections)
                    }
                }, json_file, indent=4, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save JSON: {e}")

    def _update_csv(
            self,
            timestamp: str,
            image_name: str,
            json_name: str,
            all_labels: List[str],
            confidence_threshold: float
    ):
        """Update or create the CSV results file."""
        try:
            label_counts = Counter(all_labels)
            formatted_labels = sorted(label_counts.keys())
            total_detected = len(formatted_labels)
            total_objects = len(all_labels)

            percentage_detected = (total_detected / len(label_counts) * 100) if label_counts else 0

            file_exists = self.csv_path.exists()

            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "image_name", "json_file",
                        "detected_classes", "total_detected_classes",
                        "total_objects", "percentage_detected",
                        "conf_threshold"
                    ])
                writer.writerow([
                    timestamp,
                    image_name,
                    json_name,
                    ", ".join(formatted_labels),
                    total_detected,
                    total_objects,
                    f"{percentage_detected:.2f}%",
                    f"{confidence_threshold:.2f}"
                ])
        except Exception as e:
            raise RuntimeError(f"Failed to update CSV: {e}")

    def _show_image(self, image_path: Path):
        """Open the detected image in default viewer."""
        try:
            if platform.system() == 'Darwin':  # macOS
                os.system(f'open "{image_path}"')
            elif platform.system() == 'Windows':
                os.startfile(image_path)  # type: ignore
            else:  # Linux variants
                webbrowser.open(f'file://{image_path.resolve()}')
        except Exception as e:
            logger.warning(f"Could not open image preview: {e}")


def main():
    """Main execution function with comprehensive error handling."""
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE,
                        help="Confidence threshold (0-1)", metavar="[0.0-1.0]")
    parser.add_argument("--image", type=str,
                        help="Path to an image file (skip camera capture)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save results")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable image preview (headless mode)")
    parser.add_argument("--cam-width", type=int, default=DEFAULT_CAMERA_RESOLUTION[0],
                        help=f"Camera width (default: {DEFAULT_CAMERA_RESOLUTION[0]})")
    parser.add_argument("--cam-height", type=int, default=DEFAULT_CAMERA_RESOLUTION[1],
                        help=f"Camera height (default: {DEFAULT_CAMERA_RESOLUTION[1]})")

    try:
        args = parser.parse_args()

        # Validate confidence threshold
        if not 0 <= args.conf <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        # Initialize components
        detector = ObjectDetector(args.model)
        result_saver = ResultSaver(args.output)

        # Get input image
        if args.image:
            if not FileHandler.validate_image_path(args.image):
                return

            original_img = cv2.imread(args.image)
            if original_img is None:
                raise RuntimeError("Failed to read image despite validation")
        else:
            with CameraHandler(args.cam_width, args.cam_height) as camera:
                original_img = camera.capture_image()
                if original_img is None:
                    return

        # Process image
        start_time = time.time()
        detections, filtered_image, all_labels = detector.detect_objects(
            original_img,
            args.conf
        )

        # Save results
        result_saver.save_results(
            filtered_image,
            detections,
            all_labels,
            args.conf,
            not args.no_show
        )

        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

    except argparse.ArgumentError as e:
        logger.error(f"Argument error: {e}")
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()