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
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# [Previous constants remain the same...]
SUPPORTED_INPUT_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')


class ImageConverter:
    """Handles image format conversion with validation and error handling"""

    @staticmethod
    def convert_to_jpg(input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        Convert an image to JPG format with validation.

        Args:
            input_path: Path to source image
            output_path: Path to save JPG image

        Returns:
            bool: True if conversion succeeded, False otherwise
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)

            # Validate input
            if not input_path.exists():
                logger.error(f"Input image not found: {input_path}")
                return False

            if not input_path.suffix.lower() in SUPPORTED_INPUT_IMAGE_EXTENSIONS:
                logger.error(f"Unsupported input format: {input_path.suffix}")
                return False

            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Perform conversion
            with Image.open(input_path) as img:
                # Convert to RGB if needed (JPEG doesn't support alpha channel)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save as JPEG with quality=95 (adjustable)
                img.save(output_path, 'JPEG', quality=95)

            # Verify output
            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(f"Failed to verify output image: {output_path}")
                return False

            logger.info(f"Converted {input_path} to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Conversion failed for {input_path}: {e}")
            return False

    @staticmethod
    def batch_convert_to_jpg(
            input_folder: Union[str, Path],
            output_folder: Union[str, Path],
            max_workers: int = 4
    ) -> Tuple[int, int]:
        """
        Convert all images in a folder to JPG format with parallel processing.

        Args:
            input_folder: Folder containing source images
            output_folder: Folder to save JPG images
            max_workers: Maximum parallel conversion threads

        Returns:
            Tuple[int, int]: (success_count, failure_count)
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)

        if not input_folder.exists():
            logger.error(f"Input folder not found: {input_folder}")
            return (0, 0)

        # Create output directory if needed
        output_folder.mkdir(parents=True, exist_ok=True)

        success_count = 0
        failure_count = 0

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for input_path in input_folder.iterdir():
                if input_path.suffix.lower() in SUPPORTED_INPUT_IMAGE_EXTENSIONS:
                    output_path = output_folder / f"{input_path.stem}.jpg"
                    futures.append(executor.submit(
                        ImageConverter.convert_to_jpg,
                        input_path,
                        output_path
                    ))

            for future in futures:
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1

        logger.info(f"Batch conversion complete: {success_count} succeeded, {failure_count} failed")
        return (success_count, failure_count)


class ImageHandler:
    """Enhanced ImageHandler with conversion support"""

    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load an image from file with format conversion if needed."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return None

            # Convert non-JPG images to temporary JPG first
            if image_path.suffix.lower() not in ('.jpg', '.jpeg'):
                temp_jpg = Path("temp_conversion.jpg")
                if ImageConverter.convert_to_jpg(image_path, temp_jpg):
                    try:
                        img = cv2.imread(str(temp_jpg))
                        if img is not None:
                            return img
                    finally:
                        try:
                            temp_jpg.unlink()
                        except:
                            pass
                return None

            # Directly load JPG images
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Failed to read image")
            return img

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    # [Rest of the existing ImageHandler methods remain the same...]


# [Previous class definitions remain the same...]


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline with Image Conversion")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE,
                        help="Confidence threshold (0-1)", metavar="[0.0-1.0]")
    parser.add_argument("--image", type=str,
                        help="Path to an image file (skip camera capture)")
    parser.add_argument("--image-folder", type=str,
                        help="Folder containing images to process (batch mode)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save results")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable image preview (headless mode)")
    parser.add_argument("--cam-width", type=int, default=DEFAULT_CAMERA_RESOLUTION[0],
                        help=f"Camera width (default: {DEFAULT_CAMERA_RESOLUTION[0]})")
    parser.add_argument("--cam-height", type=int, default=DEFAULT_CAMERA_RESOLUTION[1],
                        help=f"Camera height (default: {DEFAULT_CAMERA_RESOLUTION[1]})")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert images to JPG without detection")

    try:
        args = parser.parse_args()

        # Validate confidence threshold
        if not 0 <= args.conf <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        # Handle conversion-only mode
        if args.convert_only:
            if not args.image_folder:
                raise ValueError("--image-folder required for conversion-only mode")

            success, failures = ImageConverter.batch_convert_to_jpg(
                args.image_folder,
                Path(args.output) / "converted_images"
            )
            logger.info(f"Conversion completed: {success} succeeded, {failures} failed")
            return

        # Initialize components
        detector = ObjectDetector(args.model)
        result_saver = ResultSaver(args.output)

        # Handle batch processing
        if args.image_folder:
            input_folder = Path(args.image_folder)
            for image_path in input_folder.iterdir():
                if image_path.suffix.lower() in SUPPORTED_INPUT_IMAGE_EXTENSIONS:
                    logger.info(f"Processing {image_path.name}...")
                    process_single_image(image_path, detector, result_saver, args)
            return

        # Handle single image processing
        if args.image:
            process_single_image(args.image, detector, result_saver, args)
        else:
            with CameraHandler(args.cam_width, args.cam_height) as camera:
                original_img = camera.capture_image()
                if original_img is not None:
                    process_single_image(original_img, detector, result_saver, args)

    except argparse.ArgumentError as e:
        logger.error(f"Argument error: {e}")
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()


def process_single_image(
        image_input: Union[str, Path, np.ndarray],
        detector: ObjectDetector,
        result_saver: ResultSaver,
        args: argparse.Namespace
):
    """Process a single image through the detection pipeline."""
    try:
        # Load or use the provided image
        if isinstance(image_input, (str, Path)):
            original_img = ImageHandler.load_image(str(image_input))
            if original_img is None:
                return
        else:
            original_img = image_input

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

        logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing image: {e}")


if __name__ == "__main__":
    main()