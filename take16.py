import os
import cv2
import numpy as np
import yaml


# Function to convert YOLO annotations to EAST-compatible bounding boxes
def convert_yolo_to_bboxes(yolo_label_path, image_width, image_height):
    bboxes = []
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert YOLO format [x_center, y_center, width, height] to [x1, y1, x2, y2]
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        bboxes.append([x1, y1, x2, y2])

    return bboxes


# Function to generate EAST dataset (bounding boxes in .txt format)
def generate_east_dataset(yolo_dataset_path, output_path):
    # Load data.yaml file
    with open(os.path.join(yolo_dataset_path, 'data.yaml'), 'r') as file:
        data = yaml.safe_load(file)

    # Process each image in the train folder
    train_images_path = os.path.join(yolo_dataset_path, 'train/images')
    labels_path = os.path.join(yolo_dataset_path, 'train/labels')

    for image_file in os.listdir(train_images_path):
        if image_file.endswith('.jpg'):  # or other image formats
            image_path = os.path.join(train_images_path, image_file)
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # Process corresponding label file
            label_file = image_file.replace('.jpg', '.txt')
            yolo_label_path = os.path.join(labels_path, label_file)

            bboxes = convert_yolo_to_bboxes(yolo_label_path, image_width, image_height)

            # Save the bounding boxes for EAST
            east_annotation_path = os.path.join(output_path, 'annotations', label_file)
            with open(east_annotation_path, 'w') as f:
                for bbox in bboxes:
                    f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


# Function to generate CRAFT dataset (bounding boxes in .txt format + masks)
def generate_craft_dataset(yolo_dataset_path, output_path):
    # Load data.yaml file
    with open(os.path.join(yolo_dataset_path, 'data.yaml'), 'r') as file:
        data = yaml.safe_load(file)

    # Process each image in the train folder
    train_images_path = os.path.join(yolo_dataset_path, 'train/images')
    labels_path = os.path.join(yolo_dataset_path, 'train/labels')

    for image_file in os.listdir(train_images_path):
        if image_file.endswith('.jpg'):  # or other image formats
            image_path = os.path.join(train_images_path, image_file)
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # Process corresponding label file
            label_file = image_file.replace('.jpg', '.txt')
            yolo_label_path = os.path.join(labels_path, label_file)

            bboxes = convert_yolo_to_bboxes(yolo_label_path, image_width, image_height)

            # Save the bounding boxes for CRAFT
            craft_annotation_path = os.path.join(output_path, 'annotations', label_file)
            with open(craft_annotation_path, 'w') as f:
                for bbox in bboxes:
                    f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

            # Generate pixel-level masks for CRAFT
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for bbox in bboxes:
                # Ensure that bounding boxes are correctly drawn in the mask
                cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)  # Draw filled rectangle

            # Saving the mask (making sure we save the mask with the right extension)
            mask_path = os.path.join(output_path, 'masks', label_file.replace('.txt', '.png'))
            cv2.imwrite(mask_path, mask)


# Paths to your YOLO dataset and output directories
yolo_dataset_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Datasets\image_to_text_extraction\yolov8.dataset\v3"
output_path = r"C:\path_to_output_dataset"

# Create output directories
os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)

# Generate EAST and CRAFT datasets
generate_east_dataset(yolo_dataset_path, output_path)
generate_craft_dataset(yolo_dataset_path, output_path)