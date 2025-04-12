import os
import cv2
import yaml


def convert_yolo_to_east_format(yolo_label_path, image_path):
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    for line in lines:
        parts = line.strip().split()
        class_id, x_center, y_center, width, height = map(float, parts)

        # Convert YOLO to [x1, y1, x2, y2]
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height

        bboxes.append(f"{int(x1)} {int(y1)} {int(x2)} {int(y2)}")

    return bboxes


def create_east_dataset(yolo_dataset_path, output_path):
    # Load data.yaml
    with open(os.path.join(yolo_dataset_path, 'data.yaml'), 'r') as file:
        data = yaml.safe_load(file)

    # Set image dimensions
    image_dir = os.path.join(yolo_dataset_path, 'train/images')
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):  # or any other image format
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            global image_width, image_height
            image_height, image_width = image.shape[:2]

            # Prepare corresponding label path
            label_path = os.path.join(yolo_dataset_path, 'train/labels', image_file.replace('.jpg', '.txt'))

            bboxes = convert_yolo_to_east_format(label_path, image_path)

            # Save the bbox data to the EAST format text file
            with open(os.path.join(output_path, image_file.replace('.jpg', '.txt')), 'w') as f:
                for bbox in bboxes:
                    f.write(bbox + '\n')


# Set paths
yolo_dataset_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Datasets\image_to_text_extraction\yolov8.dataset\v3"
output_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Datasets\image_to_text_extraction\east_dataset\v1"

# Create EAST dataset
create_east_dataset(yolo_dataset_path, output_path)