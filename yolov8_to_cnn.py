import os
import cv2
import yaml

# Paths
base_path = r"/Users/sai/Developer/Resources/datasets/yolov8_dataset"
split =   "test"

images_dir = os.path.join(base_path, split, "images")
labels_dir = os.path.join(base_path, split, "labels")
output_dir = os.path.join("classification_dataset", split)

# Load label names from YAML
with open(os.path.join(base_path, "data.yaml"), "r") as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml["names"]

# Create output dir
os.makedirs(output_dir, exist_ok=True)

# Loop through label files
for label_file in os.listdir(labels_dir):
    if not label_file.endswith(".txt"):
        continue

    image_name = label_file.replace(".txt", ".jpg")  # or .png if needed
    image_path = os.path.join(images_dir, image_name)
    label_path = os.path.join(labels_dir, label_file)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        continue
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])
        
        # Convert YOLO coords to pixels
        x1 = int((x_center - bbox_w / 2) * w)
        y1 = int((y_center - bbox_h / 2) * h)
        x2 = int((x_center + bbox_w / 2) * w)
        y2 = int((y_center + bbox_h / 2) * h)

        # Clamp coords to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop and save
        crop = img[y1:y2, x1:x2]
        class_label = class_names[class_id]
        cls_folder = os.path.join(output_dir, class_label)
        os.makedirs(cls_folder, exist_ok=True)
        out_path = os.path.join(cls_folder, f"{label_file[:-4]}_{i}.jpg")
        cv2.imwrite(out_path, crop)
