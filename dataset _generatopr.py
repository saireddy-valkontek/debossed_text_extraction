import os
import random
from PIL import Image
import shutil
import numpy as np


def random_crop(image, crop_width, crop_height):
    """
    Randomly crop the image to a given width and height.
    """
    img_width, img_height = image.size
    left = random.randint(0, img_width - crop_width)
    top = random.randint(0, img_height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def adjust_labels(labels, crop_box, img_width, img_height):
    """
    Adjust the labels for the cropped image.
    Only keep the boxes that are within the crop region.
    """
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    new_labels = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        # Convert from relative coordinates to absolute
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

        # Check if the bounding box is within the crop
        if (crop_left <= x_center_abs <= crop_right) and (crop_top <= y_center_abs <= crop_bottom):
            # Adjust the bounding box to fit the cropped region
            new_x_center = (x_center_abs - crop_left) / (crop_right - crop_left)
            new_y_center = (y_center_abs - crop_top) / (crop_bottom - crop_top)
            new_width = width_abs / (crop_right - crop_left)
            new_height = height_abs / (crop_bottom - crop_top)
            new_labels.append([class_id, new_x_center, new_y_center, new_width, new_height])
    return new_labels


def save_cropped_image_and_labels(image, labels, output_image_path, output_label_path):
    """
    Save the cropped image and its corresponding labels to the specified paths.
    """
    image.save(output_image_path)

    with open(output_label_path, 'w') as f:
        for label in labels:
            class_id, x_center, y_center, width, height = label
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def generate_crops(input_image_path, input_label_path, output_image_dir, output_label_dir, num_crops=10,
                   crop_size=(416, 416)):
    """
    Generate random crops for an image and its labels.
    """
    # Load image and labels
    image = Image.open(input_image_path)
    img_width, img_height = image.size

    with open(input_label_path, 'r') as f:
        labels = []
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append([class_id, x_center, y_center, width, height])

    # Create crops
    for i in range(num_crops):
        # Generate random crop size within the bounds of the original image
        crop_width = random.randint(int(crop_size[0] * 0.8), crop_size[0])
        crop_height = random.randint(int(crop_size[1] * 0.8), crop_size[1])

        # Crop the image
        cropped_image, crop_box = random_crop(image, crop_width, crop_height)

        # Adjust labels based on the crop
        adjusted_labels = adjust_labels(labels, crop_box, img_width, img_height)

        # Generate output filenames
        base_filename = os.path.splitext(os.path.basename(input_image_path))[0]
        cropped_image_filename = f"{base_filename}_crop_{i}.jpg"
        cropped_label_filename = f"{base_filename}_crop_{i}.txt"

        output_image_path = os.path.join(output_image_dir, cropped_image_filename)
        output_label_path = os.path.join(output_label_dir, cropped_label_filename)

        # Save the cropped image and corresponding labels
        save_cropped_image_and_labels(cropped_image, adjusted_labels, output_image_path, output_label_path)
        print(f"Saved cropped image and label: {output_image_path}, {output_label_path}")


# Example usage
input_image_dir = 'path/to/images'  # Directory containing your original images
input_label_dir = 'path/to/labels'  # Directory containing your original labels.txt files
output_image_dir = 'path/to/output_images'  # Directory to save cropped images
output_label_dir = 'path/to/output_labels'  # Directory to save cropped labels

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

for image_file in os.listdir(input_image_dir):
    if image_file.endswith('.jpg') or image_file.endswith('.png'):  # Adjust as needed
        image_path = os.path.join(input_image_dir, image_file)
        label_path = os.path.join(input_label_dir, os.path.splitext(image_file)[0] + '.txt')

        # Generate crops for each image
        generate_crops(image_path, label_path, output_image_dir, output_label_dir, num_crops=10, crop_size=(416, 416))
