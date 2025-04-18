import os
import cv2

def crop_and_create_new_dataset(image_folder, label_folder, output_folder):
    # Ensure the output directories exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, "images")):
        os.makedirs(os.path.join(output_folder, "images"))
    if not os.path.exists(os.path.join(output_folder, "labels")):
        os.makedirs(os.path.join(output_folder, "labels"))

    # Iterate through each image in the folder
    for image_name in os.listdir(image_folder):
        # Check if it's an image file
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            # Read the image
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)

            # Remove the extension and handle the filename correctly
            base_name = os.path.splitext(image_name)[0]

            # Get corresponding label file (same base name, .txt extension)
            label_name = base_name + '.txt'
            label_path = os.path.join(label_folder, label_name)

            if not os.path.exists(label_path):
                print(f"Label file not found for {image_name}, skipping...")
                continue

            # Read the label file
            with open(label_path, 'r') as f:
                labels = f.readlines()

            # Iterate through each bounding box in the label file
            for i, label in enumerate(labels):
                # Parse the label: class_id center_x center_y width height
                parts = label.strip().split()
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert YOLO coordinates to pixel values
                image_height, image_width, _ = image.shape
                x_min = int((center_x - width / 2) * image_width)
                x_max = int((center_x + width / 2) * image_width)
                y_min = int((center_y - height / 2) * image_height)
                y_max = int((center_y + height / 2) * image_height)

                # Crop the character from the image
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Check if cropping results in a valid region
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    # Save the cropped image
                    cropped_image_name = f"{base_name}_{i}.jpg"
                    cropped_image_path = os.path.join(output_folder, "images", cropped_image_name)
                    cv2.imwrite(cropped_image_path, cropped_image)

                    # Create the corresponding label file for the cropped image
                    cropped_label_name = f"{base_name}_{i}.txt"
                    cropped_label_path = os.path.join(output_folder, "labels", cropped_label_name)

                    # Write the new label (class_id, center_x, center_y, width, height) in YOLO format
                    cropped_center_x = 0.5  # Center of the cropped image (always 0.5 for single object)
                    cropped_center_y = 0.5  # Center of the cropped image (always 0.5 for single object)
                    cropped_width = 1.0     # Width is 1.0 (single object in cropped image)
                    cropped_height = 1.0    # Height is 1.0 (single object in cropped image)

                    with open(cropped_label_path, 'w') as f_label:
                        f_label.write(f"{class_id} {cropped_center_x} {cropped_center_y} {cropped_width} {cropped_height}\n")

                    print(f"Cropped and saved: {cropped_image_name}, Label: {cropped_label_name}")

    print("Cropping and dataset creation complete!")


# Example usage:
image_folder = r"C:\Users\sai\Downloads\chassis_number.v4i.yolov8\test\images"  # Folder containing original images
label_folder = r"C:\Users\sai\Downloads\chassis_number.v4i.yolov8\test\labels"  # Folder containing YOLO label files
output_folder = r"C:\Users\sai\Desktop\yolo_data\test"  # Folder to save cropped images and labels


crop_and_create_new_dataset(image_folder, label_folder, output_folder)