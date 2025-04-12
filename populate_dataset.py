import cv2
import os
import numpy as np
import random
from tqdm import tqdm

# -------------------------------
# Augmentation Functions
# -------------------------------

def add_random_shadow(image):
    height, width = image.shape[:2]
    top_x, top_y = random.randint(0, width), 0
    bot_x, bot_y = random.randint(0, width), height
    shadow_mask = np.zeros_like(image, dtype=np.uint8)

    polygon = np.array([[top_x, top_y], [bot_x, bot_y],
                        [bot_x + 100, bot_y], [top_x + 100, top_y]], np.int32)
    cv2.fillPoly(shadow_mask, [polygon], (50, 50, 50))
    return cv2.addWeighted(image, 1, shadow_mask, 0.5, 0)

def add_reflection(image, intensity=0.2):
    height, width = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)
    x = random.randint(int(width * 0.1), int(width * 0.9))
    y = random.randint(int(height * 0.1), int(height * 0.9))
    radius = random.randint(30, 100)
    cv2.circle(overlay, (x, y), radius, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (101, 101), 0)
    return cv2.addWeighted(image, 1, overlay, intensity, 0)

def add_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def add_noise(image):
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def apply_augmentations(image):
    img = image.copy()
    if random.random() < 0.5:
        img = add_random_shadow(img)
    if random.random() < 0.5:
        img = add_reflection(img, intensity=random.uniform(0.1, 0.3))
    if random.random() < 0.5:
        img = add_blur(img)
    if random.random() < 0.5:
        img = add_noise(img)
    return img

# -------------------------------
# Dataset Augmentation Runner
# -------------------------------

def augment_dataset(
    input_folder='original_images',
    output_folder='augmented_images',
    copies_per_image=10
):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_files)} images. Generating {copies_per_image} augmentations each.")

    for image_file in tqdm(image_files):
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)

        for i in range(copies_per_image):
            aug_img = apply_augmentations(img)
            out_name = f"{os.path.splitext(image_file)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, out_name), aug_img)

    print("✅ Augmentation complete!")

# -------------------------------
# Run It
# -------------------------------

if __name__ == "__main__":
    augment_dataset(
        input_folder=r'C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers',     # Folder with your 500 base images
        output_folder=r'C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\augmented', # Will be created if it doesn’t exist
        copies_per_image=10                 # 500 x 10 = 5,000 total images
    )