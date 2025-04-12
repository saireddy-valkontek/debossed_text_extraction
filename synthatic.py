import cv2
import numpy as np
import random
import string
from PIL import Image, ImageDraw, ImageFont
import os

# Set up folders
os.makedirs("synthetic_plates/images", exist_ok=True)
os.makedirs("synthetic_plates/labels", exist_ok=True)

FONT_PATH = "arialbd.ttf"  # You can replace with a font that resembles the real plate
CHARS = string.ascii_uppercase + string.digits

def random_plate_text(length=4):
    return ''.join(random.choices(CHARS, k=length))

def generate_plate_image(text, idx):
    # Create blank image
    img = np.ones((80, 250, 3), dtype=np.uint8) * 255

    # Draw text
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_PATH, 40)
    w, h = draw.textsize(text, font=font)
    draw.text(((250 - w)//2, (80 - h)//2), text, font=font, fill=(0, 0, 0))

    # Convert back to OpenCV
    img = np.array(pil_img)

    # Simulate glare / noise
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    if random.random() > 0.5:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    # Save
    img_path = f"synthetic_plates/images/plate_{idx}.png"
    label_path = f"synthetic_plates/labels/plate_{idx}.txt"
    cv2.imwrite(img_path, img)
    with open(label_path, 'w') as f:
        f.write(text)

    print(f"[+] Saved {text} -> {img_path}")

# Generate 1000 synthetic samples
for i in range(1000):
    text = random_plate_text(random.randint(3,16))
    generate_plate_image(text, i)
