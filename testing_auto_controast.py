import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (change the path to your image file)
image_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\200.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. **Contrast Stretching**
def contrast_stretching(img):
    min_pixel = np.min(img)
    max_pixel = np.max(img)
    stretched_img = ((img - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)
    return stretched_img

# 2. **Histogram Equalization**
def histogram_equalization(img):
    equalized_img = cv2.equalizeHist(img)
    return equalized_img

# 3. **Adaptive Histogram Equalization**
def adaptive_histogram_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # You can adjust the clipLimit and tileGridSize
    adaptive_eq_img = clahe.apply(img)
    return adaptive_eq_img

# Apply all methods
contrast_stretched = contrast_stretching(image)
hist_eq = histogram_equalization(image)
adaptive_eq = adaptive_histogram_equalization(image)

# Plot original and processed images
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(contrast_stretched, cmap='gray')
axs[0, 1].set_title("Contrast Stretching")
axs[0, 1].axis('off')

axs[1, 0].imshow(hist_eq, cmap='gray')
axs[1, 0].set_title("Histogram Equalization")
axs[1, 0].axis('off')

axs[1, 1].imshow(adaptive_eq, cmap='gray')
axs[1, 1].set_title("Adaptive Histogram Equalization")
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
