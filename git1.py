import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50

# --------- Load Pre-trained Segmentation Model ---------
model = fcn_resnet50(pretrained=True).eval()

# --------- Load and Preprocess Input Image ---------
# Replace with your image file path
img_path = r"C:\vltk sai reddy\OneDrive - VALKONTEK EMBEDDED IOT SERVICES PRIVATE LTD\Resources\Images\Engine_Chassis_Numbers\240.jpg"
original = cv2.imread(img_path)

if original is None:
    raise FileNotFoundError(f"Image not found at path: {img_path}")

# Convert to RGB
image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Resize and normalize
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

input_tensor = transform(image_rgb).unsqueeze(0)  # [1, 3, H, W]

# --------- Run Inference ---------
with torch.no_grad():
    output = model(input_tensor)['out'][0]
seg_mask = output.argmax(0).byte().cpu().numpy()  # [H, W]

# Resize mask to original size
seg_mask_resized = cv2.resize(seg_mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

# --------- Optional: Highlight Possible Engraved Areas ---------
highlighted = original.copy()
highlighted[seg_mask_resized == 15] = [0, 255, 0]  # Highlight "person" class (often used for text)

# --------- Show Results ---------
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmentation Mask")
plt.imshow(seg_mask_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Highlighted Engraved Regions")
plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()