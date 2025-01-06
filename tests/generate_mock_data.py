import os
import numpy as np
import cv2

test_dir = "tests/mock_data"
color_path = os.path.join(test_dir, "color")
gray_path = os.path.join(test_dir, "gray")
os.makedirs(color_path, exist_ok=True)
os.makedirs(gray_path, exist_ok=True)

for i in range(5):  # Create 5 color and grayscale images
    color_img = np.ones((100, 100, 3), dtype=np.uint8) * (i + 1) * 50
    gray_img = np.ones((100, 100), dtype=np.uint8) * (i + 1) * 50

    cv2.imwrite(os.path.join(color_path, f"color_{i}.png"), color_img)
    cv2.imwrite(os.path.join(gray_path, f"gray_{i}.png"), gray_img)

print("Mock data created successfully.")
