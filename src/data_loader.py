# src/data_loader.py

import os
import cv2
import numpy as np

def data_generator(
    color_path='data/raw/archive/landscapeImages/color',
    gray_path='data/raw/archive/landscapeImages/gray',
    target_size=(256, 256),
    batch_size=8,
    # max_samples=500  # Limit the number of samples for testing. DELETE LATER
):
    """
    Generator that yields batches of grayscale and color images.
    """
    # color_files = sorted(os.listdir(color_path))[:max_samples] For testing DELETE LATER
    # gray_files = sorted(os.listdir(gray_path))[:max_samples] For testing DELETE LATER
    color_files = sorted(os.listdir(color_path))
    gray_files = sorted(os.listdir(gray_path))
    num_samples = len(color_files)

    while True:  # Infinite loop for generator
        for i in range(0, num_samples, batch_size):
            batch_color = []
            batch_gray = []

            for j in range(i, min(i + batch_size, num_samples)):
                # Load color image
                color_img = cv2.imread(os.path.join(color_path, color_files[j]))
                color_img = cv2.resize(color_img, target_size).astype(np.float32) / 255.0

                # Load grayscale image
                gray_img = cv2.imread(os.path.join(gray_path, gray_files[j]), cv2.IMREAD_GRAYSCALE)
                gray_img = cv2.resize(gray_img, target_size).astype(np.float32) / 255.0
                gray_img = np.expand_dims(gray_img, axis=-1)  # Add channel dimension

                batch_color.append(color_img)
                batch_gray.append(gray_img)

            yield np.array(batch_gray), np.array(batch_color)



def load_images(
    color_path='data/raw/archive/landscapeImages/color',
    gray_path='data/raw/archive/landscapeImages/gray',
    target_size=(256, 256)
):
    """
    Loads and resizes paired color and grayscale images.
    Returns two NumPy arrays: (gray_images, color_images).
    """
    color_files = sorted(os.listdir(color_path))
    gray_files = sorted(os.listdir(gray_path))

    color_images = []
    gray_images = []

    for cf, gf in zip(color_files, gray_files):
        # Load color image
        color_img = cv2.imread(os.path.join(color_path, cf))
        color_img = cv2.resize(color_img, target_size).astype(np.float32) / 255.0

        # Load grayscale image
        gray_img = cv2.imread(os.path.join(gray_path, gf), cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, target_size).astype(np.float32) / 255.0
        gray_img = np.expand_dims(gray_img, axis=-1)  # Add channel dimension

        color_images.append(color_img)
        gray_images.append(gray_img)

    return np.array(gray_images), np.array(color_images)
