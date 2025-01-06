import os
import cv2
import numpy as np

def data_generator(
    color_path='data/raw/archive/landscapeImages/color',
    gray_path='data/raw/archive/landscapeImages/gray',
    target_size=(256, 256),
    batch_size=8,
    max_iterations=None
):

    color_files = sorted(os.listdir(color_path))
    gray_files = sorted(os.listdir(gray_path))
    num_samples = len(color_files)

    iteration = 0
    # print("Starting data_generator...")  # Debugging, delete later

    while True:
        for i in range(0, num_samples, batch_size):
            # print(f"Generating batch: iteration {iteration}, index {i}")  # Debugging line
            if max_iterations is not None and iteration >= max_iterations:
                return  # Exit the generator after max_iterations
            
            batch_color = []
            batch_gray = []

            for j in range(i, min(i + batch_size, num_samples)):
                # Load color image 1
                color_img = cv2.imread(os.path.join(color_path, color_files[j]))
                color_img = cv2.resize(color_img, target_size).astype(np.float32) / 255.0

                # Load grayscale image 1
                gray_img = cv2.imread(os.path.join(gray_path, gray_files[j]), cv2.IMREAD_GRAYSCALE)
                gray_img = cv2.resize(gray_img, target_size).astype(np.float32) / 255.0
                gray_img = np.expand_dims(gray_img, axis=-1)

                batch_color.append(color_img)
                batch_gray.append(gray_img)

            yield np.array(batch_gray), np.array(batch_color)

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                # print("Reached max_iterations. Exiting generator.")  # Debugging line
                return

def load_images(
    color_path='data/raw/archive/landscapeImages/color',
    gray_path='data/raw/archive/landscapeImages/gray',
    target_size=(256, 256)
):

    color_files = sorted(os.listdir(color_path))
    gray_files = sorted(os.listdir(gray_path))

    color_images = []
    gray_images = []

    for cf, gf in zip(color_files, gray_files):
        # Load color image 2 
        color_img = cv2.imread(os.path.join(color_path, cf))
        color_img = cv2.resize(color_img, target_size).astype(np.float32) / 255.0

        # Load grayscale image 2
        gray_img = cv2.imread(os.path.join(gray_path, gf), cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, target_size).astype(np.float32) / 255.0
        gray_img = np.expand_dims(gray_img, axis=-1)

        color_images.append(color_img)
        gray_images.append(gray_img)

    return np.array(gray_images), np.array(color_images)
