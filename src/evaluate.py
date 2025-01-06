import os
import numpy as np
import cv2
from model import build_colorization_model
from data_loader import load_images, data_generator
import matplotlib.pyplot as plt

def get_latest_model(models_dir='models'):
    """Retrieve the most recently modified model file in the models directory to evaluate."""
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.keras')]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def visualize_predictions(gray_images, pred_images, gt_images, num_images=5):
    for i in range(num_images):
        plt.figure(figsize=(15, 5))

        # Grayscale input
        plt.subplot(1, 3, 1)
        plt.imshow(gray_images[i].squeeze(), cmap='gray')
        plt.title("Grayscale Input")
        plt.axis('off')

        # Predicted colorization
        plt.subplot(1, 3, 2)
        plt.imshow(pred_images[i])
        plt.title("Predicted Colorization")
        plt.axis('off')

        # Actual original colored image
        plt.subplot(1, 3, 3)
        plt.imshow(gt_images[i])
        plt.title("Ground Truth")
        plt.axis('off')

        plt.show()

def evaluate_model(
    # model_path='models/colorization_model_epoch_21.keras',
    model_path=None,
    color_path='data/raw/archive/landscapeImages/color',
    gray_path='data/raw/archive/landscapeImages/gray',
    output_dir='outputs/predictions',
    batch_size=8,
    max_samples=10  # Adjust for visualization (depends on mem)
):
    print("Starting evaluation...")

    if not model_path:
        model_path = get_latest_model()
    print(f"Loading model from {model_path}")

    # print(f"Loading model from {model_path}")
    # model = build_colorization_model(input_shape=(128, 128, 1))  # Matches input shape
    model = build_colorization_model(input_shape=(256, 256, 1))  # Matches input shape
    model.load_weights(model_path)
    print("Model loaded successfully.")

    print(f"Loading a subset of data for visualization (max_samples={max_samples})")
    # gray_images, gt_images = load_images(color_path, gray_path, target_size=(128, 128))
    gray_images, gt_images = load_images(color_path, gray_path, target_size=(256, 256))


    predictions = model.predict(gray_images[:max_samples])
    pred_images = np.clip(predictions, 0, 1)  # Ensures valid pixel values

    print(f"Saving predictions to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(max_samples, len(pred_images))):
        gray_img = (gray_images[i] * 255.0).astype('uint8').squeeze()
        pred_img = (pred_images[i] * 255.0).astype('uint8')
        gt_img = (gt_images[i] * 255.0).astype('uint8')

        cv2.imwrite(os.path.join(output_dir, f"gray_{i}.png"), gray_img)
        cv2.imwrite(os.path.join(output_dir, f"pred_{i}.png"), pred_img)
        cv2.imwrite(os.path.join(output_dir, f"gt_{i}.png"), gt_img)

    print(f"Sample predictions saved in {output_dir}.")

    print("Visualizing predictions...")
    visualize_predictions(gray_images[:max_samples], pred_images, gt_images[:max_samples])

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_model(model_path=model_path)
