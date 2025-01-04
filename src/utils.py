import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def split_dataset(gray_images, color_images, test_size=0.2, val_size=0.25):
    gray_train, gray_test, color_train, color_test = train_test_split(
        gray_images, color_images, test_size=test_size, random_state=42
    )
    gray_train, gray_val, color_train, color_val = train_test_split(
        gray_train, color_train, test_size=val_size, random_state=42
    )
    return gray_train, gray_val, gray_test, color_train, color_val, color_test

def visualize_predictions(gray_images, pred_images, gt_images, num_images=5):
    for i in range(num_images):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(gray_images[i].squeeze(), cmap='gray')
        plt.title("Grayscale Input")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_images[i])
        plt.title("Predicted Colorization")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(gt_images[i])
        plt.title("Ground Truth")
        plt.axis('off')

        plt.show()