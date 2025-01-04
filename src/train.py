# src/train.py

import os
import numpy as np
# from sklearn.model_selection import train_test_split
# from data_loader import load_images
from model import build_colorization_model
from data_loader import load_images, data_generator
from tensorflow import keras

class SaveIntermediateModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:  # Save model every 10 epochs
            model_path = f'models/colorization_model_epoch_{epoch + 1}.keras'
            print(f"Saving model to {model_path}")
            self.model.save(model_path)

def train_model(
    color_path='data/raw/archive/landscapeImages/color',
    gray_path='data/raw/archive/landscapeImages/gray',
    # epochs=5, small for testing DELETE LATER
    # batch_size=8, small for testing DELETE LATER
    epochs=50,
    batch_size=16,
    model_save_path='models/colorization_model.keras'
):
    """
    Trains a colorization model using data generators.
    """
    # Build model
    model = build_colorization_model(input_shape=(256, 256, 1))
    model.summary()

    # Initialize data generators
    train_gen = data_generator(color_path, gray_path, batch_size=batch_size)
    val_gen = data_generator(color_path, gray_path, batch_size=batch_size)

    # Calculate steps per epoch
    num_samples = len(os.listdir(color_path))
    steps_per_epoch = int(0.8 * num_samples) // batch_size  # 80% for  UNCOMMENT LATER
    validation_steps = int(0.2 * num_samples) // batch_size  # 20% for validation UNCOMMENT LATER
    # steps_per_epoch = int(0.2 * num_samples) // batch_size  # Use 20% for training DELETE LATER
    # validation_steps = int(0.1 * num_samples) // batch_size  # Use 10% for validation DELETE LATER

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        SaveIntermediateModel()
    ]

    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        # steps_per_epoch=10,  # Run only 10 batches for testing
        validation_data=val_gen,
        validation_steps=validation_steps, # for testing UNCOMMENT LATER
        # validation_steps=5,  # Run only 5 batches
        epochs=epochs, # for testing UNCOMMENT LATER
        # epochs=1  # Run only 1 epoch for testing
        callbacks=callbacks
    )

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    return history


if __name__ == "__main__":
    train_model()



# def train_model(
#     color_path='data/raw/archive/landscapeImages/color',
#     gray_path='data/raw/archive/landscapeImages/gray',
#     epochs=20,
#     batch_size=8,
#     model_save_path='models/colorization_model.keras'
# ):
#     # Build model
#     model = build_colorization_model(input_shape=(256, 256, 1))
#     model.summary()

#     # Initialize data generators
#     train_gen = data_generator(color_path, gray_path, batch_size=batch_size)
#     val_gen = data_generator(color_path, gray_path, batch_size=batch_size)

#     # Calculate steps per epoch
#     num_samples = len(os.listdir(color_path))
#     steps_per_epoch = num_samples // batch_size
#     validation_steps = int(0.2 * num_samples) // batch_size

#     # Train model
#     history = model.fit(
#         train_gen,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=val_gen,
#         validation_steps=validation_steps,
#         epochs=epochs
#     )

#     # Save model
#     os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
#     model.save(model_save_path)

#     return history


# def train_model(
#     color_path='data/raw/archive/landscapeImages/color',
#     gray_path='data/raw/archive/landscapeImages/gray',
#     epochs=20,
#     batch_size=8,
#     model_save_path='models/colorization_model.keras'
# ):
#     # 1. Load images
#     gray_images, color_images = load_images(color_path, gray_path)

#     # 2. Split dataset (train/val)
#     gray_train, gray_val, color_train, color_val = train_test_split(
#         gray_images, color_images, test_size=0.2, random_state=42
#     )

#     # 3. Build model
#     model = build_colorization_model(input_shape=(256, 256, 1))
#     model.summary()

#     # 4. Train model
#     history = model.fit(
#         gray_train,
#         color_train,
#         validation_data=(gray_val, color_val),
#         epochs=epochs,
#         batch_size=batch_size
#     )

#     # 5. Save model
#     os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
#     model.save(model_save_path)

#     return history

# if __name__ == "__main__":
#     train_model()
