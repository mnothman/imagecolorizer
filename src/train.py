import os
import numpy as np
from model import build_colorization_model
from data_loader import load_images, data_generator
from tensorflow import keras

class SaveIntermediateModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            model_path = f'models/colorization_model_epoch_{epoch + 1}.keras'
            print(f"Saving model to {model_path}")
            self.model.save(model_path)

def train_model(
    color_path='data/raw/archive/landscapeImages/color',
    gray_path='data/raw/archive/landscapeImages/gray',
    # epochs=5, small for testing DELETE LATER
    # batch_size=8, small for testing DELETE LATER
    epochs=50,
    batch_size=8,
    model_save_path='models/colorization_model.keras',
    checkpoint_path=None
):

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)

        try:
            start_epoch = int(checkpoint_path.split('_epoch_')[-1].split('.')[0])
        except (IndexError, ValueError):
            print("Could not infer starting epoch from checkpoint filename. Defaulting to epoch 0.")
            start_epoch = 0
    else:
        print("Training from scratch...")
        # model = build_colorization_model(input_shape=(256, 256, 1))
        model = build_colorization_model(
            input_shape=(256, 256, 1),
            num_filters=(32, 64, 128, 256),
            dropout_rate=0.1,
            learning_rate=1e-4
        )
        start_epoch = 0 #default from start

    model.summary()

    train_gen = data_generator(color_path, gray_path, batch_size=batch_size)
    val_gen = data_generator(color_path, gray_path, batch_size=batch_size)

    # Calculate steps per epoch
    num_samples = len(os.listdir(color_path))
    steps_per_epoch = int(0.8 * num_samples) // batch_size
    validation_steps = int(0.2 * num_samples) // batch_size 


    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='models/colorization_model_epoch_{epoch:02d}.keras',
        save_best_only=False,  # Save every checkpoint
        save_weights_only=False,  # Save full model
        verbose=1
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    callbacks = [checkpoint_callback, early_stopping, reduce_lr]

    # Train model
    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            initial_epoch=start_epoch,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")
        model.save("models/colorization_model_interrupted.keras")
        raise
    # history = model.fit(
    #     train_gen,
    #     steps_per_epoch=steps_per_epoch,
    #     validation_data=val_gen,
    #     validation_steps=validation_steps,
    #     epochs=epochs,
    #     initial_epoch=start_epoch,
    #     callbacks=callbacks
    # )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    return history


if __name__ == "__main__":
    import sys
    resume_checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    train_model(checkpoint_path=resume_checkpoint)