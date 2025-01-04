import tensorflow as tf
from tensorflow import keras

def build_colorization_model(input_shape=(256, 256, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    outputs = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# # larger model (use after testing)
# def build_colorization_model(input_shape=(256, 256, 1)):
#     """
#     Builds a simple autoencoder-like model that maps grayscale to RGB.
#     """
#     inputs = keras.layers.Input(shape=input_shape)

#     # --- Encoder ---
#     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     x = keras.layers.MaxPooling2D((2, 2))(x)
#     x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.MaxPooling2D((2, 2))(x)
#     x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.MaxPooling2D((2, 2))(x)

#     # --- Bottleneck ---
#     x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

#     # --- Decoder ---
#     x = keras.layers.UpSampling2D((2, 2))(x)
#     x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.UpSampling2D((2, 2))(x)
#     x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.UpSampling2D((2, 2))(x)
#     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

#     # Output: 3 channels (R, G, B)
#     outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

#     model = keras.Model(inputs=inputs, outputs=outputs)
#     # model = models.Model(inputs, outputs)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss='mean_squared_error',
#         metrics=['mae']
#     )

#     return model
