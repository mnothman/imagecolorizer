import tensorflow as tf
# import keras
from tensorflow import keras
# from tensorflow.keras import layers, Model, optimizers


def conv_block(x, filters, kernel_size=(3, 3), activation="relu", padding="same", use_batch_norm=True):
    """
    Creates a convolutional block with optional: BatchNorm and ReLU for activation
    """
    x = keras.layers.Conv2D(filters, kernel_size, padding=padding)(x)
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    return x


def encoder_block(x, filters, pool_size=(2, 2), dropout_rate=0.0):
    """
    Encoder block: Conv -> Conv -> Pool -> Dropout (dropout is optional/dependent)
    """
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    skip_connection = x  # Save for skip connection
    x = keras.layers.MaxPooling2D(pool_size=pool_size)(x)
    if dropout_rate > 0.0:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x, skip_connection


def decoder_block(x, skip_connection, filters, upsample_size=(2, 2)):
    """
    Decoder block: Upsample -> Concatenate -> Conv -> Conv
    """
    x = keras.layers.UpSampling2D(size=upsample_size)(x)
    x = keras.layers.Concatenate()([x, skip_connection])
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    return x


def build_colorization_model(input_shape=(256, 256, 1), num_filters=(32, 64, 128, 256), dropout_rate=0.1, learning_rate=1e-4):
    """
    Scalable 'u net' style model for image the colorization
    """
    inputs = keras.layers.Input(shape=input_shape)

    # Encoder
    skips = []
    x = inputs
    for filters in num_filters[:-1]:  # Exclude bottleneck layer
        x, skip = encoder_block(x, filters, dropout_rate=dropout_rate)
        skips.append(skip)

    # Bottleneck
    x = conv_block(x, num_filters[-1])
    x = conv_block(x, num_filters[-1])

    # Decoder
    skips = skips[::-1]  # Reverse order for decoding
    for filters, skip in zip(num_filters[:-1][::-1], skips):
        x = decoder_block(x, skip, filters)

    # Output Layer
    outputs = keras.layers.Conv2D(3, (1, 1), activation="sigmoid", padding="same")(x)

    # Compile model
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


# # Smaller model for testing originally (expanded later)
# import tensorflow as tf
# from tensorflow import keras

# def build_colorization_model(input_shape=(256, 256, 1)):
#     inputs = keras.layers.Input(shape=input_shape)

#     # Encoder
#     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     x = keras.layers.MaxPooling2D((2, 2))(x)
#     x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.MaxPooling2D((2, 2))(x)
#     x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.MaxPooling2D((2, 2))(x)

#     # Bottleneck
#     x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

#     # Decoder
#     x = keras.layers.UpSampling2D((2, 2))(x)
#     x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.UpSampling2D((2, 2))(x)
#     x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = keras.layers.UpSampling2D((2, 2))(x)
#     x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

#     # Output layer
#     outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

#     model = keras.Model(inputs=inputs, outputs=outputs)
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=1e-4),
#         loss='mean_squared_error',
#         metrics=['mae']
#     )

#     return model
