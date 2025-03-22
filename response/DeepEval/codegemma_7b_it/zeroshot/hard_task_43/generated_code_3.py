from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x1 = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(inputs)
    x1 = layers.AveragePooling2D(pool_size=1, strides=1, padding='same')(x1)

    x2 = layers.Conv2D(filters=32, kernel_size=2, strides=1, padding='same')(inputs)
    x2 = layers.AveragePooling2D(pool_size=2, strides=2, padding='same')(x2)

    x3 = layers.Conv2D(filters=32, kernel_size=4, strides=1, padding='same')(inputs)
    x3 = layers.AveragePooling2D(pool_size=4, strides=4, padding='same')(x3)

    # Flatten and concatenate outputs of Block 1
    x = layers.Concatenate()([layers.Flatten()(x1), layers.Flatten()(x2), layers.Flatten()(x3)])

    # Fully connected layer between Block 1 and Block 2
    x = layers.Dense(units=128, activation='relu')(x)

    # Reshape for Block 2
    x = layers.Reshape(target_shape=(4, 4, 32))(x)

    # Block 2
    x1 = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    x1 = layers.AveragePooling2D(pool_size=1, strides=1, padding='same')(x1)

    x2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x2 = layers.AveragePooling2D(pool_size=1, strides=1, padding='same')(x2)

    x3 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same')(x)
    x3 = layers.AveragePooling2D(pool_size=1, strides=1, padding='same')(x3)

    x4 = layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(x)
    x4 = layers.AveragePooling2D(pool_size=1, strides=1, padding='same')(x4)

    # Flatten and concatenate outputs of Block 2
    x = layers.Concatenate()([layers.Flatten()(x1), layers.Flatten()(x2), layers.Flatten()(x3), layers.Flatten()(x4)])

    # Fully connected layer for classification
    outputs = layers.Dense(units=10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model