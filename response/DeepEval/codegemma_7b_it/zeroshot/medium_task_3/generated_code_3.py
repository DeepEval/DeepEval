import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer for the image data
    input_img = layers.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    # Path A
    path_a = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    path_a = layers.MaxPooling2D()(path_a)
    path_a = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(path_a)
    path_a = layers.MaxPooling2D()(path_a)
    path_a = layers.Flatten()(path_a)

    # Path B
    path_b = layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    path_b = layers.MaxPooling2D()(path_b)
    path_b = layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same')(path_b)
    path_b = layers.MaxPooling2D()(path_b)
    path_b = layers.Flatten()(path_b)

    # Path C
    path_c = layers.Conv2D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
    path_c = layers.MaxPooling2D()(path_c)
    path_c = layers.Conv2D(filters=128, kernel_size=7, activation='relu', padding='same')(path_c)
    path_c = layers.MaxPooling2D()(path_c)
    path_c = layers.Flatten()(path_c)

    # Combine Path A, Path B, and Path C
    combined = layers.concatenate([path_a, path_b, path_c])

    # Fully connected layer for classification
    output = layers.Dense(units=10, activation='softmax')(combined)

    # Create the model
    model = keras.Model(inputs=input_img, outputs=output)

    return model