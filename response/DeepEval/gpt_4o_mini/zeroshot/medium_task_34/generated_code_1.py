import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Part 1: Feature extraction
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolutional block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Third convolutional block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Part 2: Enhancing generalization
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Part 3: Upsampling with skip connections
    # Store the output of the conv layers for skip connections
    skip_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    skip_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(layers.MaxPooling2D((2, 2))(skip_1))
    skip_3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(layers.MaxPooling2D((2, 2))(skip_2))

    # First upsampling block
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, skip_3])  # Skip connection
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)

    # Second upsampling block
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, skip_2])  # Skip connection
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)

    # Third upsampling block
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, skip_1])  # Skip connection

    # Output layer with 1x1 convolution for class probabilities
    output_layer = layers.Conv2D(10, (1, 1), activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()