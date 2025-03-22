import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the image
    input_img = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 channel

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.25)(x)

    # Restoring number of channels
    x = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    # Branch path directly from the input
    branch = input_img

    # Combining the main path and branch path
    combined = layers.add([x, branch])

    # Flatten and fully connected layer for classification
    flat = layers.Flatten()(combined)
    output = layers.Dense(10, activation='softmax')(flat)  # 10 classes for digits 0-9

    # Construct the model
    model = models.Model(inputs=input_img, outputs=output)

    return model

# Example usage
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()