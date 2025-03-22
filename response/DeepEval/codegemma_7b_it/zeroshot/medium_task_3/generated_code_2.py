import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer for the image
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x1 = layers.MaxPooling2D(2, 2)(x1)

    # Block 2
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x2 = layers.MaxPooling2D(2, 2)(x2)

    # Combine the outputs from both paths with the input
    x = layers.Add()([x1, x2, inputs])

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()