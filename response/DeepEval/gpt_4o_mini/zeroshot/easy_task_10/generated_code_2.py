import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with a single color channel

    # 1x1 Convolutional Layer to increase dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)

    # 3x3 Depthwise Separable Convolutional Layer for feature extraction
    x = layers.SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(x)

    # 1x1 Convolutional Layer to reduce dimensionality
    x = layers.Conv2D(filters=10, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(x)

    # Flattening layer
    x = layers.Flatten()(x)

    # Fully connected layer
    outputs = layers.Dense(units=10, activation='softmax')(x)  # 10 classes for MNIST digits 0-9

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # Print the model summary to verify the architecture