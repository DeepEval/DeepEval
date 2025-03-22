# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.datasets import mnist

def dl_model():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape the input data to be 4D arrays (batch, height, width, channels)
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

    # Define the model architecture
    inputs = layers.Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # 3x3 depthwise separable convolutional layer for feature extraction
    x = layers DepthwiseConv2D((3, 3), activation='relu')(x)

    # 1x1 convolutional layer to reduce dimensionality
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)

    # Add the original input layer
    x = layers.Add()([x, inputs])

    # Flattening layer
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    x = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Usage
model = dl_model()
print(model.summary())