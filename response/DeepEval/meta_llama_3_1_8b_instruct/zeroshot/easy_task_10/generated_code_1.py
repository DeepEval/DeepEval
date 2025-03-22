# Import necessary Keras and other packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    Creates a deep learning model for image classification using the MNIST dataset.
    
    The model consists of the following layers:
    1. 1x1 convolutional layer to increase dimensionality
    2. 3x3 depthwise separable convolutional layer for feature extraction
    3. 1x1 convolutional layer to reduce dimensionality with a stride of 2
    4. A flattening layer
    5. A fully connected layer for classification

    Returns:
    A compiled Keras model.
    """

    # Define the input shape of the model (28x28 grayscale images for MNIST dataset)
    input_shape = (28, 28, 1)

    # Create the model using Functional API
    inputs = keras.Input(shape=input_shape)

    # Apply a 1x1 convolutional layer to increase dimensionality
    x = layers.Conv2D(32, (1, 1), strides=(2, 2), activation='relu')(inputs)

    # Apply a 3x3 depthwise separable convolutional layer for feature extraction
    x = layers.DepthwiseConv2D((3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.SeparableConv2D(32, (3, 3), strides=(1, 1), activation='relu')(x)

    # Apply another 1x1 convolutional layer to reduce dimensionality
    x = layers.Conv2D(32, (1, 1), strides=(2, 2), activation='relu')(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Apply a fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()