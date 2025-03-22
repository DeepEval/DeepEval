# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model using Keras' Functional API.
    The model is designed for image classification using the MNIST dataset.
    It consists of:
    1. A 1x1 convolutional layer to reduce dimensionality
    2. A 3x3 convolutional layer to extract features
    3. Another 1x1 convolutional layer to restore dimensionality
    4. A flattening layer to prepare for the fully connected layer
    5. A fully connected layer with 10 neurons for classification
    """

    # Create the input layer with a 1x28x28 shape (assuming 1 color channel)
    input_layer = keras.Input(shape=(28, 28, 1))

    # Apply a 1x1 convolutional layer to reduce dimensionality
    reduced_dim = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Apply a 3x3 convolutional layer to extract features
    features = layers.Conv2D(64, (3, 3), activation='relu')(reduced_dim)

    # Apply another 1x1 convolutional layer to restore dimensionality
    restored_dim = layers.Conv2D(32, (1, 1), activation='relu')(features)

    # Flatten the output to prepare for the fully connected layer
    flattened = layers.Flatten()(restored_dim)

    # Apply a fully connected layer with 10 neurons for classification
    output_layer = layers.Dense(10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model