# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    Creates a deep learning model for image classification using the MNIST dataset.
    
    The model consists of two sequential blocks for feature extraction:
    1. The first block contains three convolutional layers followed by max pooling.
    2. The second block contains four convolutional layers followed by max pooling.
    The feature maps are then flattened and fed into three fully connected layers,
    culminating in the classification output.
    
    Returns:
        A compiled Keras model.
    """

    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Create a sequential model
    model = keras.Sequential()

    # First sequential block for feature extraction
    # Convolutional layers with max pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Second sequential block for feature extraction
    # Convolutional layers with max pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the feature maps
    model.add(layers.Flatten())

    # Fully connected layers for classification
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model