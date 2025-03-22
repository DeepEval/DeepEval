# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Create the main path of the model
    main_path = keras.Sequential([
        # Convolutional layer 1 with dropout
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        # Convolutional layer 2 with dropout
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.2),
        # Convolutional layer 3 to restore the number of channels
        layers.Conv2D(32, (3, 3), activation='relu')
    ])

    # Create the branch path of the model
    branch_path = keras.Sequential([
        # Convolutional layer 1 to connect directly to the input
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
    ])

    # Combine the outputs from both paths through an addition operation
    combined_path = keras.Sequential([
        layers.Add()([main_path.output, branch_path.output])
    ])

    # Define the final layers of the model
    model = keras.Sequential([
        combined_path,
        # Flatten the output
        layers.Flatten(),
        # Fully connected layer for classification
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model