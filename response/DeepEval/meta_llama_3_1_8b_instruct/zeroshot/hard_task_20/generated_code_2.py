# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    This function creates a deep learning model for image classification using the CIFAR-10 dataset.
    The model consists of a main path and a branch path. The main path splits the input into three groups 
    by encapsulating tf.split within Lambda layer, each undergoing feature extraction with convolutional 
    layers of different kernel sizes (1x1, 3x3, and 5x5). The outputs from these three groups are then concatenated.
    The branch path processes the input with a 1x1 convolutional layer to align the number of output channels 
    with those of the main path. The outputs of the main and branch paths are combined through addition to create fused features.
    Finally, the model performs classification using two fully connected layers.
    
    Returns:
        model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model's input
    inputs = keras.Input(shape=input_shape)

    # Main Path
    # Split the input into three groups by encapsulating tf.split within Lambda layer
    split_input = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Feature extraction with convolutional layers of different kernel sizes
    # (1x1, 3x3, and 5x5)
    conv1x1 = layers.Conv2D(32, 1, activation='relu')(split_input[0])
    conv3x3 = layers.Conv2D(32, 3, activation='relu')(split_input[1])
    conv5x5 = layers.Conv2D(32, 5, activation='relu')(split_input[2])

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([conv1x1, conv3x3, conv5x5])

    # Branch Path
    # Process the input with a 1x1 convolutional layer to align the number of output channels
    # with those of the main path
    branch = layers.Conv2D(32, 1, activation='relu')(inputs)

    # Combine the outputs of the main and branch paths through addition
    fused_features = layers.Add()([concatenated, branch])

    # Flatten the concatenated output
    flattened = layers.Flatten()(fused_features)

    # Perform classification using two fully connected layers
    fc1 = layers.Dense(128, activation='relu')(flattened)
    outputs = layers.Dense(10, activation='softmax')(fc1)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model