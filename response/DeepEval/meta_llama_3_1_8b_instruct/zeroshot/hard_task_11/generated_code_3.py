# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the main pathway
    pathway = keras.Sequential()
    pathway.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=input_shape))
    pathway.add(layers.Conv2D(32, (1, 1), activation='relu'))
    pathway.add(layers.Conv2D(64, (1, 1), activation='relu'))
    pathway.add(layers.Conv2D(64, (3, 3), activation='relu'))
    pathway.add(layers.Conv2D(64, (1, 1), activation='relu'))
    pathway.add(layers.Conv2D(64, (1, 3), activation='relu'))
    pathway.add(layers.Conv2D(64, (3, 1), activation='relu'))

    # Create the parallel branch
    parallel_branch = keras.Sequential()
    parallel_branch.add(layers.Conv2D(64, (1, 1), activation='relu', input_shape=input_shape))
    parallel_branch.add(layers.Conv2D(64, (3, 3), activation='relu'))
    parallel_branch.add(layers.Conv2D(64, (1, 1), activation='relu'))

    # Concatenate the outputs from the two pathways
    concatenated = layers.Concatenate()([pathway.output, parallel_branch.output])

    # Apply a 1x1 convolution to the concatenated output
    output = layers.Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Add a direct connection from the input to the branch
    direct_connection = layers.Add()([input_shape, output])

    # Create the output layer with the same dimensions as the input
    output_layer = layers.Conv2D(3, (1, 1), activation='relu')(direct_connection)

    # Create the fully connected layers for classification
    fc1 = layers.Flatten()(output_layer)
    fc1 = layers.Dense(64, activation='relu')(fc1)
    output = layers.Dense(10, activation='softmax')(fc1)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=output)

    return model