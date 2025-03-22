import numpy as np
from tensorflow.keras import layers, models, datasets, utils

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)

    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Path 2: sequence of convolutions 1x1 -> 1x7 -> 7x1
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(path2)
    path2 = layers.Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(path2)

    # Concatenate both paths
    combined = layers.concatenate([path1, path2])

    # Final 1x1 convolution to align output dimensions
    main_output = layers.Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(combined)

    # Direct branch connecting to the input
    branch = inputs

    # Merging outputs of the main path and the branch through addition
    merged = layers.add([main_output, branch])

    # Flatten the merged output
    flattened = layers.Flatten()(merged)

    # Fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    
    # Output layer with softmax activation for classification
    outputs = layers.Dense(10, activation='softmax')(dense2)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()