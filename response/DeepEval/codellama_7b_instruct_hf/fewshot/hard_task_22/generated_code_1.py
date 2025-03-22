import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = keras.Sequential([
        # Split the input into three groups along the channel
        layers.Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1)),
        # Multi-scale feature extraction with separable convolutional layers
        layers.SeparableConv2D(32, (1, 1), activation='relu'),
        layers.SeparableConv2D(32, (3, 3), activation='relu'),
        layers.SeparableConv2D(32, (5, 5), activation='relu'),
        # Concatenate the outputs from the three groups
        layers.Concatenate()
    ])

    # Define the branch path
    branch_path = keras.Sequential([
        # Apply a 1x1 convolutional layer to align the number of output channels
        layers.Conv2D(32, (1, 1), activation='relu'),
        # Multi-scale feature extraction with separable convolutional layers
        layers.SeparableConv2D(32, (1, 1), activation='relu'),
        layers.SeparableConv2D(32, (3, 3), activation='relu'),
        layers.SeparableConv2D(32, (5, 5), activation='relu'),
        # Concatenate the outputs from the three groups
        layers.Concatenate()
    ])

    # Define the fused output
    fused_output = main_path + branch_path

    # Flatten the fused output and pass through two fully connected layers for classification
    flattened_output = layers.Flatten()(fused_output)
    output = layers.Dense(10, activation='softmax')(flattened_output)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=output)

    return model