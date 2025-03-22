import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Main path
    main_path = keras.Sequential([
        # Splitting the input into three groups along the channel dimension
        layers.Lambda(lambda x: tf.split(x, 3, axis=3)),
        # Multi-scale feature extraction with separable convolutional layers of varying kernel sizes
        layers.Conv2D(32, (1, 1), use_bias=False, activation='relu'),
        layers.SeparableConv2D(32, (3, 3), use_bias=False, activation='relu'),
        layers.SeparableConv2D(32, (5, 5), use_bias=False, activation='relu'),
        # Concatenating the outputs from the three groups
        layers.Concatenate()
    ])

    # Branch path
    branch_path = keras.Sequential([
        # Applying a 1x1 convolutional layer to align the number of output channels with the main path
        layers.Conv2D(32, (1, 1), use_bias=False, activation='relu'),
        # Multi-scale feature extraction with separable convolutional layers of varying kernel sizes
        layers.SeparableConv2D(32, (1, 1), use_bias=False, activation='relu'),
        layers.SeparableConv2D(32, (3, 3), use_bias=False, activation='relu'),
        layers.SeparableConv2D(32, (5, 5), use_bias=False, activation='relu'),
        # Concatenating the outputs from the branch path
        layers.Concatenate()
    ])

    # Fusing the outputs from both paths through addition
    fused_output = layers.Add()([main_path, branch_path])

    # Flattening the output into a one-dimensional vector
    flattened_output = layers.Flatten()(fused_output)

    # Passing the flattened output through two fully connected layers for a 10-class classification task
    dense_output = layers.Dense(128, activation='relu')(flattened_output)
    dense_output = layers.Dense(10, activation='softmax')(dense_output)

    # Creating the model
    model = keras.Model(inputs=main_path.input, outputs=dense_output)

    return model