import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def basic_block(input_tensor, filters, kernel_size=(3, 3), strides=1):
    """A basic convolutional block."""

    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def down_sample(input_tensor, filters, kernel_size=(3, 3), strides=2):
    """Downsamples the input by applying a 2x2 convolution with stride 2."""

    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def feature_fusion(main_path_output, branch_path_output):
    """Combines the outputs from the main and branch paths."""

    return main_path_output + branch_path_output

def make_basic_block_network(input_shape):
    """Constructs a network of basic blocks."""

    input_tensor = layers.Input(shape=input_shape)

    # Initial convolutional layer to reduce dimensionality.
    x = layers.Conv2D(16, (3, 3), strides=1, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Two consecutive basic blocks.
    for _ in range(2):
        x = basic_block(x, 16)

    # Branch for feature extraction.
    branch_path = layers.Conv2D(16, (3, 3), strides=1, padding='same')(input_tensor)
    branch_path = layers.BatchNormalization()(branch_path)
    branch_path = layers.ReLU()(branch_path)
    branch_path = basic_block(branch_path, 16)

    # Feature fusion.
    main_path_output = x
    branch_path_output = feature_fusion(main_path_output, branch_path)

    # Downsample the feature map.
    branch_path_output = down_sample(branch_path_output, 32)

    # Additional basic block after downsampling.
    branch_path_output = basic_block(branch_path_output, 32)

    # Average pooling to reduce dimensionality.
    branch_path_output = layers.AveragePooling2D((8, 8), strides=1)(branch_path_output)

    # Flatten the feature map and process through fully connected layers.
    branch_path_output = layers.Flatten()(branch_path_output)
    branch_path_output = layers.Dense(10, activation='softmax')(branch_path_output)

    # Define the model.
    model = keras.Model(inputs=input_tensor, outputs=branch_path_output)

    return model

# Example usage:
model = make_basic_block_network(input_shape=(32, 32, 3))