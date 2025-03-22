# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape and number of classes
    input_shape = (28, 28, 1)
    num_classes = 10

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Apply an initial convolutional layer to adjust the dimensions of the input data
    x = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Create Block 1
    block_1_output = layers.Lambda(lambda x: tf.split(x, 2, axis=-1))(x)
    x1, x2 = block_1_output

    # Apply operations to the first group
    x1 = layers.Conv2D(32, (1, 1), activation='relu')(x1)
    x1 = layers.SeparableConv2D(32, (3, 3), activation='relu')(x1)
    x1 = layers.Conv2D(32, (1, 1), activation='relu')(x1)

    # Pass the second group through without modification
    x2 = x2

    # Merge the outputs from both groups
    x = layers.Concatenate()([x1, x2])

    # Create Block 2
    block_2_output = x
    input_shape = block_2_output.shape[1:]
    num_groups = 4
    channels_per_group = input_shape[-1] // num_groups

    # Reshape the input into four groups
    x = layers.Lambda(lambda x: tf.reshape(x, [-1] + list(input_shape[:2]) + [num_groups, channels_per_group]))(block_2_output)

    # Swap the third and fourth dimensions
    x = layers.Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2]))(x)

    # Reshape the input back to its original shape to achieve channel shuffling
    x = layers.Reshape(input_shape)(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Apply a fully connected layer for classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model