import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Create a separable convolutional layer for each group with different kernel sizes
    conv1x1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv3x3 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_layer[1])
    conv5x5 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_layer[2])

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([conv1x1, conv3x3, conv5x5])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # Add three fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model