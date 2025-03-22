import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input image into three groups along the channel dimension
    groups = tf.split(input_layer, 3, axis=3)

    # Convolutional layers
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(groups[0])
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(groups[1])
    conv3 = layers.Conv2D(32, (5, 5), activation='relu')(groups[2])

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated outputs
    flattened = layers.Flatten()(concatenated)

    # Dense layers
    dense1 = layers.Dense(64, activation='relu')(flattened)
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model