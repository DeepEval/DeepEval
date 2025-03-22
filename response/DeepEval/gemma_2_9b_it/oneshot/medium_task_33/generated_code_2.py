import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    split_channels = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Create three branches for feature extraction
    branch1 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_channels[0])
    branch2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split_channels[1])
    branch3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(split_channels[2])

    # Concatenate the outputs from the three branches
    concatenated = layers.Concatenate(axis=3)([branch1, branch2, branch3])

    # Flatten the output and pass through three fully connected layers
    flatten = layers.Flatten()(concatenated)
    dense1 = layers.Dense(units=128, activation='relu')(flatten)
    dense2 = layers.Dense(units=64, activation='relu')(dense1)
    output_layer = layers.Dense(units=10, activation='softmax')(dense2)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model