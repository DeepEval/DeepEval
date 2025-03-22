import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    # Input Layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups
    split_layer = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Group 1: 1x1 Convolutions
    group1 = layers.Conv2D(64, (1, 1), activation='relu')(split_layer[0])
    # Group 2: 3x3 Convolutions
    group2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(split_layer[1])
    # Group 3: 5x5 Convolutions
    group3 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(split_layer[2])

    # Concatenate the outputs of the three groups
    concatenated_features = layers.concatenate([group1, group2, group3], axis=-1)

    # Flatten the features
    flattened_features = layers.Flatten()(concatenated_features)

    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flattened_features)

    # Build the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model