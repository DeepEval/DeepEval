import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_shape = (32, 32, 3)

    inputs = layers.Input(shape=input_shape)

    # Split the input image into three channel groups
    channels = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Feature extraction using separable convolutions
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(channels[0])
    group1 = layers.Conv2D(64, (3, 3), activation='relu')(group1)
    group2 = layers.Conv2D(32, (1, 1), activation='relu')(channels[1])
    group2 = layers.Conv2D(64, (3, 3), activation='relu')(group2)
    group3 = layers.Conv2D(32, (1, 1), activation='relu')(channels[2])
    group3 = layers.Conv2D(64, (5, 5), activation='relu')(group3)

    # Concatenate the outputs from the three groups
    merged = layers.Concatenate(axis=-1)([group1, group2, group3])

    # Fully connected layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(128, activation='relu')(merged)
    outputs = layers.Dense(10, activation='softmax')(merged)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model