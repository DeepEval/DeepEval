import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = layers.GlobalAveragePooling2D()(branch1)

    # Branch 2: 5x5 convolutions
    branch2 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    branch2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(branch2)
    branch2 = layers.GlobalAveragePooling2D()(branch2)

    # Combine branches using addition
    combined = layers.Add()([branch1, branch2])

    # Attention weights
    attention1 = layers.Dense(1, activation='softmax')(combined)
    attention2 = layers.Dense(1, activation='softmax')(combined)

    # Weighted output
    weighted_output = layers.Multiply()([branch1, attention1]) + layers.Multiply()([branch2, attention2])

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(weighted_output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model