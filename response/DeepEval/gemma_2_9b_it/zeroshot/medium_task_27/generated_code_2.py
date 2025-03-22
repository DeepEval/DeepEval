import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)

    # Branch 2: 5x5 convolutions
    branch2 = layers.Conv2D(32, (5, 5), activation='relu')(inputs)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.Conv2D(64, (5, 5), activation='relu')(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)

    # Combine branches
    combined = layers.Add()([branch1, branch2])

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(combined)

    # Attention weights
    attention1 = layers.Dense(1, activation='softmax')(x)
    attention2 = layers.Dense(1, activation='softmax')(x)

    # Weighted sum
    weighted_output = attention1 * branch1 + attention2 * branch2

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(weighted_output)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model