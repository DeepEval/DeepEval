import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch1 = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    branch1 = layers.Conv2D(32, (3, 3), activation="relu")(branch1)

    # Branch 2: 1x1 followed by 3x3 convolutions
    branch2 = layers.Conv2D(32, (1, 1), activation="relu")(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation="relu")(branch2)
    branch2 = layers.Conv2D(32, (3, 3), activation="relu")(branch2)

    # Branch 3: Max pooling
    branch3 = layers.MaxPooling2D((2, 2))(inputs)
    branch3 = layers.Conv2D(32, (3, 3), activation="relu")(branch3)

    # Concatenate branches
    concat = layers.concatenate([branch1, branch2, branch3])

    # Multi-scale feature fusion block
    fusion = layers.Conv2D(64, (1, 1), activation="relu")(concat)

    # Flatten and fully connected layers
    flattened = layers.Flatten()(fusion)
    dense1 = layers.Dense(64, activation="relu")(flattened)
    outputs = layers.Dense(10, activation="softmax")(dense1)

    model = tf.keras.Model(inputs, outputs)

    return model

model = dl_model()