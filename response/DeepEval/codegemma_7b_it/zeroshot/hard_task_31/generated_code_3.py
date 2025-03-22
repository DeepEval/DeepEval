import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # First block
    conv = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(inputs)
    conv = layers.Dropout(rate=0.2)(conv)
    conv = layers.Conv2D(filters=32, kernel_size=1, padding="same", activation="relu")(conv)
    shortcut = conv

    # Branch path
    branch_conv = layers.Conv2D(filters=32, kernel_size=1, padding="same", activation="relu")(inputs)

    # Main path
    conv = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conv)
    conv = layers.Dropout(rate=0.2)(conv)
    conv = layers.Conv2D(filters=32, kernel_size=1, padding="same", activation="relu")(conv)

    # Concatenate outputs
    output = layers.add([shortcut, conv, branch_conv])

    # Second block
    group_outputs = []
    for kernel_size in [1, 3, 5]:
        conv = layers.Conv2D(filters=64, kernel_size=kernel_size, padding="same", activation="relu")(output)
        conv = layers.Dropout(rate=0.2)(conv)
        group_outputs.append(conv)

    # Concatenate outputs from all groups
    output = layers.concatenate(group_outputs)

    # Output layer
    flatten = layers.Flatten()(output)
    dense = layers.Dense(units=10, activation="softmax")(flatten)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=dense)

    return model

# Example usage:
model = dl_model()
model.summary()