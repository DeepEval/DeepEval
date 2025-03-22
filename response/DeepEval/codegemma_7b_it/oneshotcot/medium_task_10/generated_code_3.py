import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial Convolution
    conv1 = layers.Conv2D(filters=16, kernel_size=3, padding="same", activation="relu")(inputs)

    # Residual Blocks
    def residual_block(input_tensor, filters):
        # Main Path
        conv_main = layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(input_tensor)
        bn_main = layers.BatchNormalization()(conv_main)
        conv_main = layers.Conv2D(filters=filters, kernel_size=3, padding="same", activation="relu")(bn_main)
        bn_main = layers.BatchNormalization()(conv_main)

        # Branch Path
        conv_branch = layers.Conv2D(filters=filters, kernel_size=1, padding="same", activation="relu")(input_tensor)
        bn_branch = layers.BatchNormalization()(conv_branch)

        # Combine Paths
        add = layers.Add()([bn_main, bn_branch])
        act = layers.Activation("relu")(add)

        return act

    # Level 1
    res1 = residual_block(input_tensor=conv1, filters=16)

    # Level 2
    res2 = residual_block(input_tensor=res1, filters=32)
    res2 = residual_block(input_tensor=res2, filters=32)

    # Level 3
    conv3 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(res2)

    # Global Branch
    pool3 = layers.GlobalAveragePooling2D()(conv3)

    # Classification
    flatten = layers.Flatten()(pool3)
    outputs = layers.Dense(units=10, activation="softmax")(flatten)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model