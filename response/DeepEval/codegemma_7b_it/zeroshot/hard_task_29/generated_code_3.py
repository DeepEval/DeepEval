import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(filters=6, kernel_size=5, activation="relu")(inputs)
    x = layers.Conv2D(filters=6, kernel_size=5, activation="relu")(x)
    branch = keras.Model(inputs, x)

    # Branch path
    y = keras.Input(shape=(28, 28, 1))
    branch_path = layers.Conv2D(filters=6, kernel_size=5, activation="relu")(y)
    branch_path = layers.Conv2D(filters=6, kernel_size=5, activation="relu")(branch_path)

    # Main path
    main_path = layers.Conv2D(filters=6, kernel_size=5, activation="relu")(inputs)
    main_path = layers.Conv2D(filters=6, kernel_size=5, activation="relu")(main_path)

    # Combine outputs from both paths
    output = layers.add([branch_path, main_path])

    # Block 2
    max_pool_1 = layers.MaxPooling2D(pool_size=1, strides=1)(output)
    max_pool_2 = layers.MaxPooling2D(pool_size=2, strides=2)(output)
    max_pool_3 = layers.MaxPooling2D(pool_size=4, strides=4)(output)

    # Flatten and concatenate pooling outputs
    max_pool_1 = layers.Flatten()(max_pool_1)
    max_pool_2 = layers.Flatten()(max_pool_2)
    max_pool_3 = layers.Flatten()(max_pool_3)
    concat_outputs = layers.concatenate([max_pool_1, max_pool_2, max_pool_3])

    # Fully connected layers
    x = layers.Dense(128, activation="relu")(concat_outputs)
    outputs = layers.Dense(10, activation="softmax")(x)

    # Create the model
    model = keras.Model(inputs=[inputs, y], outputs=outputs)

    return model

# Create the model
model = dl_model()