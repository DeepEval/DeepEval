import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))  

    # Split the input into three groups along the channel dimension
    split_outputs = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(inputs)

    # Process each group
    group1 = layers.Conv2D(32, (1, 1), activation="relu")(split_outputs[0])
    group1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(group1)
    group1 = layers.Conv2D(32, (1, 1), activation="relu")(group1)

    group2 = layers.Conv2D(32, (1, 1), activation="relu")(split_outputs[1])
    group2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(group2)
    group2 = layers.Conv2D(32, (1, 1), activation="relu")(group2)

    group3 = layers.Conv2D(32, (1, 1), activation="relu")(split_outputs[2])
    group3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(group3)
    group3 = layers.Conv2D(32, (1, 1), activation="relu")(group3)

    # Combine the outputs
    combined = layers.Add()([group1, group2, group3])

    # Fuse with original input
    output = layers.Add()([inputs, combined])

    # Flatten and classify
    output = layers.Flatten()(output)
    output = layers.Dense(10, activation="softmax")(output)

    model = keras.Model(inputs=inputs, outputs=output)
    return model