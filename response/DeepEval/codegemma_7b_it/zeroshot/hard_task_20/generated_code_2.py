from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x_1 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x_1 = layers.Conv2D(32, (1, 1), padding="same")(x_1[0])
    x_2 = layers.Conv2D(32, (3, 3), padding="same")(x_1[1])
    x_3 = layers.Conv2D(32, (5, 5), padding="same")(x_1[2])
    x_main = layers.concatenate([x_1, x_2, x_3])

    # Branch path
    x_branch = layers.Conv2D(32, (1, 1), padding="same")(inputs)

    # Fusion
    x_fused = layers.add([x_main, x_branch])

    # Fully connected layers for classification
    x_fused = layers.Flatten()(x_fused)
    outputs = layers.Dense(10, activation="softmax")(x_fused)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model