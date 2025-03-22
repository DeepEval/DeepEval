import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    main_path = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)
    main_path = [
        keras.Sequential([
            layers.SeparableConv2D(filters=32, kernel_size=1, padding="same"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
        ])(x)
        for x in main_path
    ]

    # Branch Path
    branch_path = keras.Sequential([
        layers.SeparableConv2D(filters=32, kernel_size=1, padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
    ])(inputs)

    # Fusion
    outputs = keras.layers.Add()([branch_path] + main_path)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Activation("relu")(outputs)

    # Classification
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = keras.layers.Dense(units=10, activation="softmax")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model