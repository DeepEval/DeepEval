import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (1, 1), strides=(2, 2), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    branch1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same")(x)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.Activation("relu")(branch1)

    branch2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
    branch2 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same")(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Activation("relu")(branch2)

    branch3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)
    branch3 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same")(branch3)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.Activation("relu")(branch3)

    # Branch path
    branch_input = keras.Input(shape=(32, 32, 3))
    branch_x = layers.Conv2D(32, (1, 1), strides=(2, 2), padding="same")(branch_input)
    branch_x = layers.BatchNormalization()(branch_x)
    branch_x = layers.Activation("relu")(branch_x)

    branch_x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same")(branch_x)
    branch_x = layers.BatchNormalization()(branch_x)
    branch_x = layers.Activation("relu")(branch_x)

    # Concatenate branches
    x = layers.Concatenate()([x, branch_x])

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    # Create model
    model = keras.Model(inputs=[inputs, branch_input], outputs=outputs)

    # Compile model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model