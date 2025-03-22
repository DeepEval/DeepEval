import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Main path
    input_img = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation="relu")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch path
    y = layers.Conv2D(64, (5, 5), activation="relu")(input_img)

    # Combine features
    combined = layers.concatenate([x, y])

    # Fully connected layers
    combined = layers.Flatten()(combined)
    combined = layers.Dense(64, activation="relu")(combined)
    output = layers.Dense(10, activation="softmax")(combined)

    # Create the model
    model = keras.Model(inputs=input_img, outputs=output)

    return model