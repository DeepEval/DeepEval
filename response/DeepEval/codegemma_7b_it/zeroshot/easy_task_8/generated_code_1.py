import keras
from keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    x = layers.SeparableConv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # 1x1 convolutional layer for feature extraction
    x = layers.Conv2D(16, (1, 1), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    # Create and return the model
    model = keras.Model(inputs, outputs)
    return model