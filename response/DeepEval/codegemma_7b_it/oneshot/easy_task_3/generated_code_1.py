import keras
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Concatenate,
    BatchNormalization,
    Flatten,
    Dense,
)

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First feature extraction block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(conv2)

    # Second feature extraction block
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(max_pool1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(conv3)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="valid", activation="relu")(conv4)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")(conv5)

    # Flatten and fully connected layers
    flattened = Flatten()(max_pool2)
    dense1 = Dense(units=128, activation="relu")(flattened)
    dense2 = Dense(units=64, activation="relu")(dense1)
    output_layer = Dense(units=10, activation="softmax")(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model