import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Lambda, SeparableConv2D, Reshape, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block of the model
    first_block = Sequential([
        MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding="same"),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same"),
        Flatten(),
        Dropout(0.2),
        Concatenate([
            SeparableConv2D(128, (1, 1), activation="relu"),
            SeparableConv2D(128, (3, 3), activation="relu"),
            SeparableConv2D(128, (5, 5), activation="relu"),
            SeparableConv2D(128, (7, 7), activation="relu")
        ])
    ])

    # Define the second block of the model
    second_block = Sequential([
        Reshape(target_shape=(8, 8, 128)),
        Concatenate([
            Lambda(lambda x: tf.split(x, 4, axis=-1))
        ]),
        SeparableConv2D(256, (1, 1), activation="relu"),
        SeparableConv2D(256, (3, 3), activation="relu"),
        SeparableConv2D(256, (5, 5), activation="relu"),
        SeparableConv2D(256, (7, 7), activation="relu")
    ])

    # Define the model
    model = Model(inputs=first_block.input, outputs=second_block(first_block.output))

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model