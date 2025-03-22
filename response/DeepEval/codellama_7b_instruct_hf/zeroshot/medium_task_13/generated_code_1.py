import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    model = keras.Sequential([
        # Convolutional layer 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        # Convolutional layer 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        # Convolutional layer 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        # Max pooling layer
        layers.MaxPooling2D((2, 2)),
        # Flatten layer
        layers.Flatten(),
        # Dense layer 1
        layers.Dense(128, activation="relu"),
        # Dense layer 2 (output layer)
        layers.Dense(10, activation="softmax")
    ])

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model