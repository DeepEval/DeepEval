import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create a model
    model = keras.Sequential()

    # Compress the input features with global average pooling
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.GlobalAveragePooling2D())

    # Fully connected layers to learn correlations among feature map channels
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))

    # Reshape weights and multiply with input feature map
    model.add(layers.Reshape((32, 32, 32)))
    model.add(layers.Multiply())

    # Flatten and produce output probability distribution
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="softmax"))

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model