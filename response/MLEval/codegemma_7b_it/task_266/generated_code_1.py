# Import necessary packages
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1) / 255.0
x_test = np.expand_dims(x_test, -1) / 255.0

# Train the model with different batch sizes and epochs
def method():
    # for batch_size in [32, 64, 128]:
    #     for epochs in [10, 20, 30]:

    # 修改为固定的batch_size和epochs
    model.fit(x_train, y_train, batch_size=32, epochs=5)

# Call the method to validate the model
method()