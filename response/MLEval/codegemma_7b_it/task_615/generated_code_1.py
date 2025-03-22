import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def method():
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape and normalize the data
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    # Define the model architecture
    model = keras.Sequential(
        [
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),  # Add dropout layer
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),  # Add dropout layer
            layers.Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    model.fit(x_train, y_train, epochs=20, batch_size=64)

    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)

    # Print the accuracy
    print("Accuracy:", accuracy)

    return output  # Return the output if needed

# Call the method for validation
method()