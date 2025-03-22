import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Assuming you have already loaded and prepared the data
train_set = ...
train_labels = ...

def method():

    # Create a Sequential model
    model = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1)
    ])

    # Compile the model
    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["mae"],
    )

    # Train the model
    history = model.fit(
        train_set,
        train_labels,
        epochs=10,
        batch_size=32,
    )

    # Print the training history
    print(history.history)

    # Return the trained model
    return model

# Call the method to generate the trained model
model = method()