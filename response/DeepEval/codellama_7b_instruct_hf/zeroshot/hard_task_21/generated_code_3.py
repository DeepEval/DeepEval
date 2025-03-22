import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the output shape
    output_shape = (10,)

    # Define the main path
    main_path = keras.Sequential([
        layers.Lambda(lambda x: tf.split(x, 3, axis=3), name="split"),
        layers.Conv2D(32, (1, 1), activation="relu", name="conv1"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
        layers.Conv2D(128, (5, 5), activation="relu", name="conv3"),
        layers.Flatten(),
        layers.Dense(128, activation="relu", name="fc1"),
        layers.Dense(output_shape, activation="softmax", name="fc2")
    ])

    # Define the branch path
    branch_path = keras.Sequential([
        layers.Lambda(lambda x: tf.split(x, 3, axis=3), name="split"),
        layers.Conv2D(64, (1, 1), activation="relu", name="conv1"),
        layers.Flatten(),
        layers.Dense(128, activation="relu", name="fc1"),
        layers.Dense(output_shape, activation="softmax", name="fc2")
    ])

    # Define the model
    model = keras.Model(inputs=main_path, outputs=main_path + branch_path)

    return model