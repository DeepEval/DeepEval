# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Define the first sequential block
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Define the main path
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(x)
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.AveragePooling2D((2, 2))(main_path)
    main_path = layers.Flatten()(main_path)
    main_path = layers.Dense(64, activation='relu')(main_path)

    # Add the input to the main path via addition
    x = layers.Add()([inputs, main_path])

    # Define the second sequential block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    weights = layers.Dense(64, activation='linear')(x)
    weights = layers.Reshape((1, 64))(weights)
    x = layers.Multiply()([inputs, weights])

    # Define the output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
model = dl_model()
model.summary()