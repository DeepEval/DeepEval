import numpy as np
import tensorflow as tf

def method():
    # Define the input and output shapes
    input_shape = (28, 28, 1)
    output_shape = (10,)

    # Define the layers
    layer_1 = tf.keras.layers.Dense(units=32, activation='relu', input_shape=input_shape)
    layer_2 = tf.keras.layers.Dense(units=10, activation='softmax')

    # Create the model
    model = tf.keras.Sequential([layer_1, layer_2])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Test the model
model = method()
model.summary()