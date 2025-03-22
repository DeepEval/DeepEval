import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the output shape
    output_shape = (10,)

    # Define the specialized block
    specialized_block = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.AveragePooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(output_shape[1], activation='softmax')
    ])

    # Define the model
    model = tf.keras.models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Reshape(target_shape=(28, 28, 1)),
        specialized_block,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(output_shape[1], activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model