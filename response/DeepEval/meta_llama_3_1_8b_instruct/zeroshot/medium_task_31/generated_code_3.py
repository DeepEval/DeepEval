# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define the dl_model function
def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape, name='input_layer')

    # Split the input into three groups along the channel dimension
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Apply different convolutional kernels to each group
    x1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x[0])
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x[1])
    x3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x[2])

    # Concatenate the outputs from the three groups
    x = layers.Concatenate()([x1, x2, x3])

    # Apply convolutional and pooling layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Apply fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Usage
model = dl_model()
model.summary()