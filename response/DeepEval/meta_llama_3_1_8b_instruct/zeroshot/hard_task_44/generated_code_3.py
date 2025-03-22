# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define the function to construct the deep learning model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Construct the input layer
    inputs = keras.Input(shape=input_shape)

    # Block 1: feature extraction with convolutional and dropout
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x1, x2, x3 = x

    # Convolutional layers for each group
    x1 = layers.Conv2D(32, (1, 1), activation='relu')(x1)
    x2 = layers.Conv2D(32, (3, 3), activation='relu')(x2)
    x3 = layers.Conv2D(32, (5, 5), activation='relu')(x3)

    # Concatenate the outputs from the three groups
    x = layers.Concatenate()([x1, x2, x3])
    
    # Dropout layer to reduce overfitting
    x = layers.Dropout(0.2)(x)

    # Block 2: feature fusion with multiple branches
    # Branch 1: 1x1 convolution
    x1 = layers.Conv2D(64, (1, 1), activation='relu')(x)

    # Branch 2: <1x1 convolution, 3x3 convolution>
    x2 = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)

    # Branch 3: <1x1 convolution, 5x5 convolution>
    x3 = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x3 = layers.Conv2D(64, (5, 5), activation='relu')(x3)

    # Branch 4: <3x3 max pooling, 1x1 convolution>
    x4 = layers.MaxPooling2D((3, 3))(x)
    x4 = layers.Conv2D(64, (1, 1), activation='relu')(x4)

    # Concatenate the outputs from all branches
    x = layers.Concatenate()([x1, x2, x3, x4])

    # Flatten the output
    x = layers.Flatten()(x)

    # Output layer with a fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Construct and print the model
model = dl_model()
model.summary()