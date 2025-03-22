import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # Define a specialized block for the branches
    def branch(input_tensor):
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        return x

    # Create three branches
    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)

    # Concatenate the outputs of the branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense1 = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    dense2 = layers.Dropout(0.5)(dense2)

    # Output layer for classification (10 classes for MNIST)
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Print model summary
model.summary()

# Fit the model (you can adjust epochs and batch_size as needed)
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))