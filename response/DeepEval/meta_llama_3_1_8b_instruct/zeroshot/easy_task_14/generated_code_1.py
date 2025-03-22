# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the model architecture using Functional API
    inputs = keras.Input(shape=(32, 32, 3), name='inputs')

    # Global average pooling to capture global information from the feature map
    x = layers.GlobalAveragePooling2D()(inputs)

    # Two fully connected layers to generate weights whose size is the same as the channels of the input
    x = layers.Dense(64, activation='relu')(x)
    weights = layers.Dense(3)(x)

    # Reshape weights to align with the input shape
    weights = layers.Reshape((3, 1, 1))(weights)

    # Element-wise multiplication of the input feature map with the weights
    x = layers.Multiply()([inputs, weights])

    # Flatten the result
    x = layers.Flatten()(x)

    # Final fully connected layer to obtain the final probability distribution
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Usage
if __name__ == "__main__":
    model = dl_model()
    print(model.summary())