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

    # Define Block 1: Global Average Pooling and Fully Connected Layers
    block1 = keras.Model(inputs=keras.Input(shape=(32, 32, 3)),
                         outputs=layers.GlobalAveragePooling2D()(layers.Input(shape=(32, 32, 3))))
    block1.add(layers.Dense(10, activation='relu'))
    block1.add(layers.Dense(10, activation='relu'))

    # Define Block 2: Convolutional Layers and Max Pooling Layer
    block2 = keras.Model(inputs=keras.Input(shape=(32, 32, 3)),
                         outputs=layers.Conv2D(32, (3, 3), activation='relu')(layers.Input(shape=(32, 32, 3))))
    block2.add(layers.Conv2D(32, (3, 3), activation='relu'))
    block2.add(layers.MaxPooling2D((2, 2)))
    block2.add(layers.Flatten())

    # Define the branch from Block 1 to Block 2
    branch = keras.Model(inputs=block1.input,
                         outputs=layers.Concatenate()([block1.output, layers.Dense(128, activation='relu')(block1.output)]))

    # Combine the outputs from Block 2 and the branch
    combined = keras.Model(inputs=block2.input,
                           outputs=layers.Add()([block2.output, branch.output]))

    # Define the final output layer
    output = layers.Concatenate()([combined.output, branch.output])
    output = layers.Dense(64, activation='relu')(output)
    output = layers.Dense(10, activation='softmax')(output)

    # Create the final model
    model = keras.Model(inputs=block2.input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model