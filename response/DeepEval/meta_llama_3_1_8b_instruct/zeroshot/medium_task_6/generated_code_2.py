# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Initial convolution
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)

    # Three parallel blocks
    block1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(block1)
    block1 = layers.BatchNormalization()(block1)

    block2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(block2)
    block2 = layers.BatchNormalization()(block2)

    block3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(block3)
    block3 = layers.BatchNormalization()(block3)

    # Add the outputs of the blocks to enhance feature representation
    x = layers.Add()([x, block1, block2, block3])

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Softmax output
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()