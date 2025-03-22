import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3)) 

    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 + 3x3 convolution
    branch2 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 + 5x5 convolution
    branch3 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(branch3)

    # Branch 4: 3x3 max pooling + 1x1 convolution
    branch4 = layers.MaxPooling2D(pool_size=(3, 3))(inputs)
    branch4 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate branches
    merged = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)

    # Flatten and fully connected layers
    flattened = layers.Flatten()(merged)
    dense1 = layers.Dense(128, activation='relu')(flattened)
    outputs = layers.Dense(10, activation='softmax')(dense1)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model