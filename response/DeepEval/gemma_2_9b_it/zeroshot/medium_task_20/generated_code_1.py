import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(filters=16, kernel_size=(1, 1))(inputs)
    
    # Path 2: 1x1 -> 3x3 -> 3x3
    path2 = layers.Conv2D(filters=16, kernel_size=(1, 1))(inputs)
    path2 = layers.Conv2D(filters=32, kernel_size=(3, 3))(path2)
    path2 = layers.Conv2D(filters=64, kernel_size=(3, 3))(path2)

    # Path 3: 1x1 -> 3x3
    path3 = layers.Conv2D(filters=32, kernel_size=(1, 1))(inputs)
    path3 = layers.Conv2D(filters=64, kernel_size=(3, 3))(path3)

    # Path 4: Max Pooling -> 1x1
    path4 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)
    path4 = layers.Conv2D(filters=64, kernel_size=(1, 1))(path4)

    # Concatenate paths
    concatenated = layers.Concatenate(axis=-1)([path1, path2, path3, path4])

    # Flatten and dense layer
    x = layers.Flatten()(concatenated)
    x = layers.Dense(128, activation='relu')(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model