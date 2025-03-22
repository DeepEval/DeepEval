# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer to accept 32x32x3 CIFAR-10 images
    inputs = keras.Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = layers.Conv2D(32, (7, 1), activation='relu')(path2)
    path2 = layers.Conv2D(32, (1, 7), activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path3 = layers.Conv2D(32, (7, 1), activation='relu')(path3)
    path3 = layers.Conv2D(32, (1, 7), activation='relu')(path3)
    path3 = layers.Conv2D(32, (7, 1), activation='relu')(path3)
    path3 = layers.Conv2D(32, (1, 7), activation='relu')(path3)

    # Path 4: Average pooling followed by 1x1 convolution
    path4 = layers.AveragePooling2D((8, 8))(inputs)
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate the outputs of all paths
    concatenated = layers.Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # Output layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model