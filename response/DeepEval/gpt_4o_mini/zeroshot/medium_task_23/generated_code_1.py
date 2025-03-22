import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Path 1: 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path2 = layers.Conv2D(32, (1, 7), padding='same', activation='relu')(path2)
    path2 = layers.Conv2D(32, (7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by two sets of 1x7 and 7x1 Convolutions
    path3 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path3 = layers.Conv2D(32, (1, 7), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(32, (7, 1), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(32, (1, 7), padding='same', activation='relu')(path3)
    path3 = layers.Conv2D(32, (7, 1), padding='same', activation='relu')(path3)

    # Path 4: Average Pooling followed by 1x1 Convolution
    path4 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    path4 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(path4)

    # Concatenate all paths
    concatenated = layers.concatenate([path1, path2, path3, path4])

    # Flatten the output
    flatten = layers.Flatten()(concatenated)

    # Fully connected layer for classification
    dense = layers.Dense(128, activation='relu')(flatten)
    outputs = layers.Dense(10, activation='softmax')(dense)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # To display the model architecture