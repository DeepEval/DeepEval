import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 color channel
    inputs = layers.Input(shape=input_shape)

    # Block 1: Parallel paths with average pooling of different scales
    path1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten and apply dropout to each path
    flat1 = layers.Flatten()(path1)
    flat2 = layers.Flatten()(path2)
    flat3 = layers.Flatten()(path3)

    drop1 = layers.Dropout(0.5)(flat1)
    drop2 = layers.Dropout(0.5)(flat2)
    drop3 = layers.Dropout(0.5)(flat3)

    # Concatenate the outputs of the three paths
    concatenated = layers.Concatenate()([drop1, drop2, drop3])

    # Fully connected layer before Block 2
    fc1 = layers.Dense(128, activation='relu')(concatenated)
    reshaped = layers.Reshape((2, 2, 32))(fc1)  # Reshape to a suitable format for Block 2

    # Block 2: Multiple branch connections for feature extraction
    # 1st branch: 1x1 Convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)

    # 2nd branch: 1x1 Convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)

    # 3rd branch: 1x1 and 3x3 Convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu')(branch3)

    # 4th branch: Average Pooling and 1x1 Convolution
    branch4 = layers.AveragePooling2D(pool_size=(2, 2))(reshaped)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate all branches
    combined = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the combined output
    combined_flat = layers.Flatten()(combined)

    # Fully connected layers for classification
    fc2 = layers.Dense(128, activation='relu')(combined_flat)
    outputs = layers.Dense(10, activation='softmax')(fc2)  # 10 classes for MNIST digits

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()