import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for MNIST images
    inputs = layers.Input(shape=(28, 28, 1))

    # Block 1: Three parallel paths with max pooling
    # Path 1: 1x1 pooling
    path1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path1 = layers.Flatten()(path1)
    path1 = layers.Dropout(0.5)(path1)

    # Path 2: 2x2 pooling
    path2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path2 = layers.Flatten()(path2)
    path2 = layers.Dropout(0.5)(path2)

    # Path 3: 4x4 pooling
    path3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    path3 = layers.Flatten()(path3)
    path3 = layers.Dropout(0.5)(path3)

    # Concatenate the outputs of all paths
    concat1 = layers.Concatenate()([path1, path2, path3])

    # Fully connected layer
    fc1 = layers.Dense(128, activation='relu')(concat1)

    # Reshape output for block 2 (to fit the specifications)
    reshaped = layers.Reshape((1, 1, 128))(fc1)

    # Block 2: Four parallel paths for feature extraction
    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)
    path2 = layers.Conv2D(32, (1, 7), activation='relu')(path2)
    path2 = layers.Conv2D(32, (7, 1), activation='relu')(path2)

    # Path 3: 1x1 convolution followed by alternating 7x1 and 1x7 convolutions
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)
    path3 = layers.Conv2D(32, (7, 1), activation='relu')(path3)
    path3 = layers.Conv2D(32, (1, 7), activation='relu')(path3)

    # Path 4: Average pooling followed by 1x1 convolution
    path4 = layers.AveragePooling2D(pool_size=(2, 2))(reshaped)
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate the outputs of all paths
    concat2 = layers.Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flatten = layers.Flatten()(concat2)

    # Fully connected layers for classification
    fc2 = layers.Dense(128, activation='relu')(flatten)
    outputs = layers.Dense(10, activation='softmax')(fc2)  # 10 classes for MNIST

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()  # Uncomment to view the model architecture