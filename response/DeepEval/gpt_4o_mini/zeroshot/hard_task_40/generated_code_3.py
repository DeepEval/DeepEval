import numpy as np
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Dropout, Reshape, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for MNIST images (28x28 pixels with 1 channel)
    input_layer = Input(shape=(28, 28, 1))

    # First block: Average pooling layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the outputs from pooling layers
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Concatenate flattened vectors
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer
    dense1 = Dense(128, activation='relu')(concatenated)

    # Reshape the output into a 4D tensor (assuming we want to reshape to 4x4x8)
    reshaped = Reshape((4, 4, 8))(dense1)

    # Second block: Parallel paths for feature extraction
    # Path 1: 1x1 convolution
    path1 = Conv2D(16, (1, 1), activation='relu')(reshaped)
    path1 = Dropout(0.5)(path1)

    # Path 2: Two stacked 3x3 convolutions
    path2 = Conv2D(16, (1, 1), activation='relu')(reshaped)
    path2 = Conv2D(16, (3, 3), activation='relu', padding='same')(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: One 3x3 convolution
    path3 = Conv2D(16, (1, 1), activation='relu')(reshaped)
    path3 = Conv2D(16, (3, 3), activation='relu', padding='same')(path3)
    path3 = Dropout(0.5)(path3)

    # Path 4: Average pooling followed by 1x1 convolution
    path4 = AveragePooling2D(pool_size=(2, 2))(reshaped)
    path4 = Conv2D(16, (1, 1), activation='relu')(path4)
    path4 = Dropout(0.5)(path4)

    # Concatenate outputs from all paths
    concatenated_paths = Concatenate()([path1, path2, path3, path4])

    # Final feature vector
    final_flatten = Flatten()(concatenated_paths)
    dense2 = Dense(64, activation='relu')(final_flatten)

    # Output layer for classification (10 classes for MNIST)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Summary of the model
model.summary()