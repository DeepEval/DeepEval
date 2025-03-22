import numpy as np
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Conv2D, Dropout, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))

    # First block: Average pooling layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer
    dense = Dense(128, activation='relu')(concatenated)
    reshaped = Reshape((1, 1, 128))(dense)  # Reshape to a 4D tensor

    # Second block: Four parallel paths for multi-scale feature extraction
    # Path 1: 1x1 Convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path1 = Dropout(0.5)(path1)

    # Path 2: Two 3x3 convolutions after 1x1 convolution
    path2 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path2 = Conv2D(32, (3, 3), activation='relu')(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: Single 3x3 convolution after 1x1 convolution
    path3 = Conv2D(32, (1, 1), activation='relu')(reshaped)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path3 = Dropout(0.5)(path3)

    # Path 4: Average pooling + 1x1 convolution
    path4 = AveragePooling2D(pool_size=(2, 2))(reshaped)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)
    path4 = Dropout(0.5)(path4)

    # Concatenate the outputs from all paths
    concatenated_paths = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated outputs
    flatten_paths = Flatten()(concatenated_paths)

    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flatten_paths)
    output_layer = Dense(10, activation='softmax')(dense1)  # 10 classes for MNIST

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()