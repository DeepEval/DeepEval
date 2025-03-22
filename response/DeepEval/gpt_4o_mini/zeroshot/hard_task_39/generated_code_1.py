import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D

def dl_model():
    # Input layer for the MNIST images, which are 28x28 pixels with 1 channel (grayscale)
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Max Pooling layers with varying scales
    # Max Pooling 1x1
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    flatten1 = Flatten()(pool1)

    # Max Pooling 2x2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    flatten2 = Flatten()(pool2)

    # Max Pooling 4x4
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    flatten3 = Flatten()(pool3)

    # Concatenate the results of the max pooling layers
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer to transform output into a suitable shape for Block 2
    fc_layer = Dense(128, activation='relu')(concatenated)

    # Reshape to create a 4D tensor (1, 4, 4, 8)
    reshaped = Reshape((4, 4, 8))(fc_layer)

    # Block 2: Multiple branches
    # 1x1 Convolution branch
    conv1 = Conv2D(16, (1, 1), activation='relu', padding='same')(reshaped)

    # 3x3 Convolution branch
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(reshaped)

    # 5x5 Convolution branch
    conv3 = Conv2D(16, (5, 5), activation='relu', padding='same')(reshaped)

    # 3x3 Max Pooling branch
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)

    # Concatenate outputs from all branches
    concatenated_branches = Concatenate()([conv1, conv2, conv3, pool4])

    # Flatten the output for classification
    flatten_output = Flatten()(concatenated_branches)

    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flatten_output)  # 10 classes for MNIST digits

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # Display the model architecture