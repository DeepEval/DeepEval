import numpy as np
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer with a 1x1 kernel
    conv1 = Conv2D(32, (1, 1), activation='relu')(Input(shape=input_shape))

    # Define the first average pooling layer with a 2x2 kernel and stride of 2
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2)(conv1)

    # Define the second convolutional layer with a 2x2 kernel
    conv2 = Conv2D(64, (2, 2), activation='relu')(pool1)

    # Define the second average pooling layer with a 4x4 kernel and stride of 4
    pool2 = AveragePooling2D(pool_size=(4, 4), strides=4)(conv2)

    # Define the third convolutional layer with a 4x4 kernel
    conv3 = Conv2D(128, (4, 4), activation='relu')(pool2)

    # Define the third average pooling layer with a 8x8 kernel and stride of 8
    pool3 = AveragePooling2D(pool_size=(8, 8), strides=8)(conv3)

    # Flatten the output of the third pooling layer
    flattened = Flatten()(pool3)

    # Define the first fully connected layer with 128 units
    fc1 = Dense(128, activation='relu')(flattened)

    # Define the second fully connected layer with 10 units (number of classes)
    fc2 = Dense(10)(fc1)

    # Define the output layer with softmax activation
    output = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=Input(shape=input_shape), outputs=output)

    # Compile the model with the Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model