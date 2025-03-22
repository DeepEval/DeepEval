import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, concatenate

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(32, kernel_size=1, activation='relu')(inputs)
    branch1 = Dropout(0.2)(branch1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(32, kernel_size=1, activation='relu')(inputs)
    branch2 = Conv2D(32, kernel_size=3, activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(32, kernel_size=1, activation='relu')(inputs)
    branch3 = Conv2D(32, kernel_size=3, activation='relu')(branch3)
    branch3 = Conv2D(32, kernel_size=3, activation='relu')(branch3)
    branch3 = Dropout(0.2)(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=3, strides=2, padding='same')(inputs)
    branch4 = Conv2D(32, kernel_size=1, activation='relu')(branch4)
    branch4 = Dropout(0.2)(branch4)

    # Concatenate the outputs from all branches
    merged = concatenate([branch1, branch2, branch3, branch4])

    # Add a flatten layer to reshape the output into a 1D vector
    merged = Flatten()(merged)

    # Add three fully connected layers for classification
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
print(model.summary())