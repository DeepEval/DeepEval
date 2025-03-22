import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    # 1x1 convolution for dimensionality reduction
    main_path = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)

    # Parallel convolutional layers (1x1 and 3x3)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(main_path)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)

    # Concatenate the outputs of the two convolutional layers
    concatenated = Concatenate()([conv1, conv2])

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Ensure the dimensions match in the addition
    combined = Add()([concatenated, branch_path])

    # Flatten the output
    flatten = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()