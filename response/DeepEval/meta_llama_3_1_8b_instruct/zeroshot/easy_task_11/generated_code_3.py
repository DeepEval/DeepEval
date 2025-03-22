# Import necessary packages
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

# Define the function to create the deep learning model
def dl_model():
    # Define the input shape for the MNIST dataset (28x28 grayscale images)
    input_shape = (28, 28, 1)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Apply the average pooling layer with a 5x5 window and a 3x3 stride
    x = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(inputs)

    # Apply the 1x1 convolutional layer to enhance depth and introduce nonlinearity
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Apply the first fully connected layer
    x = Dense(128, activation='relu')(x)

    # Apply the dropout layer to mitigate overfitting
    x = Dropout(0.2)(x)

    # Apply the second fully connected layer
    outputs = Dense(10, activation='softmax')(x)

    # Create the model by combining the input and output layers
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the deep learning model
model = dl_model()
model.summary()