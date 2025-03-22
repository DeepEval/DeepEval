import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # First 1x1 convolution layer to reduce dimensionality
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)

    # 3x3 convolutional layer to extract features
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)

    # Another 1x1 convolution layer to restore dimensionality
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer with 10 neurons for classification
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model