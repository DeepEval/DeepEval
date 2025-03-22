import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape for the MNIST dataset (28x28 grayscale images)
    input_shape = (28, 28, 1)

    # Input layer
    inputs = Input(shape=input_shape)

    # Depthwise separable convolutional layer
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = Dropout(0.25)(x)  # Dropout layer to mitigate overfitting

    # 1x1 Convolutional layer for feature extraction
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(0.25)(x)  # Dropout layer to mitigate overfitting

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()