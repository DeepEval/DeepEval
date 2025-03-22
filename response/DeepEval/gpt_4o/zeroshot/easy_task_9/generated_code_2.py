import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with a single channel
    num_classes = 10  # Number of classes for MNIST

    # Input layer
    inputs = Input(shape=input_shape)

    # 1x1 Convolutional layer to increase dimensionality
    x = Conv2D(32, (1, 1), activation='relu', strides=1, padding='same')(inputs)

    # 3x3 Depthwise Separable Convolutional layer for feature extraction
    x = SeparableConv2D(32, (3, 3), activation='relu', strides=1, padding='same')(x)

    # 1x1 Convolutional layer to reduce dimensionality
    x = Conv2D(1, (1, 1), activation='relu', strides=1, padding='same')(x)

    # Add the output to the original input
    x = Add()([x, inputs])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()