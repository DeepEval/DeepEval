import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def specialized_block(input_tensor, filters, kernel_size=(3, 3), dropout_rate=0.3):
    # Depthwise Separable Convolution
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
    x = Dropout(dropout_rate)(x)
    # 1x1 Convolution to extract features
    x = Conv2D(filters, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    return x

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels and grayscale
    num_classes = 10  # There are 10 classes in MNIST

    inputs = Input(shape=input_shape)

    # Branch 1
    branch1 = specialized_block(inputs, filters=32)

    # Branch 2
    branch2 = specialized_block(inputs, filters=64)

    # Branch 3
    branch3 = specialized_block(inputs, filters=128)

    # Concatenating the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated outputs
    x = Flatten()(concatenated)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    # Output layer for classification
    outputs = Dense(num_classes, activation='softmax')(x)

    # Build the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to use the model
model = dl_model()
model.summary()