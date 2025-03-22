import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape for MNIST images
    input_shape = (28, 28, 1)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # 1x1 Convolutional layer to increase dimensionality
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(inputs)

    # 3x3 Depthwise separable convolutional layer
    x = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(x)

    # 1x1 Convolutional layer to reduce dimensionality
    x = Conv2D(1, (1, 1), strides=1, padding='same', activation='relu')(x)

    # Adding the output from the last convolutional layer to the original input
    x = Add()([x, inputs])

    # Flattening layer
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
# model = dl_model()
# model.summary()