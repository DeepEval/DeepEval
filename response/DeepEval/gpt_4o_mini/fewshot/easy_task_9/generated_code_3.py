import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer for MNIST images (28x28 pixels, 1 channel)
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer to increase dimensionality
    conv1x1_increase = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_increase)

    # Second 1x1 convolutional layer to reduce dimensionality
    conv1x1_reduce = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)

    # Adding the output of the last layer to the original input layer
    added_output = Add()([conv1x1_reduce, input_layer])

    # Flatten the output for the fully connected layer
    flatten_layer = Flatten()(added_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model