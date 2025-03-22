import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First 1x1 convolution to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Depthwise separable convolution layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Second 1x1 convolution to reduce dimensionality
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)

    # Adding the output of the last layer to the original input
    added_layer = Add()([conv2, input_layer])

    # Flattening the output for the fully connected layer
    flatten_layer = Flatten()(added_layer)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model