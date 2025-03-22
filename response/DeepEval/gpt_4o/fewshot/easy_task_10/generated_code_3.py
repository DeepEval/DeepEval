import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolution to increase dimensionality
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Depthwise separable convolution for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)

    # 1x1 convolution to reduce dimensionality
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(depthwise_conv)

    # Flatten and output classification probabilities
    flatten_layer = Flatten()(conv2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model