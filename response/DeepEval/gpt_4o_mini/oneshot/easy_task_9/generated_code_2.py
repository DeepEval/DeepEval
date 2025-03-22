import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolution to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 Depthwise Separable Convolution for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # 1x1 Convolution to reduce dimensionality
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)

    # Adding the output from the last convolution to the original input
    added_output = Add()([conv2, input_layer])

    # Flattening the output and adding a fully connected layer
    flatten_layer = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model