import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, DepthwiseConv2D, BatchNormalization, ReLU
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    conv1_relu = ReLU()(conv1_bn)
    
    # 3x3 depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1_relu)
    depthwise_conv_bn = BatchNormalization()(depthwise_conv)
    depthwise_conv_relu = ReLU()(depthwise_conv_bn)
    
    # 1x1 convolutional layer to reduce dimensionality with stride 2
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same')(depthwise_conv_relu)
    conv2_bn = BatchNormalization()(conv2)
    conv2_relu = ReLU()(conv2_bn)
    
    # Flatten the output and add a fully connected layer for classification
    flatten_layer = Flatten()(conv2_relu)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model