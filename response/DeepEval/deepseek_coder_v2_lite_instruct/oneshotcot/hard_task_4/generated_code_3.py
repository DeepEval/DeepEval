import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Increase dimensionality by a factor of 3 with a 1x1 convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Depthwise separable convolution using 3x3
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Channel attention mechanism
    gap = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=depthwise_conv.shape[-1], activation='relu')(gap)
    dense2 = Dense(units=depthwise_conv.shape[-1], activation='sigmoid')(dense1)
    reshape = Reshape((1, 1, depthwise_conv.shape[-1]))(dense2)
    attention_weights = Multiply()([depthwise_conv, reshape])

    # Add the weighted features back to the original features
    combined = Add()([attention_weights, depthwise_conv])

    # Flatten the output
    flatten_layer = Flatten()(combined)

    # Fully connected layer for classification
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model