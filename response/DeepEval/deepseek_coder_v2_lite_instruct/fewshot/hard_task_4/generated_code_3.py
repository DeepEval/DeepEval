import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolution to increase the number of channels
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Initial 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1)

    # Channel attention mechanism
    gap = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=depthwise_conv.get_shape()[-1] // 4, activation='relu')(gap)
    dense2 = Dense(units=depthwise_conv.get_shape()[-1], activation='sigmoid')(dense1)
    dense2_reshaped = Dense(units=depthwise_conv.get_shape()[-1], activation='sigmoid')(dense1)
    dense2_reshaped = dense2_reshaped.reshape((depthwise_conv.get_shape()[-1], 1, 1))

    # Apply channel attention weights
    attention_weighted_features = Multiply()([depthwise_conv, dense2_reshaped])

    # Add the initial input and the attention weighted features
    added_features = Add()([conv1, attention_weighted_features])

    # Flatten the output and pass through a fully connected layer
    flattened_features = Flatten()(added_features)
    output_layer = Dense(units=10, activation='softmax')(flattened_features)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model