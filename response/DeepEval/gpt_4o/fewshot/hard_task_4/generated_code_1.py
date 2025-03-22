import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input's channels threefold
    conv1 = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Compute channel attention weights
    gap = GlobalAveragePooling2D()(depthwise_conv)
    fc1 = Dense(units=9 // 2, activation='relu')(gap)
    fc2 = Dense(units=9, activation='sigmoid')(fc1)
    channel_weights = Multiply()([depthwise_conv, fc2])

    # Reduce dimensionality with a 1x1 convolution
    conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_weights)

    # Combine with the initial input
    combined = Add()([input_layer, conv2])

    # Flatten and classify
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model