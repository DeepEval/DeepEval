import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality threefold with a 1x1 convolution
    expand_channels = Conv2D(filters=9, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features with a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(expand_channels)

    # Compute channel attention weights
    global_avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    fc1 = Dense(units=9 // 2, activation='relu')(global_avg_pool)  # Reduce by half
    fc2 = Dense(units=9, activation='sigmoid')(fc1)  # Restore to original channel size

    # Reshape and multiply to achieve channel attention weighting
    channel_weights = Multiply()([depthwise_conv, fc2])

    # Reduce dimensionality back with a 1x1 convolution
    reduced_channels = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(channel_weights)

    # Combine with the initial input
    combined_output = Add()([input_layer, reduced_channels])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model