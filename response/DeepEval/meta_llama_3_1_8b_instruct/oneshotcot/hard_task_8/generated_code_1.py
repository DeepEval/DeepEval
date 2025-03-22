import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Permute, Reshape, Lambda, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(path1)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    output_tensor = Concatenate()([path2, branch_path])

    # Block 2
    def channel_shuffle(input_tensor):
        channels = keras.backend.int_shape(input_tensor)[-1]
        groups = 4
        channels_per_group = channels // groups
        input_shape = keras.backend.int_shape(input_tensor)
        output_shape = (input_shape[0], input_shape[1], input_shape[2], groups, channels_per_group)
        output = Reshape(output_shape)(input_tensor)
        output = Permute((2, 3, 4, 1))(output)
        output = Reshape(input_shape)(output)
        return output

    shuffled_features = channel_shuffle(output_tensor)

    # Batch normalization and flatten
    bath_norm = BatchNormalization()(shuffled_features)
    flatten_layer = Flatten()(bath_norm)

    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model