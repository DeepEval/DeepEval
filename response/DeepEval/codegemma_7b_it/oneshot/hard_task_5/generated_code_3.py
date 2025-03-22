import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, DepthwiseConv2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    conv_groups = []
    for x in split_input:
        conv_groups.append(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))
    fused_features = Concatenate(axis=-1)(conv_groups)

    # Block 2
    shape_info = keras.backend.int_shape(fused_features)
    reshape_input = Reshape((-1, shape_info[1], shape_info[2], 3))(fused_features)
    permuted_input = Permute((0, 2, 3, 1))(reshape_input)
    reshaped_input = Reshape((shape_info[1], shape_info[2], 3, 64))(permuted_input)
    channel_shuffled_features = Reshape((shape_info[1], shape_info[2], 64, 3))(reshaped_input)

    # Block 3
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_shuffled_features)

    # Branch
    branch_input = input_layer
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)

    # Output
    combined_features = keras.layers.add([depthwise_conv, branch_conv])
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model