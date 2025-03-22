import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, DepthwiseConv2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    split_input = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    depthwise_conv1_1x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(split_input[0])
    depthwise_conv2_1x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(split_input[1])
    depthwise_conv3_1x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(split_input[2])

    # Depthwise separable convolution with 3x3 kernel
    depthwise_conv1_3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(split_input[0])
    depthwise_conv2_3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(split_input[1])
    depthwise_conv3_3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(split_input[2])

    # Depthwise separable convolution with 5x5 kernel
    depthwise_conv1_5x5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(split_input[0])
    depthwise_conv2_5x5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(split_input[1])
    depthwise_conv3_5x5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(split_input[2])

    # Concatenate outputs of main path
    concat_main_path = Concatenate(axis=3)([depthwise_conv1_1x1, depthwise_conv2_1x1, depthwise_conv3_1x1,
                                          depthwise_conv1_3x3, depthwise_conv2_3x3, depthwise_conv3_3x3,
                                          depthwise_conv1_5x5, depthwise_conv2_5x5, depthwise_conv3_5x5])

    # Batch normalization and flattening
    concat_main_path = BatchNormalization()(concat_main_path)
    flatten_main_path = Flatten()(concat_main_path)

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = BatchNormalization()(branch_path)

    # Concatenate outputs of branch path
    concat_branch_path = Concatenate()([branch_path])

    # Add outputs of main and branch paths
    concat_output = Add()([concat_main_path, concat_branch_path])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model