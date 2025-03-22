import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups for main path
    group1, group2, group3 = Lambda(lambda t: tf.split(t, 3, axis=3))(input_layer)

    # Depthwise separable convolutions for main path
    def separable_conv(tensor, kernel_size):
        conv = Conv2D(filters=kernel_size ** 2, kernel_size=kernel_size, strides=1, padding='same')(tensor)
        bn = BatchNormalization()(conv)
        relu = tf.nn.relu(bn)
        return MaxPooling2D(pool_size=(kernel_size, kernel_size), strides=1, padding='same')(relu)

    # 1x1, 3x3, and 5x5 convolutions
    conv_1x1 = separable_conv(group1, 1)
    conv_3x3 = separable_conv(group2, 3)
    conv_5x5 = separable_conv(group3, 5)

    # Concatenate outputs from main path
    main_path_output = Concatenate(axis=3)([conv_1x1, conv_3x3, conv_5x5])

    # Branch path
    branch_input = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_path_output)

    # Add a 1x1 convolution to match channel numbers with main path
    branch_output = Conv2D(filters=kernel_size ** 2, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch_input)

    # Add branch output to main path output
    final_output = Concatenate(axis=3)([main_path_output, branch_output])

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(final_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model