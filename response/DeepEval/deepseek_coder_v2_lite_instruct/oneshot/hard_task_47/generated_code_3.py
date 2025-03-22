import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: Depthwise separable convolutions
    def depthwise_conv_block(input_tensor, kernel_size):
        conv = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        return BatchNormalization()(conv)

    split_1x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    split_3x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    split_5x5 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    conv1x1_1 = depthwise_conv_block(split_1x1[0], (1, 1))
    conv1x1_2 = depthwise_conv_block(split_1x1[1], (1, 1))
    conv1x1_3 = depthwise_conv_block(split_1x1[2], (1, 1))

    conv3x3_1 = depthwise_conv_block(split_3x3[0], (3, 3))
    conv3x3_2 = depthwise_conv_block(split_3x3[1], (3, 3))
    conv3x3_3 = depthwise_conv_block(split_3x3[2], (3, 3))

    conv5x5_1 = depthwise_conv_block(split_5x5[0], (5, 5))
    conv5x5_2 = depthwise_conv_block(split_5x5[1], (5, 5))
    conv5x5_3 = depthwise_conv_block(split_5x5[2], (5, 5))

    concatenated = Concatenate(axis=-1)([conv1x1_1, conv3x3_1, conv5x5_1])
    concatenated = Concatenate(axis=-1)([concatenated, conv1x1_2, conv3x3_2, conv5x5_2])
    concatenated = Concatenate(axis=-1)([concatenated, conv1x1_3, conv3x3_3, conv5x5_3])

    # Second block: Multiple branches for feature extraction
    def branch_extraction(input_tensor, branch_type):
        if branch_type == '1x1':
            return Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        elif branch_type == '3x3':
            return Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        elif branch_type == '1x7-7x1':
            x = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(input_tensor)
            return Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(x)

    branch1 = branch_extraction(concatenated, '1x1')
    branch2 = branch_extraction(concatenated, '3x3')
    branch3 = branch_extraction(concatenated, '1x7-7x1')
    branch4 = branch_extraction(concatenated, '3x3')
    branch5 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(concatenated)

    concatenated_branches = Concatenate(axis=-1)([branch1, branch2, branch3, branch4, branch5])

    # Flatten and fully connected layers
    flattened = Flatten()(concatenated_branches)
    dense1 = Dense(units=256, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model