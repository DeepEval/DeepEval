from tensorflow.keras.layers import (Input, Lambda, Conv2D, DepthwiseConv2D,
                                 SeparableConv2D, MaxPooling2D, AveragePooling2D,
                                 BatchNormalization, Activation, concatenate,
                                 Dense)
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1: Feature extraction using depthwise separable convolutional layers
    split_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    conv_outputs = []
    for kernel_size in [(1, 1), (3, 3), (5, 5)]:
        conv_output = DepthwiseConv2D(kernel_size, padding='same')(split_input)
        conv_output = BatchNormalization()(conv_output)
        conv_output = Activation('relu')(conv_output)
        conv_outputs.append(conv_output)

    # Concatenate outputs from different kernel sizes
    concat_outputs = concatenate(conv_outputs)

    # Block 2: Multi-branch feature extraction
    branch_outputs = []
    for filters in [[64, 64, 256], [64, 128, 256], [32, 64, 256]]:
        branch_output = inputs
        for filter_size in filters:
            branch_output = Conv2D(filter_size, (1, 1), padding='same')(branch_output)
            branch_output = BatchNormalization()(branch_output)
            branch_output = Activation('relu')(branch_output)
        branch_outputs.append(branch_output)

    # Concatenate outputs from different branches
    concat_branch_outputs = concatenate(branch_outputs)

    # Pooling and fully connected layers
    global_pool = AveragePooling2D()(concat_branch_outputs)
    flatten = Flatten()(global_pool)
    dense = Dense(512, activation='relu')(flatten)
    outputs = Dense(10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model