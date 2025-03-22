import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def depthwise_separable_conv(x, kernel_size):
        depthwise = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_constraint=None, use_bias=False)(x)
        pointwise = Conv2D(filters=64, kernel_size=(1, 1), padding='same', use_bias=False)(depthwise)
        return BatchNormalization()(pointwise)

    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1_1x1 = depthwise_separable_conv(split_1[0], kernel_size=(1, 1))
    conv1_3x3 = depthwise_separable_conv(split_1[1], kernel_size=(3, 3))
    conv1_5x5 = depthwise_separable_conv(split_1[2], kernel_size=(5, 5))
    concatenated1 = Concatenate(axis=-1)([conv1_1x1, conv1_3x3, conv1_5x5])

    # Second block
    def branch_block(x, kernel_sizes):
        outputs = []
        for kernel_size in kernel_sizes:
            if kernel_size == (1, 1):
                branch = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
            elif kernel_size == (1, 7) and kernel_size == (7, 1):
                branch = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
            else:
                branch = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
            outputs.append(branch)
        return Concatenate(axis=-1)(outputs)

    branch1 = branch_block(concatenated1, kernel_sizes=[(1, 1), (3, 3)])
    branch2 = branch_block(concatenated1, kernel_sizes=[(1, 1), (1, 7), (7, 1), (3, 3)])
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(concatenated1)
    concatenated2 = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concatenated2)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model