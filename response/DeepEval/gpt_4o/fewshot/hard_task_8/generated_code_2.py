import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Lambda, Dense, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Primary path
    conv1_primary = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv_primary = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_primary)
    conv2_primary = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv_primary)

    # Block 1: Branch path
    depthwise_conv_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv_branch)

    # Concatenate features from both paths
    concatenated_features = Concatenate(axis=-1)([conv2_primary, conv_branch])

    # Block 2: Channel shuffling
    def channel_shuffle(x):
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        groups = 4  # Number of groups for shuffling
        channels_per_group = channels // groups

        # Reshape and permute
        reshaped = Reshape(target_shape=(height, width, groups, channels_per_group))(x)
        transposed = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])
        shuffled = Reshape(target_shape=(height, width, channels))(transposed)
        
        return shuffled

    shuffled_features = Lambda(channel_shuffle)(concatenated_features)

    # Flatten and Fully Connected Layer for classification
    flatten = Flatten()(shuffled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model