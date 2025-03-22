import keras
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))  # Input shape for MNIST images

    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    block1_input = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(conv1)
    group1 = block1_input[0]
    group2 = block1_input[1]

    # Operations on group 1
    conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1_1)
    conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)

    # Concatenate the results
    block1_output = Concatenate()([conv1x1_2, group2])

    # Block 2
    # Reshape the output into groups
    num_groups = 4
    height, width, channels = block1_output.shape[1], block1_output.shape[2], block1_output.shape[3]
    channels_per_group = channels // num_groups

    reshaped = Reshape((height, width, num_groups, channels_per_group))(block1_output)

    # Permute the dimensions for channel shuffling
    permuted = tf.transpose(reshaped, perm=[0, 1, 3, 2, 4])  # (height, width, groups, channels_per_group)

    # Reshape back to original shape
    shuffled_output = Reshape((height, width, channels))(permuted)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(shuffled_output)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)  # Final classification layer

    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model