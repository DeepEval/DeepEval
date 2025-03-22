import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolution layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    block1_input = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(initial_conv)
    
    group1 = block1_input[0]
    group2 = block1_input[1]

    # Processing the first group
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group1)
    path2 = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path1)
    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path2)
    
    # Concatenate both groups
    block1_output = Concatenate()([path3, group2])

    # Block 2
    block2_input_shape = tf.shape(block1_output)
    height, width = block2_input_shape[1], block2_input_shape[2]
    channels = block2_input_shape[3]
    groups = 4
    channels_per_group = channels // groups

    # Reshape for channel shuffling
    reshaped_output = tf.reshape(block1_output, (block2_input_shape[0], height, width, groups, channels_per_group))
    permuted_output = tf.transpose(reshaped_output, perm=[0, 1, 2, 4, 3])  # Shuffle channels
    channel_shuffled_output = tf.reshape(permuted_output, (block2_input_shape[0], height, width, channels))

    # Flatten and fully connected layer
    flatten_layer = Flatten()(channel_shuffled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model