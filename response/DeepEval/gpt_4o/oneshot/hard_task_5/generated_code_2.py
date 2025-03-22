import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Dense, Add, Lambda, Concatenate, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split, Conv2D on splits, Concatenate
    def block1(input_tensor):
        # Splitting the input into 3 parts
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Apply 1x1 convolution to each part
        convs = [Conv2D(filters=x.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split) for split in splits]
        # Concatenate the parts along the channel dimension
        return Concatenate(axis=-1)(convs)

    # Block 2: Channel Shuffle
    def block2(input_tensor):
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        channels_per_group = channels // 3
        # Reshape to (height, width, groups, channels_per_group)
        reshaped = Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], height, width, 3, channels_per_group]))(input_tensor)
        # Permute dimensions to swap groups and channels_per_group
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
        # Reshape back to (height, width, channels)
        return Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], height, width, channels]))(permuted)

    # Block 3: 3x3 Depthwise Separable Convolution
    def block3(input_tensor):
        return DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

    # Main Path
    block1_out = block1(input_layer)
    block2_out = block2(block1_out)
    block3_out = block3(block2_out)
    block1_again = block1(block3_out)

    # Direct Branch connecting to input
    direct_branch = input_layer

    # Combine main path and direct branch
    combined = Add()([block1_again, direct_branch])

    # Flatten and fully connected layer for classification
    flatten = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model