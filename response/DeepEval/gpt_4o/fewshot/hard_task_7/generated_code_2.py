import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolution to adjust dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block_1(input_tensor):
        # Split input into two groups along the last dimension
        groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        conv1x1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_1)
        conv1x1_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        
        # Second group is passed through without modification
        unchanged_group = groups[1]
        
        # Concatenate the outputs of both groups
        output_tensor = Concatenate()([conv1x1_2, unchanged_group])
        return output_tensor

    def block_2(input_tensor):
        # Obtain shape of input
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 4
        channels_per_group = channels // groups
        
        # Reshape and permute for channel shuffle
        reshaped = Reshape(target_shape=(height, width, groups, channels_per_group))(input_tensor)
        permuted = Lambda(lambda x: tf.transpose(x, perm=(0, 1, 2, 4, 3)))(reshaped)
        reshaped_back = Reshape(target_shape=(height, width, channels))(permuted)
        
        return reshaped_back

    block1_output = block_1(input_tensor=initial_conv)
    block2_output = block_2(input_tensor=block1_output)
    
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model