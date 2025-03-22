import keras
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, Reshape, Permute
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def block1(input_tensor):
        # Split input into two groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
        group1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        
        # Second group is passed through without modification
        group2 = split_groups[1]
        
        # Concatenate groups
        output_tensor = Concatenate()([group1, group2])
        
        return output_tensor
    
    def block2(input_tensor):
        # Obtain input shape
        input_shape = tf.shape(input_tensor)
        
        # Reshape input to (height, width, groups, channels_per_group)
        groups = 4
        channels = input_tensor.shape[-1]
        channels_per_group = channels // groups
        
        reshaped = Reshape((input_shape[1], input_shape[2], groups, channels_per_group))(input_tensor)
        
        # Swap third and fourth dimensions using permutation (channel shuffle)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to original shape
        reshaped_back = Reshape((input_shape[1], input_shape[2], channels))(permuted)
        
        return reshaped_back

    block1_output = block1(input_tensor=conv1)
    block2_output = block2(input_tensor=block1_output)
    
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model