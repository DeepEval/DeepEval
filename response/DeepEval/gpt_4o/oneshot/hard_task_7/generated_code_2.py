import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Concatenate, Lambda, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolution to adjust dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        # Splitting the input tensor into two groups along the last dimension
        group1, group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)

        # First group operations: 1x1 Conv -> Depthwise Separable Conv -> 1x1 Conv
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)

        # Concatenate the modified group1 with unmodified group2
        output_tensor = Concatenate()([group1, group2])
        
        return output_tensor
    
    block1_output = block1(initial_conv)
    
    # Block 2
    def block2(input_tensor):
        # Get shape of input tensor
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        
        # Define number of groups and channels per group
        groups = 4
        channels_per_group = channels // groups
        
        # Reshape input into (height, width, groups, channels_per_group)
        reshaped = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], height, width, groups, channels_per_group)))(input_tensor)
        
        # Swap the third and fourth dimensions
        permuted = Lambda(lambda x: tf.transpose(x, perm=(0, 1, 2, 4, 3)))(reshaped)
        
        # Reshape back to original shape
        shuffled = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], height, width, channels)))(permuted)
        
        return shuffled
    
    block2_output = block2(block1_output)
    
    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model