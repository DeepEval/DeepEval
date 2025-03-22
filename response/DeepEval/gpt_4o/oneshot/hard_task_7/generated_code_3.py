import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer to adjust input dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        # Split into two groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group processing
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
        group1 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        
        # Second group passes through unchanged
        group2 = split_groups[1]
        
        # Concatenate the outputs of both groups
        output_tensor = Concatenate()([group1, group2])
        
        return output_tensor
    
    block1_output = block1(initial_conv)
    
    # Block 2
    def block2(input_tensor):
        # Get the shape of the input
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        
        groups = 4
        channels_per_group = channels // groups
        
        # Reshape input into (height, width, groups, channels_per_group)
        reshaped = tf.reshape(input_tensor, [-1, height, width, groups, channels_per_group])
        
        # Permute dimensions to shuffle channels
        permuted = tf.transpose(reshaped, [0, 1, 2, 4, 3])
        
        # Reshape back to original shape
        shuffled = tf.reshape(permuted, [-1, height, width, channels])
        
        return shuffled
    
    block2_output = Lambda(lambda x: block2(x))(block1_output)
    
    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model