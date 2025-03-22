import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Lambda, Concatenate, Flatten, Dense, Reshape, Permute
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer to adjust input dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        # Split the input into two groups along the last dimension
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
        group1 = SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(group1)

        # Second group passes through unchanged
        group2 = split_tensor[1]

        # Concatenate the two groups
        output_tensor = Concatenate()([group1, group2])
        return output_tensor
    
    block1_output = block1(input_tensor=initial_conv)
    
    # Block 2
    def block2(input_tensor):
        # Get input shape
        shape = tf.shape(input_tensor)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        
        # Number of groups for shuffling
        groups = 4
        channels_per_group = channels // groups
        
        # Reshape input to (height, width, groups, channels_per_group)
        reshaped = Reshape((height, width, groups, channels_per_group))(input_tensor)
        
        # Permute dimensions to (height, width, channels_per_group, groups)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to the original input shape
        shuffled_output = Reshape((height, width, channels))(permuted)
        
        return shuffled_output
    
    block2_output = block2(input_tensor=block1_output)
    
    # Flatten and Dense layers for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model