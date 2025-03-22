import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer to adjust input dimensions
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        # Split the input into two groups along the last dimension
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group processing
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
        group1 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        
        # Second group remains unchanged
        group2 = split_groups[1]
        
        # Concatenate the outputs of both groups
        output_tensor = Concatenate()([group1, group2])
        
        return output_tensor
    
    # Block 2
    def block2(input_tensor):
        # Reshape the input into groups
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        groups = 4
        channels_per_group = channels // groups
        
        reshaped = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], height, width, groups, channels_per_group)))(input_tensor)
        
        # Permute the dimensions for channel shuffling
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
        
        # Reshape back to the original dimensions
        output_tensor = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], height, width, channels)))(permuted)
        
        return output_tensor
    
    # Pass through Block 1
    block1_output = block1(initial_conv)
    
    # Pass through Block 2
    block2_output = block2(block1_output)
    
    # Flatten and Fully Connected Layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model