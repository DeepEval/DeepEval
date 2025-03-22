import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def block_1(input_tensor):
        # Split the input into two groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        group1_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        group1_depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1_conv1x1)
        group1_conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1_depthwise)
        
        # Second group remains unchanged
        group2 = split_layer[1]
        
        # Concatenate outputs
        merged_group = Concatenate()([group1_conv1x1_2, group2])
        
        return merged_group
    
    block1_output = block_1(conv_initial)
    
    def block_2(input_tensor):
        # Get the shape of the input
        input_shape = tf.keras.backend.int_shape(input_tensor)
        
        # Reshape into groups
        reshaped = Reshape(target_shape=(input_shape[1], input_shape[2], 2, input_shape[3] // 2))(input_tensor)
        
        # Permute to swap the third and fourth dimensions
        permuted = tf.keras.backend.permute_dimensions(reshaped, (0, 1, 2, 4, 3))
        
        # Flatten the result
        flattened = Flatten()(permuted)
        
        return flattened
    
    block2_output = block_2(block1_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(block2_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model