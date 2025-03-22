import keras
from keras.layers import Input, Conv2D, Dense, Lambda, Concatenate, Add, DepthwiseConv2D, GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    
    def block1(input_tensor):
        # Split the input into 3 groups along the channel dimension
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with a 1x1 Convolution to reduce the channel dimension
        processed_splits = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), 
                                   activation='relu')(split) for split in split_layer]
        
        # Concatenate the processed splits along the channel dimension
        output_tensor = Concatenate(axis=-1)(processed_splits)
        return output_tensor
    
    def block2(input_tensor):
        # Get the shape of the input tensor
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        
        # Reshape and permute dimensions to achieve channel shuffling
        reshaped = tf.reshape(input_tensor, [-1, height, width, 3, channels // 3])
        permuted = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])
        shuffled_output = tf.reshape(permuted, [-1, height, width, channels])
        
        return shuffled_output
    
    def block3(input_tensor):
        # Apply a 3x3 Depthwise Separable Convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1, activation='relu')(input_tensor)
        return depthwise_conv

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block3_output = block3(block2_output)
    block1_output_again = block1(block3_output)
    
    # Direct branch from input
    direct_branch = input_layer
    
    # Combine outputs from the main path and direct branch
    combined_output = Add()([block1_output_again, direct_branch])
    
    # Global Average Pooling and Fully Connected Layer for classification
    global_avg_pooling = GlobalAveragePooling2D()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pooling)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model