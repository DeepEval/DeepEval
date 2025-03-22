import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Permute, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        # Split the input into three groups
        split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Process each group with a 1x1 convolutional layer
        conv_groups = [Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group) for group in split_1]
        # Concatenate the outputs along the channel dimension
        fused_features = tf.concat(conv_groups, axis=-1)
        return fused_features
    
    def block_2(input_tensor):
        # Get the shape of the input tensor
        shape = input_tensor.shape
        # Reshape into three groups
        reshaped = Lambda(lambda x: tf.reshape(x, (shape[1], shape[2], 3, shape[3] // 3)))(input_tensor)
        # Permute dimensions to swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        # Reshape back to the original shape
        shuffled_features = Lambda(lambda x: tf.reshape(x, (shape[1], shape[2], shape[3] * 3)))(permuted)
        return shuffled_features
    
    def block_3(input_tensor):
        # Process with a 3x3 depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(input_tensor)
        return depthwise_conv
    
    # Apply blocks sequentially
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)
    
    # Add a branch that connects directly to the input
    branch = input_layer
    
    # Combine the main path and the branch outputs through addition
    added_output = Add()([block3_output, branch])
    
    # Flatten the combined output
    flattened_output = Flatten()(added_output)
    
    # Pass through a fully connected layer for classification
    final_output = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=final_output)
    
    return model