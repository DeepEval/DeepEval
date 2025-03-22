import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Main path operations
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dw_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv3x3)
        
        # Branch path operations
        dw_conv3x3_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv3x3_branch)
        
        # Concatenate features along the channel dimension
        concatenated = Concatenate(axis=-1)([conv1x1_2, conv1x1_branch])
        
        return concatenated

    def block_2(input_tensor):
        # Reshape the input tensor into groups
        reshaped = Reshape(target_shape=(input_tensor.shape[1], input_tensor.shape[2], 4, -1))(input_tensor)
        
        # Permute the dimensions to swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to the original shape for channel shuffling
        final_shape = (input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3] * input_tensor.shape[4])
        reshaped_back = Reshape(target_shape=final_shape)(permuted)
        
        return reshaped_back

    # Apply Block 1 to the input layer
    block1_output = block_1(input_tensor=input_layer)
    
    # Apply Block 2 to the output of Block 1
    block2_output = block_2(input_tensor=block1_output)
    
    # Flatten the output of Block 2
    flattened = Flatten()(block2_output)
    
    # Pass through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model