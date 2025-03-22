import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    # Define the block as described
    def block(input_tensor):
        # Elevate dimension with a 1x1 convolution
        conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Depthwise separable convolution 3x3
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1_1)
        
        # Reduce dimension with another 1x1 convolution
        conv1x1_2 = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        
        # Add the input tensor to the output to form the block's output
        block_output = Add()([input_tensor, conv1x1_2])
        
        return block_output
    
    # Create the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Create the three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    # Concatenate the outputs of the three branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated_branches)
    
    # Fully connected layer to produce the classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model