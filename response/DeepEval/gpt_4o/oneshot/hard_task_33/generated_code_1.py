import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    def block(input_tensor):
        # 1x1 convolution to elevate dimension
        elevated = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 3x3 depthwise separable convolution to extract features
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(elevated)
        
        # 1x1 convolution to reduce dimension
        reduced = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(depthwise)
        
        # Adding the input to the block's output
        output_tensor = Add()([input_tensor, reduced])
        
        return output_tensor

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Create three parallel branches using the block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    # Concatenate outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated result
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer to generate classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model