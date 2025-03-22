import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Elevate the dimension
        conv1x1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1_1)
        
        # Reduce dimension
        conv1x1_2 = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        
        # Add the input to the output
        output_tensor = Add()([input_tensor, conv1x1_2])
        return output_tensor
    
    # Three branches with the same block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and final dense layer for classification
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model