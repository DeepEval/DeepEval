import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    def block(input_tensor):
        # Elevate dimension
        path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Depthwise separable convolution
        path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path)
        # Reduce dimension
        path = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path)
        # Add input tensor to output of the block
        output_tensor = Add()([input_tensor, path])
        return output_tensor
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = block(input_layer)
    
    # Branch 2
    branch2 = block(input_layer)
    
    # Branch 3
    branch3 = block(input_layer)
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model