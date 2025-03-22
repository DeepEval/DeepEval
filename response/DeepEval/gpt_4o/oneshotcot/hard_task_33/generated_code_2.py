import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    def block(input_tensor):
        # Elevate dimension through a 1x1 convolution
        elevate = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # Extract features using a 3x3 depthwise separable convolution
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(elevate)
        # Reduce dimension through a 1x1 convolution
        reduce = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(depthwise)
        # Add the input to the output of the block
        output = Add()([reduce, input_tensor])
        return output
    
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Create three branches, each using the defined block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Add a fully connected layer to generate classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model