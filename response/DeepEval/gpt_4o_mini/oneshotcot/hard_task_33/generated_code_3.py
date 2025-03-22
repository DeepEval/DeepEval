import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Elevate dimension with 1x1 convolution
        elevated = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Apply depthwise separable convolution (3x3 depthwise)
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(elevated)
        
        # Reduce dimension with another 1x1 convolution
        reduced = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise)
        
        # Add the input tensor to the output tensor (shortcut connection)
        output_tensor = Add()([input_tensor, reduced])
        
        return output_tensor

    # Create three branches using the same block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model