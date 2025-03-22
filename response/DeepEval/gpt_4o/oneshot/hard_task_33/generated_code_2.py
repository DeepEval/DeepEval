import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        # 1x1 Convolution to elevate dimensions
        expand = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 Depthwise Separable Convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(expand)
        # 1x1 Convolution to reduce dimensions
        project = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        # Adding the input tensor to the output
        output_tensor = Add()([project, input_tensor])
        
        return output_tensor
    
    # Creating three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    # Concatenating the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flattening the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model