import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate

def block(input_tensor):
    # Increase the dimension using a 1x1 convolution
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
    
    # Apply depthwise separable convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Reduce the dimension with another 1x1 convolution
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    
    # Add the input to the output of the block (residual connection)
    output_tensor = Add()([input_tensor, x])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Create three branches using the defined block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the result
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model