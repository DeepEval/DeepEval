import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate

def block(input_tensor):
    # Elevate dimension using 1x1 convolution
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
    # Apply depthwise separable convolution (3x3)
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    # Reduce dimension using 1x1 convolution
    x = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(x)
    # Add the input to the output (skip connection)
    output_tensor = Add()([x, input_tensor])
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
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

# Example of how to create the model
model = dl_model()
model.summary()