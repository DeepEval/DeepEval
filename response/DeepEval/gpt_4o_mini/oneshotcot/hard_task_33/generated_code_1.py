import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Add, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Elevate dimensions through a 1x1 convolution
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Extract features through a 3x3 depthwise separable convolution
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
        
        # Reduce dimensions through a 1x1 convolution
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        
        # Add the input to the output (skip connection)
        output_tensor = Add()([input_tensor, x])
        
        return output_tensor

    # Create three branches using the defined block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer to generate classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model