import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Elevate dimension
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Depthwise separable convolution
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        # Reduce dimension
        x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        # Add the input to the output
        output_tensor = Add()([input_tensor, x])
        return output_tensor

    # Create three branches using the same block
    branch1_output = block(input_layer)
    branch2_output = block(input_layer)
    branch3_output = block(input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])
    
    # Flatten the concatenated output
    flatten = Flatten()(concatenated)
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model