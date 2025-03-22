import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Elevate the dimension
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # Depthwise separable convolution
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
        # Reduce the dimension
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        # Adding the input to the output of the block
        output_tensor = Add()([input_tensor, x])

        return output_tensor
    
    # Creating three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate the outputs from the branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model