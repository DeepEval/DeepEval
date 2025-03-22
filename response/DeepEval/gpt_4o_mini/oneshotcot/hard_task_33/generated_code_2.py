import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # 1x1 Convolution
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Depthwise Separable Convolution (3x3)
        x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
        # 1x1 Convolution to reduce the dimension
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        # Adding the input to the output (skip connection)
        output_tensor = Add()([x, input_tensor])
        return output_tensor

    # Three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the result
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model