import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # 1x1 Convolution to increase dimensions
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 Depthwise Convolution
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        # 1x1 Convolution to decrease dimensions
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        # Add the block's input to the output
        output_tensor = Add()([input_tensor, conv3])
        return output_tensor

    # Apply the block to each branch
    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)
    branch3 = block(input_tensor=input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Add()([branch1, branch2, branch3])
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model