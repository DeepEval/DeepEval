import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # 1x1 convolutional layer to increase the depth
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 3x3 depthwise separable convolutional layer
        depthwise_conv3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1)
        
        # 1x1 convolutional layer to reduce the depth
        conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv3x3)
        
        # Add the block's input to the output
        output_tensor = keras.layers.add([input_tensor, conv1x1_2])
        
        return output_tensor

    # Apply the block to each of the three branches
    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)
    branch3 = block(input_tensor=input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    dense_layer = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model