import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # First 1x1 convolutional layer to increase the dimension
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 3x3 depthwise separable convolutional layer
        path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        
        # Second 1x1 convolutional layer to reduce the dimension
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Add the block's input to the output
        output_tensor = Add()([input_tensor, path3])
        
        return output_tensor

    # Apply the block to each of the three branches
    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)
    branch3 = block(input_tensor=input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer to generate classification probabilities
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model