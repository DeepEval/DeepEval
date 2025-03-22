import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Elevate dimension through 1x1 convolution
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Extract features through 3x3 depthwise separable convolution
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

        # Reduce dimension through 1x1 convolution
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)

        # Add block's input to the output
        output_tensor = Concatenate()([conv1, depthwise, conv2, input_tensor])

        return output_tensor

    # Apply block to each of the three branches
    branch1_output = block(input_tensor)
    branch2_output = block(branch1_output)
    branch3_output = block(branch2_output)

    # Concatenate outputs from all branches
    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Flatten and pass through fully connected layer
    flatten_layer = Flatten()(concat_output)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model