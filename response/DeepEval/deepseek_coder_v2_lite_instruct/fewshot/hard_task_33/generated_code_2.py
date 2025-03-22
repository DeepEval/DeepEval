import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # First 1x1 convolutional layer to increase the dimension
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 depthwise separable convolutional layer
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        # Second 1x1 convolutional layer to reduce the dimension
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        # Add the block's input to the output of the 3x3 depthwise separable convolutional layer
        output_tensor = Add()([conv3, input_tensor])
        return output_tensor

    # Apply the block to each of the three branches
    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)
    branch3 = block(input_tensor=input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Add()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model