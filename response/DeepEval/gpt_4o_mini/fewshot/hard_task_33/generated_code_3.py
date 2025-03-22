import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # 1x1 Convolution to elevate dimension
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 Depthwise Separable Convolution for feature extraction
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        # 1x1 Convolution to reduce dimension
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        # Adding the input back to the output
        output_tensor = Add()([conv2, input_tensor])
        return output_tensor

    # Creating three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model