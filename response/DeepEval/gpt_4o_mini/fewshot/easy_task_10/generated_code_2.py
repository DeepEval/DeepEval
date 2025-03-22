import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Step 1: 1x1 Convolution to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Step 2: 3x3 Depthwise Separable Convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)

    # Step 3: Another 1x1 Convolution to reduce dimensionality
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(depthwise_conv)

    # Flatten the output for the fully connected layer
    flatten = Flatten()(conv2)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model