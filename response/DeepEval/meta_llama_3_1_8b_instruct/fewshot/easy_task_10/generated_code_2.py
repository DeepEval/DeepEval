import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Increase dimensionality with a 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract features with a 3x3 depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv1)

    # Reduce dimensionality with a 1x1 convolutional layer
    conv2 = Conv2D(filters=10, kernel_size=(1, 1), strides=(2, 2), padding='same')(depthwise_conv)

    # Output classification probabilities
    output_layer = Flatten()(conv2)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model