import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Flatten, Dense

def dl_model():
    
    # Input layer with shape (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolutional Layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)

    # 3x3 Depthwise Separable Convolutional Layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)

    # 1x1 Convolutional Layer to reduce dimensionality
    conv3 = Conv2D(filters=10, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(conv2)

    # Flatten the output
    flatten_layer = Flatten()(conv3)

    # Output through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create a Keras Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model