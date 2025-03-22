import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Increase dimensionality with a 1x1 convolution
    conv1x1_increase = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Depthwise separable convolution (3x3)
    depthwise_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_increase)

    # Reduce dimensionality with another 1x1 convolution
    conv1x1_reduce = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)

    # Add the result to the original input layer
    add_layer = Add()([conv1x1_reduce, input_layer])

    # Flatten the output
    flatten_layer = Flatten()(add_layer)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model