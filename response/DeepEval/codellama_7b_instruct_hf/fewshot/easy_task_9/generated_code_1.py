import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    conv1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv2 = DepthwiseConv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv2)
    add = Add()([input_layer, conv3])

    flatten = Flatten()(add)
    output_layer = Dense(10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model