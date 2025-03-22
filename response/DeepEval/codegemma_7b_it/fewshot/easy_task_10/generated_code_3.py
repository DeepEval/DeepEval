import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(conv2)

    flatten_layer = Flatten()(conv3)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model