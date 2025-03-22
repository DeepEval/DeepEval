import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_separable_conv2 = DepthwiseSeparableConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    dropout1 = Dropout(rate=0.2)(depthwise_separable_conv2)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(conv2)
    flatten_layer = Flatten()(dropout2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model