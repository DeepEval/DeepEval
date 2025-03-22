import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    depthwise_conv = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv)
    conv_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(depthwise_conv)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2)
    flatten_layer = Flatten()(max_pooling)
    dense = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model