import keras
from keras.layers import Input, DepthwiseSeparableConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    depthwise_conv = DepthwiseSeparableConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(depthwise_conv)
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    dropout = Dropout(rate=0.2)(conv)
    flatten = Flatten()(dropout)
    dense = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense)
    model = keras.Model(inputs=input_layer, outputs=output)
    return model