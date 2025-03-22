import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    dropout = Dropout(0.5)(max_pooling)
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout)
    dropout = Dropout(0.5)(conv)
    flatten_layer = Flatten()(dropout)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dropout)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model