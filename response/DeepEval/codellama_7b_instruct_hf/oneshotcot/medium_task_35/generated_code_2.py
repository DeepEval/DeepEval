import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    dropout = Dropout(rate=0.5)(max_pooling2)
    flatten = Flatten()(dropout)
    dense = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=dense)
    return model