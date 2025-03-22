import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    flatten = Flatten()(conv2)
    dense = Dense(10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=dense)
    return model