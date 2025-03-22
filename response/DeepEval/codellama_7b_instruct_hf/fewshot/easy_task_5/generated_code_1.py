import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)
    flatten = Flatten()(pool)
    dense = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=dense)
    return model