import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(conv3)
    flat = Flatten()(pool3)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)
    model = keras.Model(inputs=input_layer, outputs=output)
    return model