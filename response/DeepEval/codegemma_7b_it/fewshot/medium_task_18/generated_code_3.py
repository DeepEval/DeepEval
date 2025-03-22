import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32,32,3))

    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv3)

    conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv4)

    conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool4)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv5)

    flatten = Flatten()(pool5)
    dense1 = Dense(units=2048, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model