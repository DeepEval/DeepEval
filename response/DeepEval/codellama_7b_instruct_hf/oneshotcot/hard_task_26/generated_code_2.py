import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    branch1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(branch1)
    branch4 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(branch3)
    branch5 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch6 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(branch5)
    branch6 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch6)
    x = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6])
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model