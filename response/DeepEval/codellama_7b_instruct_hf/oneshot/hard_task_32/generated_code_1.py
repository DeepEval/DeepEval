import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)
    branch1 = Dropout(rate=0.2)(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Dropout(rate=0.2)(branch1)

    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)
    branch2 = Dropout(rate=0.2)(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(rate=0.2)(branch2)

    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch3)
    branch3 = Dropout(rate=0.2)(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Dropout(rate=0.2)(branch3)

    concatenated = Concatenate()([branch1, branch2, branch3])
    flattened = Flatten()(concatenated)
    fc1 = Dense(units=128, activation='relu')(flattened)
    fc1 = Dropout(rate=0.2)(fc1)
    fc2 = Dense(units=64, activation='relu')(fc1)
    fc2 = Dropout(rate=0.2)(fc2)
    output_layer = Dense(units=10, activation='softmax')(fc2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model