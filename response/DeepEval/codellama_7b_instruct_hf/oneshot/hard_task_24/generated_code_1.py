import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch3)

    concat_layer = Concatenate()([branch1, branch2, branch3])
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model