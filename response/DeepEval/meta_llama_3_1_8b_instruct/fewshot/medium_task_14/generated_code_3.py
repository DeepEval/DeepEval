import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = keras.layers.ReLU()(block1)

    block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    block2 = BatchNormalization()(block2)
    block2 = keras.layers.ReLU()(block2)

    block3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    block3 = BatchNormalization()(block3)
    block3 = keras.layers.ReLU()(block3)

    adding_layer = Add()([block1, block2, block3])

    parallel_branch = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    parallel_branch = BatchNormalization()(parallel_branch)
    parallel_branch = keras.layers.ReLU()(parallel_branch)

    adding_layer = Add()([adding_layer, parallel_branch])

    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model