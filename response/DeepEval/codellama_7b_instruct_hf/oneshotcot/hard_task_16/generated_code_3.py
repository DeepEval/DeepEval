import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))


    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Concatenate()([block1, block1, block1])


    transition_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)


    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(transition_conv)
    block2 = Flatten()(block2)
    block2 = Dense(units=64, activation='relu')(block2)
    block2 = Dense(units=10, activation='softmax')(block2)


    branch = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch)
    branch = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch)
    branch = Concatenate()([branch, branch])


    final_output = Concatenate()([block1, block2, branch])
    final_output = Flatten()(final_output)
    final_output = Dense(units=64, activation='relu')(final_output)
    final_output = Dense(units=10, activation='softmax')(final_output)


    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model