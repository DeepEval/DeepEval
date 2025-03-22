import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)

    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(transition_conv)
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    main_path_output = Concatenate()([block1, branch])

    fc1 = Dense(units=64, activation='relu')(main_path_output)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    model = keras.Model(inputs=input_layer, outputs=fc2)

    return model