import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate, MaxPooling2D, Dense, AveragePooling2D, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def basic_block(input_tensor):

        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = Activation('relu')(main_path)

        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch = BatchNormalization()(branch)
        branch = Activation('relu')(branch)

        output_tensor = concatenate([main_path, branch])

        return output_tensor

    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)

    block1 = basic_block(input_tensor=initial_conv)
    block2 = basic_block(input_tensor=block1)

    branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2)
    branch = BatchNormalization()(branch)
    branch = Activation('relu')(branch)

    output = concatenate([block2, branch])
    output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(output)

    output = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(output)
    output = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model