import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, AveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def basic_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        branch = input_tensor
        adding_layer = Add()([batch_norm, branch])
        return adding_layer

    def level_1(input_tensor):
        output_tensor = basic_block(input_tensor)
        return output_tensor

    def level_2(input_tensor):
        branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = basic_block(input_tensor)
        adding_layer = Add()([output_tensor, branch])
        return adding_layer

    def level_3(input_tensor):
        global_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = level_2(input_tensor)
        adding_layer = Add()([output_tensor, global_branch])
        return adding_layer

    level1_output = level_1(input_tensor=input_layer)
    level2_output = level_2(input_tensor=level1_output)
    level3_output = level_3(input_tensor=level2_output)

    pool = AveragePooling2D(pool_size=(8, 8), strides=1)(level3_output)
    flatten = Flatten()(pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model