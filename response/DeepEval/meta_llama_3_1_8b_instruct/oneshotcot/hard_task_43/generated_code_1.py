import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape
from keras import backend as K

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):

        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        flatten_layer = Flatten()(output_tensor)
        return flatten_layer

    flatten_layer = block1(input_layer)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    reshape_layer = Reshape((128,))(dense1)
    block2_output = reshape_layer

    def block2(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path7 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4, path5, path6, path7])
        batch_norm = BatchNormalization()(output_tensor)
        flatten_layer = Flatten()(batch_norm)
        return flatten_layer

    block2_output = block2(Reshape((1, 1, 128))(block2_output))

    dense3 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model