import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = BatchNormalization()(path1)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = BatchNormalization()(path2)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = BatchNormalization()(path3)
        path4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    block_output = block(input_tensor=input_layer)
    block_output = block(input_tensor=block_output)
    block_output = block(input_tensor=block_output)
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model