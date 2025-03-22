import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten1 = Flatten()(path1)
        flatten2 = Flatten()(path2)
        flatten3 = Flatten()(path3)
        dropout1 = Dropout(0.2)(flatten1)
        dropout2 = Dropout(0.2)(flatten2)
        dropout3 = Dropout(0.2)(flatten3)
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    block1_output = block_1(input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)

    def block_2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        output_tensor = Concatenate(axis=3)([path1, path2, path3, path4])
        return output_tensor

    block2_output = block_2(reshaped)
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model