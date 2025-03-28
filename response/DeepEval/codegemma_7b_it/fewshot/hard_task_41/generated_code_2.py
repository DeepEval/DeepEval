import keras
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, Dense, Reshape, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        flatten2 = Flatten()(maxpool2)
        flatten3 = Flatten()(maxpool3)
        dropout1 = Dropout(0.2)(flatten1)
        dropout2 = Dropout(0.2)(flatten2)
        dropout3 = Dropout(0.2)(flatten3)
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    def block_2(input_tensor):
        x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        x4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x4)
        output_tensor = Concatenate()([x1, x2, x3, x4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model