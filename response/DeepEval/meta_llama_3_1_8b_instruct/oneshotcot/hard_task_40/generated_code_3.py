import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Dropout

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    concat_output = Concatenate()([pool1, pool2, pool3])
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    reshape_layer = Reshape((128, 1, 1))(dense1)

    # Second Block
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.2)(path1)

        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Dropout(0.2)(path2)

        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Dropout(0.2)(path3)

        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(path4)
        path4 = Dropout(0.2)(path4)

        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block_output = block(reshape_layer)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model