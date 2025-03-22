import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape
from keras.layers import AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):

        path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor
        
    block1_output = block1(input_layer)
    flatten_layer = Flatten()(block1_output)
    dropout1 = Dropout(0.2)(flatten_layer)
    reshape_layer = Reshape((3, 3, 3))(dropout1)
    dense1 = Dense(units=128, activation='relu')(reshape_layer)

    def block2(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=3, padding='same')(input_tensor))
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
        
    block2_output = block2(dense1)
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    dense2 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model