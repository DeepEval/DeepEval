import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Conv2D as C2D
from keras.layers import MaxPooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_1 = C2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3_2 = C2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([path3_1, path3_2])
        
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([path4_1, path4_2])
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_1 = C2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_2 = C2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([path4_1, path4_2])

        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
    
    block_output = block(conv)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model