import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv_1)

    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
        
    block_output = block(input_tensor=max_pooling_1)
    bath_norm = BatchNormalization()(block_output)
    max_pooling_2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(bath_norm)

    conv_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_2)
    bath_norm_2 = BatchNormalization()(conv_2)
    max_pooling_3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(bath_norm_2)

    flatten_layer = Flatten()(max_pooling_3)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model