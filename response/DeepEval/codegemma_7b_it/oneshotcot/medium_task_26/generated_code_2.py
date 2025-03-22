import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 64))
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2])

        return output_tensor
        
    block_output = block(input_tensor=conv)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=2, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model