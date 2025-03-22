import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def block(input_tensor):

        # Path 1: single 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: average pooling followed by a 1x1 convolution
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(path3)
        path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(path3)
        path3 = Concatenate()([path3, path3_1x3, path3_3x1])

        # Path 4: 1x1 convolution followed by a 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(path4)
        path4_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(path4)
        path4 = Concatenate()([path4, path4_1x3, path4_3x1])

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