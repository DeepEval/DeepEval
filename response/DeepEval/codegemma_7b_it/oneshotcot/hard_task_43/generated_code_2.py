import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, AveragePooling2D, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block_one(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor
    
    block_one_output = block_one(max_pooling)
    bath_norm_1 = BatchNormalization()(block_one_output)
    flatten_layer_1 = Flatten()(bath_norm_1)
    dense1 = Dense(units=128, activation='relu')(flatten_layer_1)
    reshape_layer = Reshape((1, 1, 128))(dense1)

    def block_two(input_tensor):
        path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2_1 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2_2 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2_1)
        path2_3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2_2)
        path3 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2_3, path3])

        return output_tensor
    
    block_two_output = block_two(reshape_layer)
    bath_norm_2 = BatchNormalization()(block_two_output)
    flatten_layer_2 = Flatten()(bath_norm_2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model