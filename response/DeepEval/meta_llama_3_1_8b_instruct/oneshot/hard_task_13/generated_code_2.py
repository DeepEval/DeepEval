import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    output_tensor = Concatenate()([branch1, branch2, branch3, branch4])

    # Apply batch normalization and relu activation
    bath_norm = BatchNormalization()(output_tensor)
    relu = keras.layers.Activation('relu')(bath_norm)

    def block_2(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        flatten = Flatten()(global_avg_pool)
        dense1 = Dense(units=64, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        weights = Dense(units=64, activation='linear')(dense2)
        reshaped_weights = Reshape((1, 1, 64))(weights)
        element_wise_product = Multiply()([input_tensor, reshaped_weights])
        output_tensor = element_wise_product
        return output_tensor

    block_2_output = block_2(relu)
    output_layer = Dense(units=10, activation='softmax')(block_2_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model