import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=input_tensor.shape[3], activation='sigmoid')(dense1)
        reshape_layer = Reshape((1, 1, input_tensor.shape[3]))(dense2)
        element_wise_mul = Multiply()([input_tensor, reshape_layer])
        output_tensor = element_wise_mul

        return output_tensor
    
    block_output_1 = block(input_tensor=max_pooling)
    block_output_2 = block(input_tensor=max_pooling)
    concat_layer = Concatenate()([block_output_1, block_output_2])
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model