import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, pool])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    ga_pool = GlobalAveragePooling2D()(block1_output)

    dense1 = Dense(units=128, activation='relu')(ga_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    weights = Dense(units=64)(dense2)
    weights = Reshape(target_shape=(1, 1, 64))(weights)
    
    output = Multiply()([block1_output, weights])
    output = Dense(units=64, activation='relu')(output)
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model