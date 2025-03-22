import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2])

        return output_tensor
    
    block_output = block(input_tensor=max_pooling)
    bath_norm = BatchNormalization()(block_output)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(bath_norm)

    def branch_path(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)
        return path2

    branch_output = branch_path(input_tensor=max_pooling2)

    add_output = Add()([max_pooling2, branch_output])
    flatten_layer = Flatten()(add_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model