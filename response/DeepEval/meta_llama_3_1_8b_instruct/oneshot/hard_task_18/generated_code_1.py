import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    main_path = avg_pool
    block_output = conv1
    add_layer = Add()([main_path, block_output])
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(add_layer)

    def second_block(input_tensor):

        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(global_avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_tensor = Multiply()([Reshape((64,))(global_avg_pool), dense2])
        output_tensor = Reshape((1, 1, 64))(output_tensor)
        output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
        output_tensor = Flatten()(output_tensor)
        output_layer = Dense(units=10, activation='softmax')(output_tensor)

        return output_layer

    output_layer = second_block(avg_pool_2)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model