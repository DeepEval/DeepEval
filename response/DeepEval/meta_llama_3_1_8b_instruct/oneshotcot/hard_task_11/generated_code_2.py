import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_branch1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(parallel_branch)
    parallel_branch2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(parallel_branch)
    parallel_branch = Concatenate()([parallel_branch1, parallel_branch2])
    output_parallel = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(parallel_branch)
    fuse = Add()([main_path, output_parallel])
    bath_norm = BatchNormalization()(fuse)
    conv_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bath_norm)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_output)

    def block(input_tensor):

        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor
        
    block_output = block(max_pooling)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model