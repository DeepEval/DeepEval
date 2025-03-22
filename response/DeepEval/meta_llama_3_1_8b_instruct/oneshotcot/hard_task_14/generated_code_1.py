import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)

    def branch_path(input_tensor):
        path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return path
    
    branch_output = branch_path(input_tensor=max_pooling)
    branch_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_output)

    main_path = GlobalAveragePooling2D()(conv)
    main_path = Dense(units=32, activation='relu')(main_path)
    weights = Dense(units=32, activation='linear')(main_path)
    weights = keras.layers.Reshape((1, 1, 32))(weights)

    main_path = keras.layers.multiply([conv, weights])
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_path)
    main_path = Concatenate()([main_path, branch_output])

    bath_norm = BatchNormalization()(main_path)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model