import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    def branch_2(input_tensor):
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
        upsample = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return upsample

    def branch_3(input_tensor):
        max_pooling = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
        upsample = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same')(conv)
        return upsample

    main_path = branch_1(input_layer)
    main_path = Concatenate()([main_path, branch_2(input_layer), branch_3(input_layer)])
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    adding_layer = Add()([conv, branch_path])

    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model