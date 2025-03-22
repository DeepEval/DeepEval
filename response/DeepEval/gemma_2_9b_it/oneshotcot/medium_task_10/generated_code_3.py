import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)

        output_tensor = Add()([main_path, branch])
        return output_tensor

    x = basic_block(x)

    def residual_block(input_tensor):
        main_path = basic_block(input_tensor)
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Add()([main_path, branch])
        return output_tensor

    x = residual_block(x)
    x = residual_block(x)

    global_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Add()([global_branch, x])

    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model