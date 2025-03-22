from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        normal = BatchNormalization()(conv)
        main_path = ReLU()(normal)
        output_tensor = Add()([main_path, input_tensor])

        return output_tensor

    conv = block(input_tensor=initial_conv)
    outer_main_path = block(input_tensor=conv)
    outer_branch_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    added = Add()([outer_main_path, outer_branch_path])

    ave_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(added)
    dense = Dense(units=128, activation='relu')(ave_pool)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model