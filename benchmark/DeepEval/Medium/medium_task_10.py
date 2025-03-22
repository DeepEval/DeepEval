from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense  


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    def block(input_tensor):
        
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        normal = BatchNormalization()(conv)
        main_path = ReLU()(normal)
        
        output_tensor = Add()([main_path, input_tensor])

        return output_tensor
    
    middle_main_path_1 = block(input_tensor=initial_conv)
    middle_branch_path_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)
    added_1 = Add()([middle_main_path_1, middle_branch_path_1])

    middle_main_path_2 = block(input_tensor=added_1)
    middle_branch_path_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(added_1)

    added_2 = Add()([middle_main_path_2, middle_branch_path_2])

    global_branch_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)
    added = Add()([added_2, global_branch_path])

    output_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(added)
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
