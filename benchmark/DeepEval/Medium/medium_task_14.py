from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):

        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        normal = BatchNormalization()(conv)
        output_tensor = ReLU()(normal)

        return output_tensor

    main_path_1 = block(input_tensor=input_layer)
    main_path_2 = block(input_tensor=main_path_1)
    main_path_3 = block(input_tensor=main_path_2)
    
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)
    added = Add()([main_path_1,main_path_2,main_path_3,branch_path])

    dense = Dense(units=128, activation='relu')(added)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model