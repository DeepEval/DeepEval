from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)

    def block(input_tensor):

        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        conv = BatchNormalization()(conv)
        main_path = ReLU()(conv)

        return main_path
    
    main_path_1 = block(input_tensor=initial_conv)
    main_path_2 = block(input_tensor=initial_conv)
    main_path_3 = block(input_tensor=initial_conv)

    added = Add()([main_path_1,main_path_2,main_path_3,initial_conv])

    flatten = Flatten()(added)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

