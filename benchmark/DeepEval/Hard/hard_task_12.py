from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 64))
        
    squeeze = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    expand1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(squeeze)
    expand2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(squeeze)
    
    main_path = Concatenate()([expand1, expand2])
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)
    added = Add()([main_path, branch_path])

    flatten_output = Flatten()(added)
    dense_1_output = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model