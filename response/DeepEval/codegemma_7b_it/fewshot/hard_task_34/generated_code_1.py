import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Activation, SeparableConv2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):

        conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        relu = Activation('relu')(conv)
        concat = Concatenate(axis=3)([input_tensor, relu])

        return concat
    
    main_path = block(input_tensor=input_layer)
    main_path = block(input_tensor=main_path)
    main_path = block(input_tensor=main_path)

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    merged = keras.layers.add([main_path, branch_path])
    flatten_layer = Flatten()(merged)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model