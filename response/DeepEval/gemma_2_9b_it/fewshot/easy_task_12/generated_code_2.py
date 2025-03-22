import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Add, Flatten, Dense, DepthwiseConv2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    main_path_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(main_path_1)
    main_path_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path_1)

    main_path_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_1)
    main_path_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(main_path_2)
    main_path_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path_2)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_path_1)

    # Concatenate
    combined_path = Add()([main_path_2, branch_path])
    
    flatten_layer = Flatten()(combined_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model