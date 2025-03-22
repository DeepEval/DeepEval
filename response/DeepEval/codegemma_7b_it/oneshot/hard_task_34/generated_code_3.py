import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def block(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(conv)
        bath_norm = BatchNormalization()(conv)
        output_tensor = Add()([bath_norm, input_tensor])
        return output_tensor

    main_path_output = input_layer
    for _ in range(3):
        main_path_output = block(main_path_output)

    # Branch Path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    fuse_output = Add()([main_path_output, branch_path_output])

    # Classification
    flatten_layer = Flatten()(fuse_output)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model