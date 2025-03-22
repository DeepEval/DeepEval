import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, UpSampling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_initial = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Main Path
    def main_path(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = UpSampling2D(size=(2, 2))(path2)
        path3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = UpSampling2D(size=(2, 2))(path3)
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    main_output = main_path(conv_initial)
    main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_output)

    # Branch Path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_initial)

    # Add Outputs from Main and Branch Paths
    added_output = Add()([main_output, branch_output])

    # Flatten and Dense Layers
    bath_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model