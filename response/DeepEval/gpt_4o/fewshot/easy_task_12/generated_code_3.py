import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense, Conv2D, ReLU

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path_block(input_tensor):
        x = ReLU()(input_tensor)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        return x

    main_path_1 = main_path_block(input_layer)
    main_path_2 = main_path_block(main_path_1)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Summing the outputs from both paths
    summed_paths = Add()([main_path_2, branch_path])

    # Final layers
    flatten_layer = Flatten()(summed_paths)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model