import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    main_input = Input(shape=(28, 28, 1))
    branch_input = Input(shape=(28, 28, 1))

    main_path_output = block(main_input)
    branch_path_output = block(branch_input)

    concat_output = Concatenate(axis=-1)([main_path_output, branch_path_output])

    batch_norm = BatchNormalization()(concat_output)
    flatten = Flatten()(batch_norm)

    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=[main_input, branch_input], outputs=output_layer)

    return model