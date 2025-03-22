import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, SeparableConv2D, Activation, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        relu = Activation('relu')(conv)
        sep_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        concat_features = Concatenate(axis=-1)([input_tensor, sep_conv])
        return concat_features

    main_path = block(input_tensor)
    for _ in range(3):
        main_path = block(main_path)

    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    merged_features = Add()([main_path, branch_path])
    flatten_features = Flatten()(merged_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_features)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model