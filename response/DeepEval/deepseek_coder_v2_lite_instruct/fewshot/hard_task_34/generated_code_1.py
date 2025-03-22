import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, SeparableConv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    # Main path
    main_path = block(input_layer)
    main_path = block(main_path)
    main_path = block(main_path)
    main_path = block(main_path)

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Merge paths
    adding_layer = Add()([main_path, branch_path])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model