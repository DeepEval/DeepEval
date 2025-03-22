import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, SeparableConv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Separable Convolution
        separable_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return separable_conv

    # Main path with three blocks
    main_path_input = input_layer
    main_path_outputs = []
    for _ in range(3):
        main_path_output = block(main_path_input)
        main_path_outputs.append(main_path_output)
        main_path_input = Concatenate(axis=-1)([main_path_input, main_path_output])

    # Branch path
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Addition of main path and branch path
    added_output = Add()([main_path_outputs[-1], branch_path_output])

    # Flattening and fully connected layer
    flattened_output = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model